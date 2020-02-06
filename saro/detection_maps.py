import numpy as np
from scipy.stats import chi2,ncx2
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import xara
from scipy.linalg import sqrtm,pinv



def residual_pdet(lamb, xsi, rank, targ):
    """
    Computes the residual of Pdet
    lamb     : The noncentrality parameter representing the feature
    targ     : The target Pdet to converge to
    xsi      : The location of threshold
    rang     : The rank of observable
    Returns the Pdet difference
    """
    respdet = 1 - ncx2.cdf(xsi,rank,lamb) - targ
    return respdet

    
def build_grid(sz, pscale):
    """
    Builds a grid a cartesian grid of the parameters of a companion
    sz    : The number of pixels grid will be (sz,sz)
    pscale: The scale in mas/pix for the grid
    
    returns: an array of parameters [rho, theta, c]
    """
    xs, ys = np.meshgrid(np.arange(sz), np.arange(sz))
    ds = np.round(sz/2)
    cpform = ((xs-ds)*pscale).flatten() + 1j*((ys-ds)*pscale).flatten()
    rhos, thetas = np.abs(cpform), np.angle(cpform)*180./np.pi 
    params = np.array([rhos, thetas, 1000*np.ones_like(rhos)]).T
    return params


def get_contrast(params, thekpo, pfa, pdet, wl, W=None, dtheta=None):
    if W is None:
        print("No whitening matrix provided: building it from the kpo.kp_cov")
        try:
            kpcov = thekpo.kp_cov
        except:
            print("You must provide a covariance matrix")
            return 1.
        kernelcov = thekpo.kpi.KPM.dot(thekpo.kp_cov).dot(thekpo.kpi.KPM.T)
        W = sqrtm(np.linalg.inv(kernelcov))
    else:
        print("Using the provided whitening matrix")
    WK = W.dot(thekpo.kpi.KPM)
    
    nbkp = thekpo.kpi.nbkp
    xsi = chi2.ppf(1.-pfa, nbkp)
    c0 = 1.e3 #Fixing reference contrast
    
    lambda0 = 0.2**2 * nbkp
    
    conts = []
    if dtheta is None:
        
        for p in params:
            p[2] = c0
            phi = np.angle(thekpo.get_cvis(p, wl))
            xu = c0 * WK.dot(phi)  #Hypothetical unitary contrast signature
            sol = leastsq(residual_pdet, lambda0,args=(xsi,nbkp, pdet))
            lamb = sol[0][0]
            contrast = np.sqrt( np.sum(xu.dot(xu)) / lamb )
            conts.append(contrast)
    else:
        print("Building an ADK detection map")
        for p in params:
            p[2] = c0
            adk = thekpo.get_adk_signature(p, dtheta, verbose=False)
            xu = c0 * W.dot(adk)  #Hypothetical unitary contrast signature
            sol = leastsq(residual_pdet, lambda0,args=(xsi,nbkp, pdet))
            lamb = sol[0][0]
            contrast = np.sqrt( np.sum(xu.dot(xu)) / lamb )
            conts.append(contrast)
    conts = np.array(conts)
    return conts


def get_sadk_signature(self, params, wl, verbose=False):
    """Returns the theoretical ADK signature of a binary signal
    
    The result is given (end - start)
    Parameters:
    -----------
    - params    : An array of parameters
    - wl
    - verbose   : If true it says what parameters are subtracted to what
    
    """
    
    kappa = np.array([self.get_kernel_signature(params[i,:], wl) for i in range(params.shape[0])])
    if verbose:
        print("Obtaining kerneks signatures for ", params )
    return kappa


def get_sadk_contrast_cpu(self, params, pfa, pdet, wl, W=None, dthetas=None):
    """
    The goto function to derive the attainable contrast from a covariance matrix.
    This version is suited to both ADK and classical calibration.
    params : An array of parameter vectors
    Pfa    : False Alarm Rate
    Pdet   : Detection probability
    wl     : The wavelength
    W      : The whitening matrix to use in single frames
    dthetas: The detector position angles (for ADK, or long series)
    """
    try :
        nf = dthetas.shape[0]
    except :
        nf = 1
        print("You must provide an array of the detector position angles")
    if W is not None:
        print("You should pass the whitening matrix through kpo.WlLp")
        return 1
    else:
        print("Using the provided whitening matrix")
    
    #rank of the chi2 distribution 
    therank = self.WlLp.shape[0]
    #Finding xsi, the detection threshold
    xsi = chi2.ppf(1.-pfa, therank)
    c0 = 1.e3 #Fixing reference contrast
    
    #This is just a starting point that seems to work reliably
    lambda0 = 0.2**2 * therank
    print("lambda0",("xsi","therank", "pdet"))
    print(lambda0,(xsi,self.WlLp.shape[0], pdet))
    #Using linearity: we need to compute lambda only once
    sol = leastsq(residual_pdet, lambda0,args=(xsi,therank, pdet))
    #lamb is the noncentrality parameter for the requested Pdet
    lamb = sol[0][0]
    
    print("lambda = %.2f"%(lamb))
    conts = []
    if dthetas is None:
        
        for p in params:
            p[2] = c0
            phi = np.angle(self.get_cvis(p))
            xu = c0 * self.kpi.KPM.dot(phi)  #Hypothetical unitary contrast signature
            contrast = np.sqrt( np.sum(xu.T.dot(xu)) / lamb )
            conts.append(contrast)
    else:
        print("Building an ADK detection map")
        blank = np.zeros_like(params[0])
        adks = []#All the individual signals for the map
        for p in params:
            p[2] = c0
            pp = np.array([p + (blank + np.array([0,ang,0])) for ang in dthetas])
            sigs = self.get_sadk_signature(pp, verbose=False)
 
            adks.append(sigs.flatten())
        adks = np.vstack(adks)
        xu = c0 * self.WlLp.dot(adks.T)#Hypothetical unitary contrast signature 
        for i in range(adks.shape[0]):
            #here we exploit lambda = sum(mu^2) = sum((xu/c0)^2) = 1/c0 * lambda_u
            #therefore c = sqrt(lambda_u / lambda)
            contrast = np.sqrt( np.sum(xu.T[i].T.dot(xu.T[i])) / lamb )
            conts.append(contrast)
    conts = np.array(conts)
    return conts

xara.KPO.get_sadk_contrast_cpu = get_sadk_contrast_cpu
xara.KPO.get_sadk_signature = get_sadk_signature

def plot_Pdet(cov, model, name="Detection map", seps=np.linspace(0,500,200), angles=np.linspace(0,360,200), mags=np.linspace(1,12,200), mode="mag", quiet=False):
    W = sqrtm(np.linalg.inv(cov))
    WK=W.dot(model.kpi.KPM)
    # In this case, we are looking fo the output of the fit of the kernal phases against a model
    
    #seps = np.linspace(100,1000,200)
    #angles=np.linspace(0,360,200)
    #mags= np.linspace(10,12,200)
    ctr	= 10**(mags/2.5)
    
    
    # Get separation, contrast, PA and contrast in a single convenient array
    sc=np.array(np.meshgrid(seps,ctr))
    if mode == "mag":
        scdisp=np.array(np.meshgrid(seps,mags))
    else :
        scdisp=np.array(np.meshgrid(seps,ctr))
    
    u,v=model.kpi.uv[:,0],model.kpi.uv[:,1]
    # Contrast / Separation map at PA=0 degs
    sac=np.zeros((3,200**2))
    sac[0]=sc.reshape((2,200**2))[0]
    sac[2]=sc.reshape((2,200**2))[1]
    
    # Compute the signatures for the desired contrast and separations 
    vis_obj=np.array([xara.core.cvis_binary(u,v,wl,i) for i in sac.T])
    phis=np.angle(vis_obj)
    
    lbds=np.array([np.sum(np.dot(WK,i)**2) for i in phis])
    
    
    # From the lambdas, and the rank of the matrix K, get the detection likelihoods 
    Pdet=get_pdet(rank=model.kpi.KPM.shape[0],lbd=lbds,Pfa=1e-2).reshape((200,200))
    Pdet[np.isnan(Pdet)]=0.   			# transform the NaNs into zeros
    
    # Plot the detection map
    if not quiet:
        fig,ax=plt.subplots(1,1,figsize=(9,9))
        
        CSS	 = ax.contour(scdisp[0], scdisp[1], Pdet  ,levels=[.68,.95,.997],label="Independant K-phases",cmap='Reds_r',vmin=.65,vmax=1.3)
        ax.set_ylabel('Contrast (magnitude)')
        ax.set_xlabel('Separation (mas)')
        ax.clabel(CSS, inline=1, fontsize=10)
        ax.grid()
        plt.title(name)
    return Pdet, scdisp
    
    
#chi2s, thekpo = mykernelopt.evaluate(pupilmask*1)

    
