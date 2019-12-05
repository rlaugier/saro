gpu = False
import numpy as np
if not gpu:
    import numpy as cp
else:
    import cupy as cp
import xara

from kernel_cm import br, bo, vdg, bbr

import xaosim
from tqdm import  tqdm
import matplotlib.pyplot as plt

import scipy.sparse as sparse
from scipy.linalg import sqrtm

from xara import fft, ifft, shift

import detection_maps as dmap
"""
The functions uses the wavelength provided by KPO.CWAVEL
"""


# Utility for phase optimization
def distance(yy, xx, yx):
    """
    Returns a map of the distance to a point yx
    Parameters:
    -----------
    yy         : map of y coordinates
    xx         : map of x coordinates
    yx         : y,x coordinates of the point of interest
    
    """
    dist = np.sqrt((xx-yx[1])**2 + (yy-yx[0])**2)
    return dist

def get_cvis(self, params):
    """
    Just a macro that returns complex visibilities for the parameter
    Parameters:
    -----------
    - params   : A parameter vector for a binary model
                    - p[0] = sep (mas)
                    - p[1] = PA (deg) E of N.
                    - p[2] = contrast ratio (primary/secondary)
                    optional:
                    - p[3] = angular size of primary (mas)
                    - p[4] = angular size of secondary (mas)
    """
    u = self.kpi.UVC[:,0]
    v = self.kpi.UVC[:,1]
    return xara.cvis_binary(u,v, self.CWAVEL, params, 0)
xara.KPO.get_cvis = get_cvis
    
def get_kernel_signature(self, params):
    """Returns the theoretical kernel signature of a binary signal
    
    The result is given (end - start)
    Parameters:
    -----------
    - params   : A parameter vector for a binary model
                    - p[0] = sep (mas)
                    - p[1] = PA (deg) E of N.
                    - p[2] = contrast ratio (primary/secondary)
                    optional:
                    - p[3] = angular size of primary (mas)
                    - p[4] = angular size of secondary (mas)
    """
    cvis = self.get_cvis(params)
    Phi = np.angle(cvis)
    kappa = self.kpi.KPM.dot(Phi)
    return kappa
xara.KPO.get_kernel_signature = get_kernel_signature

    
def get_sadk_signature(self, params, verbose=False):
    """Returns a series of kernel signatures 
    The result is given (end - start)
    Parameters:
    -----------
    - params    : An array of parameters
                    - p[0] = sep (mas)
                    - p[1] = PA (deg) E of N.
                    - p[2] = contrast ratio (primary/secondary)
                    optional:
                    - p[3] = angular size of primary (mas)
                    - p[4] = angular size of secondary (mas)

    - verbose   : If true it says what parameters are subtracted to what
    
    """
    
    kappa = np.array([self.get_kernel_signature(params[i,:]) for i in range(params.shape[0])])
    if verbose:
        print("Obtaining kerneks signatures for ", params )
    return kappa
xara.KPO.get_sadk_signature = get_sadk_signature


def whitened_kpd_binary_match_map(self, gsz, gstep, kp_signal,W=None, cref=1000, full_output=False):
    """ Produces a 2D-map showing where the best binary fit occurs

    Computes the dot product between the kp_signal and a grid (x,y) grid of 
    possible positions for the companion, for a pre-set contrast.

    Parameters:
    ----------
    - gsz       : grid size (gsz x gsz)
    - gstep     : grid step in mas
    - kp_signal : the kernel-phase vector
    - cref      : reference contrast (optional, default = 1000)
    - W         : a whitening matrix for the data (if not provided, will use kp_cov to build one)

    Remarks:
    -------
    In the high-contrast regime, the amplitude is inversely 
    proportional to the contrast.
    ---------------------------------------------------------------
    """
    if W is None:
        print("No whitening matrix provided: building it from the kpo.kp_cov")
        try:
            kpcov = self.kp_cov
        except:
            print("You must provide a covariance matrix")
            return 1.
        kernelcov = self.kpi.KPM.dot(self.kp_cov).dot(self.kpi.KPM.T)
        W = sqrtm(np.linalg.inv(kernelcov))
    mgrid = np.zeros((gsz, gsz))

    cvis = 1.0 + cref * xara.grid_precalc_aux_cvis(
        self.kpi.UVC[:,0],
        self.kpi.UVC[:,1],
        self.CWAVEL, mgrid, gstep)

    kpmap = self.kpi.KPM.dot(np.angle(cvis))
    crit  = W.dot(kpmap).T.dot(W.dot(kp_signal)) / np.linalg.norm(W.dot(kpmap), axis=0)
    print("kpmap shape", kpmap.shape)
    print("kp_signal shape", kp_signal.shape)
    if not full_output:
        return(crit.reshape(gsz, gsz) )
    else:
        return(crit.reshape(gsz,gsz), np.linalg.norm(W.dot(kpmap), axis=0).reshape(gsz,gsz),
               rhos.reshape(gsz,gsz), thetas.reshape(gsz,gsz))
xara.KPO.whitened_kpd_binary_match_map = whitened_kpd_binary_match_map

    
def whitened_adk_binary_match_map(self, gsz, gstep, adk_signal,W=None,dtheta=None, cref=1000, full_output=False):
    """ Produces a 2D-map showing where the best binary fit occurs

    Computes the dot product between the kp_signal and a grid (x,y) grid of 
    possible positions for the companion, for a pre-set contrast.

    Parameters:
    ----------
    - gsz       : grid size (gsz x gsz)
    - gstep     : grid step in mas
    - adk_signal : the kernel-phase vector, kappa2-kappa1
    - cref      : reference contrast (optional, default = 1000)
    - W         : a whitening matrix for the data (if not provided, will use kp_cov to build one)
    - dtheta    : the field rotation angle from first to last

    Remarks:
    -------
    In the high-contrast regime, the amplitude is proportional to the
    companion brightness ratio.
    ---------------------------------------------------------------
    """
    if dtheta is None:
        print("Error: need a rotation angle")
        return 1
    if W is None:
        print("No whitening matrix provided: building it from the kpo.kp_cov")
        try:
            kpcov = self.kp_cov
        except:
            print("You must provide a covariance matrix")
            return 1.
        kernelcov = self.kpi.KPM.dot(self.kp_cov).dot(self.kpi.KPM.T)
        W = sqrtm(np.linalg.inv(kernelcov))
    wadk_signal = W.dot(adk_signal)
    #building a grid
    xs, ys = np.meshgrid(np.arange(gsz), np.arange(gsz))
    ds = np.round(gsz/2)
    cpform = ((ys-ds)*gstep).flatten() + 1j*((-(xs-ds))*gstep).flatten()
    rhos, thetas = np.abs(cpform), np.angle(cpform)*180./np.pi
    params = np.array([rhos, thetas, cref*np.ones_like(rhos)]).T
    
    wsigs = np.array([W.dot(self.get_adk_signature(p, wl, dtheta, verbose=False)) for p in params]).T
    #print("wsigs shape", wsigs.shape)
    #print("adk_signal shape", adk_signal.shape)
    crit  = (wsigs.T.dot(wadk_signal)) / np.linalg.norm(wsigs, axis=0)
    if not full_output:
        return(crit.reshape(gsz, gsz))
    else :
        return(crit.reshape(gsz,gsz), np.linalg.norm(wsigs, axis=0).reshape(gsz,gsz),
               rhos.reshape(gsz,gsz), thetas.reshape(gsz,gsz))
xara.KPO.whitened_adk_binary_match_map = whitened_adk_binary_match_map


    
def build_L_matrix(self, nf, crop=True, giveUf=True):
    Uf = sparse.vstack([sparse.identity(self.kpi.nbkp) for i in range(nf)])
    L = (sparse.identity(Uf.shape[0]) - 1/nf * Uf.dot(Uf.T))
    if crop:
        L = L[:-self.kpi.nbkp,:]
    if giveUf:
        return L, Uf
    else :
        return L
xara.KPO.build_L_matrix = build_L_matrix


def get_adk_residual(self, params, y, dtheta):
    pparams = np.array([np.array([params["rho"],params["theta"],params["contrast"]]) + np.array([0, theta, 0]) for theta in dtheta])
    kappa = get_sadk_signature(self, pparams, verbose=False)
    residual = self.Mp.dot(kappa.flatten()) - y
    return residual
def get_kpd_residual(self, params, y, W, wl, dtheta):
    kappa = get_sadk_signature(self, np.array([params["rho"],params["theta"],params["contrast"]]), dtheta, verbose=False)
    residual = W.dot(kappa) - y
    return residual

xara.KPO.get_adk_residual = get_adk_residual






def gpu_sadk_binary_match_map(self, gsz, gstep, adk_signalg, W=None,deltas=None,
                                   verbose=False, project = True, cref=1000, full_output=False,
                                   thetype=cp.float32):
    """ Produces a 2D-map showing where the best binary fit occurs

    Computes the dot product between the kp_signal and a grid (x,y) grid of 
    possible positions for the companion, for a pre-set contrast.

    Parameters:
    ----------
    - gsz       : grid size (gsz x gsz)
    - gstep     : grid step in mas
    - adk_signal: a 2d array (numpy or cupy) containing the kernel-phase vectors
    - cref      : reference contrast (optional, default = 1000)
    - W         : a cube of whitening matrices for the data 
    - deltas    : An array containing the values of field rotation angles
    - thetype   : the cupy dtype for the GPU matrices (cp.float64 or cp.float32 recommended)

    Remarks:
    -------
    In the high-contrast regime, the amplitude of the signature is inversely 
    proportional to the contrast.
    ---------------------------------------------------------------
    """
    import cupy as cp
    #A few assertions:

    if W is not None:
        print("W is not necessary! This function now uses a whitening L matrix kpo.Mp")
        return 1
    if deltas.shape[0] != adk_signalg.shape[0]:
        print("Bad kernel shape")
        return 1
    if deltas is None:
        print("Error: need a rotation angle")
        return 1
    
    nf = deltas.shape[0]

    if cp.get_array_module(self.Mp) is not cp:
        print("This is the GPU version, it wants a cp array!")
        return 1


    #building a grid
    a = cp.vstack(adk_signalg.flatten())
    lwadk_signal = self.Mp.dot(a)

    xs, ys = np.meshgrid(np.arange(gsz), np.arange(gsz))
    ds = np.round(gsz/2)
    cpform = ((ys-ds)*gstep).flatten() + 1j*((-(xs-ds))*gstep).flatten()
    rhos, thetas = np.abs(cpform), np.angle(cpform)*180./np.pi
    thetasdeltas = thetas[:, None] + deltas[None, :]
    dparams = np.array([np.array([0,deltas[i],0]) for i in range(nf)])
    params = np.array([rhos, thetas, cref*np.ones_like(rhos)]).T

    #Creating the larger array of parameters for individual observations
    superparams = params[:, None, :] + dparams[None, :,:]
    #Retrieving the corresponding model signatures
    if verbose:
        print("Getting the model observables")
        sys.stdout.flush()
        itparams = tqdm(superparams)
    else:
        itparams = superparams
    signatures = np.array([self.get_sadk_signature(p, verbose=False) for p in itparams])
    
    signaturesg = cp.asarray(signatures, dtype=thetype)
    
    if verbose:
        print("Projection and reduction")
        sys.stdout.flush()
        pixels = tqdm(range(rhos.shape[0]))
    else:
        pixels = range(rhos.shape[0])
    projectedth = []
    crit = []
    wsigss = []
    for pix in pixels:
        wsigss.append(signaturesg[pix].flatten())
    wsigss = cp.vstack(wsigss)
    pixelthproj = self.Mp.dot(wsigss.T)
    pixnorm = cp.squeeze(cp.linalg.norm(pixelthproj, axis=0))
    crit = cp.squeeze(pixelthproj.T.dot(lwadk_signal)) / pixnorm

    #print("crit",crit.shape)
    #print("gsz", gsz)
    #print("pixelthproj", pixelthproj.shape)
    if not full_output:
        return(crit.reshape(gsz, gsz))
    else :
        return(cp.asnumpy(crit.reshape(gsz,gsz)), cp.asnumpy(pixnorm.reshape(gsz,gsz)),
               rhos.reshape(gsz,gsz), thetas.reshape(gsz,gsz))

xara.KPO.gpu_sadk_binary_match_map = gpu_sadk_binary_match_map

def cpu_sadk_binary_match_map(self, gsz, gstep, adk_signal,W=None,deltas=None,
                                   verbose=False, project=True, cref=1000, full_output=False):
    """ Produces a 2D-map showing where the best binary fit occurs

    Computes the dot product between the kp_signal and a grid (x,y) grid of 
    possible positions for the companion, for a pre-set contrast.

    Parameters:
    ----------
    - gsz       : grid size (gsz x gsz)
    - gstep     : grid step in mas
    - adk_signal: a 2d array containing the kernel-phase vectors
    - cref      : reference contrast (optional, default = 1000)
    - W         : a cube of whitening matrices for the data 
    - deltas    : An array containing the values of field rotation angles

    Remarks:
    -------
    In the high-contrast regime, the amplitude of the signature is inversely 
    proportional to the contrast.
    ---------------------------------------------------------------
    """
    #import cupy as cp
    #A few assertions:
    

    if W is not None:
        print("W is not necessary! This function now uses a whitening L matrix kpo.Mp")
        return 1
#    if deltas.shape[0] != adk_signal.shape[0]:
#        print("Bad kernel shape")
#        return 1
    if deltas is None:
        print("Error: need a rotation angle")
        return 1
    
    nf = deltas.shape[0]
    L = self.Mp
    
    a = np.vstack(adk_signal.flatten())
    if project:
        if len(self.Mp.shape)==3:
            np.array([self.Mp[i].dot(adk_signal[i]) for i in range(self.Mp.shape[0])])
        else:
            lwadk_signal = self.Mp.dot(a)
    else: 
        lwadk_signal = a

    
    #building a grid

    xs, ys = np.meshgrid(np.arange(gsz), np.arange(gsz))
    ds = np.round(gsz/2)
    cpform = ((ys-ds)*gstep).flatten() + 1j*((-(xs-ds))*gstep).flatten()
    rhos, thetas = np.abs(cpform), np.angle(cpform)*180./np.pi
    thetasdeltas = thetas[:, None] + deltas[None, :]
    dparams = np.array([np.array([0,deltas[i],0]) for i in range(nf)])
    params = np.array([rhos, thetas, cref*np.ones_like(rhos)]).T

    #Creating the larger array of parameters for individual observations
    superparams = params[:, None, :] + dparams[None, :,:]
    #Retrieving the corresponding model signatures
    if verbose:
        print("Getting the model observables")
        sys.stdout.flush()
        itparams = tqdm(superparams)
    else:
        itparams = superparams
    signaturesg = np.array([self.get_sadk_signature(p, verbose=False) for p in itparams])
    
    
    if verbose:
        print("Projection and reduction")
        sys.stdout.flush()
        pixels = tqdm(range(rhos.shape[0]))
    else:
        pixels = range(rhos.shape[0])
    projectedth = []
    crit = []
    wsigss = []
    for pix in pixels:
        wsigss.append(signaturesg[pix].flatten())
    wsigss = np.vstack(wsigss)
    pixelthproj = self.Mp.dot(wsigss.T)
    pixnorm = cp.squeeze(np.linalg.norm(pixelthproj, axis=0))
    crit = np.squeeze(pixelthproj.T.dot(lwadk_signal)) / pixnorm

    #print("crit",crit.shape)
    #print("gsz", gsz)
    #print("pixelthproj", pixelthproj.shape)
    #print("lwadk",lwadk_signal.shape)
    if not full_output:
        return(crit.reshape(gsz, gsz))
    else :
        return(crit.reshape(gsz,gsz), pixnorm.reshape(gsz,gsz),
               rhos.reshape(gsz,gsz), thetas.reshape(gsz,gsz))

xara.KPO.cpu_sadk_binary_match_map = cpu_sadk_binary_match_map

def Sglr(y, xhat):
    """
    A simple function to return the GLRt statistic (Ceau et al. 2019)
    
    Parameters:
    ----------
    y         : The signature to test
    xhat      : The Maximum Likelihood Estimate signature
    
    """
    St = 2*np.squeeze(y).T.dot(xhat) - xhat.T.dot(xhat)
    return St
xara.Sglr = Sglr

from lmfit import minimize, Parameters, report_errors, Minimizer

def define_roi(self, verbose=True):
    """
    A method that defines the region of interest probed by the kernel model (kpi)
    Reads values from the kpi object ant writes to:
    
    self.rhomin (the Inner Working Angle)
        defined as 0.5 lambda/D (D = longest baseline)
    self.rhomax (the Outer Working Angle)
        defined as 0.5 lambda/b (D = shortest baseline)
        
    returns 
    resol       : Resolution (number of elements along axis) 
    gstep       : step of a minimal grid for an exploration map.
    
    Remark: resol*2 and gstep/2 are recommended for pretty pictures.
    """
    lengths = np.sqrt(self.kpi.UVC[:,0]**2 + self.kpi.UVC[:,1]**2)
    D = np.max(lengths)
    b = np.min(lengths)
    self.rhomin = 0.5*self.CWAVEL / D * 180/np.pi * 3600 * 1000
    self.rhomax = 0.5*self.CWAVEL / b * 180/np.pi * 3600 * 1000
    gstep = self.rhomin
    resol = np.round((self.rhomax/gstep) *2).astype(np.int16)
    return resol, gstep
xara.KPO.define_roi = define_roi

def global_GLR(self, signal, W=None, dtheta=None, n=10, N=1000, mode="cmap", verbose=True):
    """Computes a complete GLR detection test on provided data
  
    Parameters:
    ----------
    signal     : The non whitened calibrated data (if none, use self.KPDT)
    W          : Deprecated -> use self.Mp instead for post-processing matrix
    dtheta     : The rotation angle for ADK (if none, consider classical data)
    n          : When n realizations of the GLR under H0 are obtained above Sglr(y)
                the algorithm stops (sufficient statistics)
    N          : Total number of realizations to do for a positive detection
    
    Explored space is by default lambda/(2D) to lambda/(2b), the corresponding
    values are stored to kpo as self.rhomin and self.rhomax.
    

    """
    if W is None:
        print("Matrix should be provided in KPO.Mp")
        try:
            W = self.Mp
        except:
            print("You must provide a covariance matrix")
            return 1.
    #Whitening the provided data:
    y = W.dot(signal.flatten())
    #First step is to define explored space.
    resol, gstep = self.define_roi()
    if verbose:
        print("rho min",self.rhomin)
        print("rho max", self.rhomax)
    #Then we can get the test statistic on our data:
    Sglry, phat = self.Sglr_fitandget(kappa=signal,W=W,dtheta=dtheta,mode=mode, showmap=verbose)
    print("The statistcs value Sglrb = %f"%(Sglry))
    print(phat)
    #Now we want to evaluate the Pfa (Pvalue) of this result.
    
    Sglrymc = []
    n_fp = 0
    N_done = 0
    for i in tqdm(np.arange(N)):
        print("dtheta", dtheta)
        S, phat = self.Sglr_fitandget(kappa=None,W=W, dtheta=dtheta, mode=mode, showmap=(N_done<10))
        #print("S shape", S.shape)
        #print(phat)
        Sglrymc.append(S)
        N_done += 1
        print("Sglr mc = %f"%(S))
        if S >= Sglry:
            print("Found a new false positive")
            n_fp += 1
        if n_fp >= n:
            print("Reached %d false positives"%(n))
            break
    Sglrymc = np.array(Sglrymc)
    if n_fp == 0:
        print("No false positive found")
        print("Pfa < %f percent"%(100.* 1./N_done))
    else :
        print("Found %d false positives"%(n_fp))
        print("Pfa = %f"%(100. * float(n_fp)/N_done))
    return Sglry, Sglrymc, phat
    
    
def Sglr_fitandget(self, kappa=None, W=None, dtheta=None, mode="cmap", showmap="False"):
    """
    Method used by global_GLR()
    Computes the GLRb statistic of a whitened signal
    Returns the statistc as well as the fitted parameters in the
    lmfit Parameters format
    
    Parameters:
    ----------
    signal     : The non whitened calibrated data (if none, use self.KPDT)
    W          : The whitening matrix for theoretical data (if none,
                build it from self.kp_cov)
    dtheta     : The rotation angle for ADK (if none, consider classical data)
    
    
    Explored space is by default lambda/(2D) - lambda/(2b)
    """
    if mode == "cmap":
        #We build a colinearity map to get a starting point
        if kappa is None:
            y = np.random.normal(loc=0.0,scale=1.0,size=(self.Mp.shape[0]))
        else :
            y = W.dot(kappa.flatten())
        gstep = self.rhomin
        gsz = np.round(2 * (self.rhomax/gstep)).astype(np.int16)
        if dtheta is not None:
            cmap, norm, rhos, thetas = self.cpu_sadk_binary_match_map(gsz, 
                                                      gstep, np.array([y]), W=None,deltas=dtheta,
                                                      cref=1000, full_output=True, project=False)
        else :
            cmap, norm, rhos, thetas = self.cpu_sadk_binary_match_map(gsz, 
                                                      gstep, np.array([kappa]), W=W,deltas=dtheta,
                                                      cref=1000, full_output=True)
        cmap = cmap * (rhos<=self.rhomax) * (rhos>=self.rhomin)
        loc = np.unravel_index(np.nanargmax(cmap),cmap.shape)

        startparams = np.array([rhos[loc], thetas[loc], 1000 * 1 / (cmap[loc] /
                         norm[loc])])

        
        params = Parameters()
        params.add("rho", value=startparams[0], min=self.rhomin, max=self.rhomax)
        params.add("theta", value=startparams[1])
        params.add("contrast", value=startparams[2])
        if dtheta is not None:
            soluce = minimize(self.get_adk_residual,params,
                            args=(np.array([y]), dtheta),
                            full_output=True)
        else:
            soluce = minimize(self.get_kpd_residual,params,
                            args=(y, W, self.CWAVEL,dtheta),
                            full_output=True)
        #paramshat = np.array([soluce.params["rho"],
        #                     soluce.params["theta"],
        #                     soluce.params["contrast"]])
        paramshat = soluce.params
        
        if showmap:
            hs =  gsz/2*gstep
            plt.figure()
            plt.imshow(cmap, cmap=vdg, extent=[-hs,+hs,-hs,+hs])
            plt.scatter(-params["rho"].value*np.sin(params["theta"].value*np.pi/180),
                       +params["rho"].value*np.cos(params["theta"].value*np.pi/180),
                        marker="x", c="w", s=200)
            plt.scatter(-paramshat["rho"].value*np.sin(paramshat["theta"].value*np.pi/180),
                       +paramshat["rho"].value*np.cos(paramshat["theta"].value*np.pi/180),
                        marker="+", c="r", s=200)
            plt.title("cmap")
            plt.show()
            #plt.figure()
            #plt.imshow(1 * (rhos<=self.rhomax))
            #plt.colorbar()
            #plt.title("rhos")
            #plt.show()
            #print("cmap",cmap.shape)
            #print("norm",norm.shape)
            #print("rhos",rhos.shape)
            #print("thetas",thetas.shape)
            print(startparams)
            print(np.array([paramshat["rho"].value, paramshat["theta"].value, paramshat["contrast"].value]))
            
        if dtheta is not None:
            pphat = np.array([[paramshat["rho"], paramshat["theta"] + atheta, paramshat["contrast"]] for atheta in dtheta])
            xhat = W.dot(get_sadk_signature(self,
                   pphat, verbose=False).flatten())
        else: 
            xhat = W.dot(get_kernel_signature(self, np.array([paramshat["rho"], paramshat["theta"], paramshat["contrast"]]) ,
                                        dtheta, verbose=False))
        Sglry = xara.Sglr(np.array([y]),xhat)
    return Sglry, paramshat


xara.KPO.global_GLR = global_GLR
xara.KPO.Sglr_fitandget = Sglr_fitandget

class kpd_fitter(object):
    
    def __init__(self,kpo,satmask,mytool):
        self.tool = mytool
        self.kpo = kpo
        self.satmask = satmask
        self.sigmares = 0

    def add_obsdata(self, targetdata):
        self.kpd = targetdata["kpd"]
        self.kpo.kpf = self.kpd
        self.Sigmatarg = targetdata["Sigma"]
    def add_calibrators(self, calibrators):
        nb = calibrators["kpd"].shape[0]
        if len(calibrators["kpd"].shape)==1 :
            self.calib = calibrators["kpd"]
            self.Sigmacal = calibrators["Sigma"]
        else :
            self.calib = np.average(calibrators["kpd"],axis=0)
            self.Sigmacal = np.sum(1./nb**2 * calibrators["Sigma"],axis=0)
        self.Sigmatot = self.Sigmatarg + self.Sigmacal
        self.W = sqrtm(np.linalg.pinv(self.Sigmatot))
    
    def xara_model_residual(self, params):
        
        phase = xara.phase_binary(self.kpo.kpi.uv[:,0],self.kpo.kpi.uv[:,1], self.tool.sim.wl, [params["rho"].value, params["theta"].value, params["cont"].value], deg=False)
        kernel = self.kpo.kpi.KPM.dot(phase)
        err = self.W.dot(self.kpd - self.calib - kernel)
        return err


def shifter(im0,vect, buildmask = True, sg_rad=40.0, verbose=False, nbit=10):


    szh = im0.shape[1] # horiz
    szv = im0.shape[0] # vertic

    temp = np.max(im0.shape) # max dimension of image

    for sz in [64, 128, 256, 512, 1024, 2048]:
        if sz >= temp: break

    dz = sz//2.           # image half-size
    if buildmask:
        #print("We have to make a new mask here")
        imcenter = xara.find_psf_center(im0, verbose=verbose)
        sgmask = xara.super_gauss(sz, sz, imcenter[1], imcenter[0], sg_rad)
    else:
        #print("Mask already exists")
        print("ERROR: HERE you should build the relevant mask")
        return
    x,y = np.meshgrid(np.arange(sz)-dz, np.arange(sz)-dz)
    wedge_x, wedge_y = x*np.pi/dz, y*np.pi/dz
    offset = np.zeros((sz, sz), dtype=complex) # to Fourier-center array

    # insert image in zero-padded array (dim. power of two)
    im = np.zeros((sz, sz))
    orih, oriv = (sz-szh)//2, (sz-szv)//2
    im[oriv:oriv+szv,orih:orih+szh] = im0
    
    #print(vect[1],vect[0])

    (x0, y0) = (vect[1], vect[0])
    
    im -= np.median(im)

    dx, dy = x0, y0
    im = np.roll(np.roll(im, -int(dx), axis=1), -int(dy), axis=0)

    #print("recenter: dx=%.2f, dy=%.2f" % (dx, dy))
    dx -= np.int(dx)
    dy -= np.int(dy)

    temp = im * sgmask
    mynorm = temp.sum()

    # array for Fourier-translation
    dummy = shift(dx * wedge_x + dy * wedge_y)
    offset.real, offset.imag = np.cos(dummy), np.sin(dummy)
    dummy = np.abs(shift(ifft(offset * fft(shift(temp)))))

    #dummy = im
    # image masking, and set integral to right value
    dummy *= sgmask

    return (dummy * mynorm / dummy.sum())

def intro_companion(image, params, pscale):
    rho = params[0]
    theta = params[1]
    c = params[2]
    xshift = (- rho * np.sin(np.deg2rad(theta))) / pscale
    yshift = ( rho * np.cos(np.deg2rad(theta)) ) / pscale
    
    compagim = shifter(image, np.array([-yshift, -xshift]))
    return image + compagim / c



#Averaging visibilities
#cubestack = (cviss +np.roll(cviss, -1 ,axis=0) + np.roll(cviss, -2, axis=0)+
#            +np.roll(cviss, -3 ,axis=0) + np.roll(cviss, -4, axis=0)+ np.roll(cviss, -5, axis=0))/6
def cvis_phase_wedge(self, offset, ysz):
    dx, dy = offset[0], offset[1]
    uvc   = self.kpi.UVC * self.M2PIX
    corr = np.exp(i2pi * uvc.dot(np.array([dx, dy])/float(ysz)))
    return corr
xara.KPO.cvis_phase_wedge = cvis_phase_wedge
def uvphase_score(akpo, offset, thecvis, ysz, order=2):
    corr = akpo.cvis_phase_wedge(offset, ysz)
    outcvis = thecvis * corr
    thescore = np.sum(np.angle(outcvis)**order)
    return thescore
def uvphase_residual(params, akpo, thecvis, ysz, order=2):
    offset = np.array([params["x"], params["y"]])
    return uvphase_score(akpo, offset, thecvis, ysz, order=2)
i2pi = 1j*2*np.pi

from lmfit import minimize, Parameters

def optimize_phase(akpo, phases, imsize):
    ysz = imsize
    cviscor = []
    phiproj = []
    for i in tqdm(np.arange(phases.shape[0])):
        params = Parameters()
        params.add("x", value=0, min=-2., max=2.)
        params.add("y", value=0, min=-2., max=2.)
        soluce = minimize(uvphase_residual, params=params,args=(akpo, phases[i], ysz, "order=4"),method="cg")
        offset = np.array([soluce.params["x"].value, soluce.params["y"].value])
        cviscor.append(phases[i] * akpo.cvis_phase_wedge(offset, ysz))
        
    cviscor = np.array(cviscor)
    return cviscor
