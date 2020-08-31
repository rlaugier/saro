gpu = False
import numpy as np
if not gpu:
    import numpy as cp
else:
    import cupy as cp
import xara

from .kernel_cm import br, bo, vdg, bbr

import xaosim
from tqdm import  tqdm
import matplotlib.pyplot as plt

import scipy.sparse as sparse
from scipy.linalg import sqrtm

from xara import fft, ifft, shift

from . import detection_maps as dmap


from lmfit import minimize, Parameters, report_errors, Minimizer



from scipy.sparse import diags




import matplotlib.cm as cm


"""
The functions uses the wavelength provided by KPO.CWAVEL
"""


def super_gauss(xs, ys, x0, y0, w,s=True, o=4):
    ''' Returns an 2D super-Gaussian function
    ------------------------------------------
    Parameters:
    - (xs, ys) : array size
    - (x0, y0) : center of the Super-Gaussian
    - w        : width of the Super-Gaussian 
    ------------------------------------------ '''

    x = np.outer(np.arange(xs), np.ones(ys))-x0
    y = np.outer(np.ones(xs), np.arange(ys))-y0
    if s:
        dist = np.max(np.array([np.abs(x), np.abs(y)]), axis=0)
    else:
        dist = np.sqrt(x**2+y**2)
        
    if o is None:
        gg = 1*(dist<w)
    else:
        gg = np.exp(-(dist/w)**o)
    return gg

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


class KPO(xara.KPO):
    """
    Same good old xara.KPO class with a few added goodies:
    
    -define_roi(): Computes the region of interest that the model can reach
    -create_cov_matrix(): Creates an estimation of the cov matrix (fixed)
    -build_L_matrix(): Builds a L matrix for ADK
    -plot_pupil_and_uv(): Improved version
    -global_GLR(): Conducts an iterative GLR test on the data
    -gpu_sadk_binary_match_map(): Colinearity maps (ADK compatible)
    -cpu_sadk_binary_match_map(): Colinearity maps (ADK compatible) GPU accelerated
    -get_cvis(): Directly gets binary cvis
    -get_sadk_signature(): Directly gets binary kernel-phase signaure in batch
    -get_kernel_signature(): Directly gets binary kernel-phase signature
    """
    

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

    
    def get_sadk_signature(self, params, verbose=False):
        """Returns a series of kernel signatures 
        The result is given (end - start)
        Parameters:
        -----------
        - params    : An array of parameters of parameters object
                     (stacked along dimension 0)
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




    
    def build_L_matrix(self, nf, crop=True, giveUf=True):
        Uf = sparse.vstack([sparse.identity(self.kpi.nbkp) for i in range(nf)])
        L = (sparse.identity(Uf.shape[0]) - 1/nf * Uf.dot(Uf.T))
        if crop:
            L = L[:-self.kpi.nbkp,:]
        if giveUf:
            return L, Uf
        else :
            return L


    def get_adk_residual(self, params, y, dtheta):
        pparams = np.array([np.array([params["rho"],params["theta"],params["contrast"]]) + np.array([0, theta, 0]) for theta in dtheta])
        kappa = self.get_sadk_signature(pparams, verbose=False)
        residual = self.Mp.dot(kappa.flatten()) - y
        return residual
    def get_kpd_residual(self, params, y, W, wl, dtheta):
        kappa = self.get_sadk_signature(np.array([params["rho"],params["theta"],params["contrast"]]),
                                   verbose=False)
        residual = W.dot(kappa) - y
        return residual




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
        - adk_signalg: a 2d array (cupy) containing the kernel-phase vectors
        - cref      : reference contrast (optional, default = 1000)
        - W         : deprecated (cube of whitening matrices) (provide through kpo.Mp (cupy array))
        - deltas    : An array containing the values of field rotation angles
        - thetype   : the cupy dtype for the GPU matrices (cp.float64 or cp.float32 recommended)
        - full_output:False: returns only an array containing the matched filter:
                      True: returns:
                              -Array containing the matched filter
                              -Array of the model norm
                              -Array of the separation (mas)
                              -Array of the position angle (deg)

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
        - full_output:False: returns only an array containing the matched filter:
                      True: returns:
                              -Array containing the matched filter
                              -Array of the model norm
                              -Array of the separation (mas)
                              -Array of the position angle (deg)

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
        if verbose:
            print("D = ", D)
            print("b = ", b)
        self.rhomin = 0.5*self.CWAVEL / D * 180/np.pi * 3600 * 1000
        self.rhomax = 0.5*self.CWAVEL / b * 180/np.pi * 3600 * 1000
        gstep = self.rhomin
        resol = np.round((self.rhomax/gstep) *2).astype(np.int16)
        return resol, gstep

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
                plt.imshow(cmap, cmap=bo, extent=[-hs,+hs,-hs,+hs])
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
                xhat = W.dot(self.get_sadk_signature(pphat, verbose=False).flatten())
            else: 
                xhat = W.dot(get_kernel_signature(self, np.array([paramshat["rho"], paramshat["theta"], paramshat["contrast"]]) ,
                                            dtheta, verbose=False))
            Sglry = Sglr(np.array([y]),xhat)
        return Sglry, paramshat
    def create_cov_matrix(self, var_img, ref_img=None, kernel=True,
                      verbose=False, option="ABS", m2pix=None):
        ''' -------------------------------------------------------------------
        generate the covariance matrix for the UV phase.

        For independant noise in the image plane, with a high SNR, the relationship between image and phase can be linearized.

        Parameters:
        ----------
        - var_img: a 2D image with variance per pixel
        - ref_img: optional, a 2D image constituting the nominal reference
                    for the complex visibility
        - kernel: weather to compute the kernel phase covariance based on
                    the phase covariance
        - m2pix:   Necessary m2pix parameter to compute the F matrix id necessary

        Option:
        ------
        This is an ongoing investigation: should the computation involve the
        model redundancy or the real part of the Fourier transform? In the 
        latter scenario, should the computation include cross-terms between
        imaginary and real parts to be more exact?
        
        - "RED":  uses the model redundancy vector
        - "REAL": uses the real part of the FT
        - "ABS" : computation based on the a visibility modulus

        Note: Covariance matrix can also be computed via MC simulations, if
        you are unhappy with the current one. See "append_cov_matrix()"
        ------------------------------------------------------------------- '''

        ISZ = var_img.shape[0]
        try:
            test = self.FF # check to avoid recomputing auxilliary arrays!

        except:
            if m2pix is not None:
                self.FF = xara.core.compute_DFTM1(self.kpi.UVC, m2pix, ISZ)
            else:
                print("Fourier matrix and/or m2pix are not available.")
                print("Please compute Fourier matrix.")
                return
            
        #The best is to create real and imaginary parts independantly
        if ref_img is not None:
            if verbose:
                print("Using a separate reference for amplitude")
            ft = self.FF.dot(ref_img.flat)
        else :
            ft = self.FF.dot(var_img.flat)
        
 
        cov_img = diags(var_img.flatten())
            
        if option == "RED":
            if verbose:
                print("Do not use that: normalization is WIP")
                B = np.diag(1.0/self.kpi.RED).dot(np.angle(self.FF))
            
            
        if option == "REAL":
            if verbose:
                print("Covariance Matrix computed using the real part of FT!")
            refj = np.real(ft)
        
        if option == "ABS":
            if verbose:
                print("Covariance Matrix computed using the abs of the FT!")
            refj = np.abs(ft)
        
        
        B = self.FF.imag / refj[:, None]
        self.phi_cov = B.dot(cov_img.dot(B.T))
        if kernel:
            if verbose:
                print("Computing the covariance of kernelized observables")
            self.kappa_cov = self.kpi.KPM.dot(self.phi_cov).dot(self.kpi.KPM.T)
    
    
    def cvis_phase_wedge(self, offset, ysz):
        dx, dy = offset[0], offset[1]
        uvc   = self.kpi.UVC * self.M2PIX
        corr = np.exp(i2pi * uvc.dot(np.array([dx, dy])/float(ysz)))
        return corr


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

def intro_companion(image, params, pscale, sg_rad=40):
    rho = params[0]
    theta = params[1]
    c = params[2]
    xshift = (- rho * np.sin(np.deg2rad(theta))) / pscale
    yshift = ( rho * np.cos(np.deg2rad(theta)) ) / pscale
    
    compagim = shifter(image, np.array([-yshift, -xshift]), sg_rad=sg_rad)
    return image + compagim / c



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


def plot_pupil_and_uv(self, xymax=None, figsize=(8,4), plot_redun = False,
                          cmap=cm.gray, ssize=7.5, lw=0, alpha=1.0, marker='o',
                          usesize=False, showminmax=False, dpi=200):
        '''Nice plot of the pupil sampling and matching uv plane.

        --------------------------------------------------------------------
        Options:
        ----------

        - xymax: radius of pupil plot in meters           (default=None)
        - figsize: matplotlib figure size                 (default=(12,6))
        - plot_redun: bool add the redundancy information (default=False)
        - cmap: matplotlib colormap                       (default:cm.gray)
        - ssize: symbol size                              (default=12)
        - lw:  line width for symbol outline              (default=0)
        - alpha: gamma (transparency)                     (default=1)
        - maker: matplotlib marker for sub-aperture       (default='o')
        - usesize: size of markers indicate transmission  (default=None)
        - showminmax: write the min and max RED values    (default=False)
        - -------------------------------------------------------------------
        '''

        

        f0 = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()
        ax0 = plt.subplot(121)

        s1, s2 = ssize**2, (ssize/2)**2
        if usesize:
            ax0.scatter(self.VAC[:,0], self.VAC[:,1], s=s1*self.VAC[:,2], c=np.zeros_like(self.VAC[:,2]),
                    cmap=cm.gray, alpha=alpha, marker=marker, lw=lw,
                    vmin=0.0, vmax=1.0)
        else:
            ax0.scatter(self.VAC[:,0], self.VAC[:,1], s=s1, c=self.VAC[:,2],
                    cmap=cmap, alpha=alpha, marker=marker, lw=lw,
                    vmin=0.0, vmax=1.0)
        if xymax is None:
            xymax = np.nanmax(np.abs(self.VAC)) *1.1
        ax0.axis([-xymax, xymax, -xymax, xymax], aspect='equal')
        ax0.set_xlabel("Aperture x-coordinate (meters)")
        ax0.set_ylabel("Aperture y-coordinate (meters)")

        ax1 = plt.subplot(122)
        ax1.scatter(-self.UVC[:,0], -self.UVC[:,1], s=s2, c=self.RED,
                    cmap=cmap, alpha=alpha, marker=marker, lw=lw)
        ax1.scatter(self.UVC[:,0], self.UVC[:,1], s=s2, c=self.RED,
                    cmap=cmap, alpha=alpha, marker=marker, lw=lw)
        ax1.axis([-2*xymax, 2*xymax, -2*xymax, 2*xymax], aspect='equal')
        ax1.set_xlabel("Fourier u-coordinate (meters)")
        ax1.set_ylabel("Fourier v-coordinate (meters)")
        if showminmax:
            stringmin = r"$R_{min}$ = %.2f"%(np.min(self.RED))
            stringmax = r"$R_{max}$ = %.0f"%(np.max(self.RED))
            plt.text(np.min(self.UVC[:,0]), np.min(self.UVC[:,1]), stringmin)
            plt.text(np.min(self.UVC[:,0]), 0.9*np.max(self.UVC[:,1]), stringmax)
        

        # complete previous plot with redundancy of the baseline
        # -------------------------------------------------------
        dy = 0.1*abs(self.uv[0,1]-self.uv[1,1]) # to offset text in the plot.
        if plot_redun:
            for i in range(self.nbuv):
                ax1.text(self.uv[i,0]+dy, self.uv[i,1]+dy, 
                         int(self.RED[i]), ha='center')        
            ax1.axis('equal')
        plt.draw()
        f0.set_tight_layout(True)
        return f0

    # =========================================================================
    # =========================================================================
    
xara.KPI.plot_pupil_and_uv = plot_pupil_and_uv
