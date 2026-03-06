from scipy.ndimage import shift
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from ipywidgets import IntProgress,Label
from IPython.display import display
from copy import deepcopy
from scipy.sparse.linalg import lsmr


class FocalPlane:
    def __init__(self, args, gain=11.):

        self.gain = gain #e/DN
        self.SkyBG = args['SkyBG']
        self.SkyTrans = args['SkyTrans']
        self.Inst = args['InstTransEm']
        self.QE = args['QE']
        self.Filter = args['Filter']
        self.pmat = args['ProjMat']
        self.lam = args['ProjLams'] 
        self.rmat = args['RectMat']
        self.c2rmat = args['C2RectMat']
        self.rlam = args['RectLams']
        self.res = args['ResMode']
        self.PSF = args['PSF']

        self.area = 76. * u.m**2 ###collecting area of telescope

        if self.res == 'Med-Res':
            self.num_spaxel = 18
            self.fov = 0.34*0.36*(u.arcsec)**2
        else:
            self.num_spaxel = 108
            self.fov = 2.16*2.16*(u.arcsec)**2
        self.dlam = self.lam[1]-self.lam[0] ###dlam of oversampled wavelength 

    
    def get_fp(self, dit, nexps, pmat, rmat, 
               Target=None, cube=None, shot_off=False,
               bg_off=False, verbose=False,return_phots=False,
               medium=False, vortex=False,extraction='optimal'):
        

        skybg = self.SkyBG.resample(self.lam) * self.fov / self.num_spaxel**2        
        instbg = self.Inst.get_em(self.lam) * self.fov / self.num_spaxel**2

        qe = self.QE.get_qe(self.lam)

        self.bg = skybg + instbg

        filtertrans = self.Filter.interp(self.lam)
        skytrans = self.SkyTrans.resample(self.lam)
        teltrans,insttrans = self.Inst.get_trans(self.lam)
        self.trans = teltrans*insttrans*filtertrans*skytrans


        bg_spec_in_phot = dit*(teltrans*insttrans*filtertrans*skybg + insttrans*filtertrans*instbg) * self.dlam * self.area.to(u.cm**2)
        bg_spec_in_dn = bg_spec_in_phot*qe / self.gain / u.electron

        IFScube_seq = np.zeros((nexps,len(self.rlam), self.num_spaxel, self.num_spaxel))
        if vortex==True:
            IFScube_seq_nc = np.zeros((nexps,len(self.rlam), self.num_spaxel, self.num_spaxel))
            img_seq_nc = np.zeros([nexps,2048,2048])
        if medium==True:
            IFScube_seq = np.zeros((nexps,len(self.rlam), self.num_spaxel, self.num_spaxel-1))
        img_seq = np.zeros([nexps,2048,2048])

        for nexp in range(nexps):
            img = np.ones((len(self.lam), self.num_spaxel, self.num_spaxel))
            if medium==True:
                img = np.ones((len(self.lam), self.num_spaxel, self.num_spaxel-1))
            if vortex==True:
                psf_nc, psf = self.PSF.PSF_sequence(nframes=1, vortex=vortex, med=medium)
                img_nc = np.ones((len(self.lam), self.num_spaxel, self.num_spaxel))

            else:
                psf = self.PSF.PSF_sequence(nframes=1, vortex=vortex, med=medium)
            
            if not bg_off:
                #print('adding bkg')
                if return_phots == True:
                    img = img * bg_spec_in_phot[:,None,None].value
                    if vortex == True:
                        img_nc = img_nc * bg_spec_in_phot[:,None,None].value
                else:
                    img = img * bg_spec_in_dn[:, None, None].si.value
                    if vortex == True:
                        img_nc = img_nc * bg_spec_in_dn[:, None, None].si.value
            else:
                #print('no bkg added')
                img[:] = 0.0

            if Target:
                source = Target.resample(self.lam)
                h = 6.621e-27*u.cm*u.cm*u.g/u.s
                c = 2.9979e10*u.cm/u.s
                lamscm = self.lam.to(u.cm)
                source2 = source.to(u.cm*u.cm*u.g/u.s/u.s/u.cm/u.cm/u.micron/u.s) * lamscm / h / c * u.ph
                source_spec_in_phot = dit*self.trans*source2 * self.dlam * self.area.to(u.cm**2)
                source_spec_in_dn = source_spec_in_phot*qe / self.gain / u.electron
       
                if return_phots == True:
                    img += psf[0] * source_spec_in_phot[:, None, None].value
                    if vortex == True:
                        img_nc += psf_nc[0] * source_spec_in_phot[:, None, None].value
                else:
                    img += psf[0] * source_spec_in_dn[:, None, None].si.value
                    if vortex ==True:
                        img_nc += psf_nc[0] * source_spec_in_dn[:, None, None].si.value

                    
            if cube is not None:
                h = 6.621e-27*u.cm*u.cm*u.g/u.s
                c = 2.9979e10*u.cm/u.s
                lamscm = self.lam.to(u.cm)
                cube2 = []
                for x in range(len(cube)):
                    tmp = cube[x].to(u.cm*u.cm*u.g/u.s/u.s/u.cm/u.cm/u.micron/u.s) * lamscm[x] / h / c * u.ph
                    cube2.append(tmp)
                cube2 = np.array(cube2)
                mult_phot = dit*self.trans * self.dlam * self.area.to(u.cm**2)
                mult_dn = mult_phot * qe / self.gain / u.electron
                if return_phots == True:
                    img += (cube2 * mult_phot[:, None, None]).value
                else:
                    img += (cube2 * mult_dn[:, None, None]).value
                    #print('scaling cube')
                img = self.PSF.convolve(psf,img)

            #if adiscene is not None:
                
            
            img_flat = np.matrix(img.reshape(np.prod(img.shape),1))
            if vortex == True:
                img_nc_flat = np.matrix(img_nc.reshape(np.prod(img_nc.shape),1))
                image_nc = np.array(pmat*img_nc_flat).reshape([2048,2048])
            
            image = np.array(pmat*img_flat).reshape([2048,2048])

            if shot_off==False: 
                image = np.random.poisson(image)
                if vortex == True:
                    image_nc = np.random.poisson(image_nc)
                        
            img_seq[nexp] = image
            if vortex == True:
                img_seq_nc[nexp] = image_nc

            if extraction=='optimal':
                detim_flat = np.matrix(image.reshape(np.prod(image.shape),1))
                if medium==False:
                    IFScube = np.array(rmat*detim_flat).reshape([54,108,108])
                else: 
                    IFScube = np.array(rmat*detim_flat).reshape([1900,18,17])
                if vortex==True:
                    detim_nc_flat = np.matrix(image_nc.reshape(np.prod(image.shape),1))
                    if medium==False:
                        IFScube_nc = np.array(rmat*detim_nc_flat).reshape([54,108,108])
                    else: 
                        IFScube_nc = np.array(rmat*detim_nc_flat).reshape([1900,18,17])
            if extraction=='chi2':
                detim_flat = np.matrix(image.reshape(np.prod(image.shape),1))
                startcube = np.array(rmat*detim_flat).flatten()
                d_prime = image.flatten()
                R_prime = self.c2rmat
                res_sp = lsmr(R_prime, d_prime,x0=startcube)
                resids = np.array(R_prime*res_sp[0]).reshape([2048,2048])-image
                plt.imshow(resids)
                plt.ylim(1000,1100)
                plt.xlim(1000,1100)
                plt.colorbar()
                plt.show()

                if medium==False:
                    IFScube = res_sp[0].reshape([54,108,108])
                else:
                    IFScube = res_sp[0].reshape([1900,18,17])
            
            IFScube_seq[nexp] = IFScube
            if vortex == True:
                IFScube_seq_nc[nexp] = IFScube_nc
        if vortex == True:
            return img_seq, img_seq_nc, IFScube_seq, IFScube_seq_nc
        else:
            return img_seq, IFScube_seq



