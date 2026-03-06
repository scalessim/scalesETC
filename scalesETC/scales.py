#from scalesETC.focal_plane import *
#from scalessim.base import *
#from scalessim.io import *
#from scalessim.detectors import *

from scalesETC.io import *
from scalesETC.focal_plane import *
from scalesETC.psfs import *
from scalesETC.targs import *

from ipywidgets import IntProgress, Label
from IPython.display import display
from scipy.sparse import *
from astropy import units as u
import numpy as np


class SCALES:
    def __init__(self,scalesmode,guidestar,verbose=False,fullfr=False,ccurves=False):
        self.res = scalesmode.value.split(' ')[0]
        self.mode = scalesmode.value.split(' ')[1].split(':')[0]
        if self.res=='Med-Res': self.med=True
        else: self.med=False

        rkey=scalesmode.value.split(' ')[0]+'_'+scalesmode.value.split(' ')[1][:-1]
        pmat_file = 'data/projmats/'+rkey+'.npz'
        self.pmat = load_npz(pmat_file)
        self.plams = np.loadtxt('data/projmats/'+rkey+'_lams.txt')*u.micron

        self.wav_min = float(scalesmode.value.split(':')[1].split('-')[0])
        self.wav_max = float(scalesmode.value.split(':')[1].split('-')[1])


        rmat_file = 'data/rectmats/'+rkey+'_ones.npz'
        c2rmat_file = 'data/rectmats/'+rkey+'_c2.npz'
        rlams_file = 'data/rectmats/'+rkey+'_lams.txt'

        self.rmat = load_npz(rmat_file)
        if os.path.isfile(c2rmat_file)==False:
            self.c2rmat = None
        else:
            self.c2rmat = load_npz(c2rmat_file)
        self.rlams = np.loadtxt(rlams_file)*u.micron

        self.filt = Filter(lmin=self.wav_min,lmax=self.wav_max,fkw='asahi')

        self.PSF = PSFs(self,guidestar,ccurves=ccurves)



    def image_and_cube(self,targ=None,cube=None,
             inst_emissivities = [0.4],inst_temps = [277*u.K],
             dit=1,nexps=1,vortex=False,extraction='optimal',
             vapor = 1.0, airmass = 1.0, change_psfs=False,
             shot_off=False,verbose=False,
             skytrans_off=False,bkg_off=False,bkgsub=True):
 
        self.skybg = SkyBG(vapor,airmass)
        self.skytrans = SkyTrans(vapor,airmass)
        if skytrans_off==True: self.skytrans.y = np.ones(self.skytrans.y.shape)*u.dimensionless_unscaled
        self.atmodisp = AtmoDispersion(90,20,600)

        self.inst = InstTransEm(inst_emissivities, inst_temps)
        self.qe = QE()


        self.args = {
                    'SkyBG':self.skybg,
                    'SkyTrans':self.skytrans,
                    'InstTransEm':self.inst,
                    'Filter':self.filt,
                    'QE':self.qe,
                    'ProjMat':self.pmat,
                    'ProjLams':self.plams,
                    'RectMat':self.rmat,
                    'C2RectMat':self.c2rmat,
                    'RectLams':self.rlams,
                    'ResMode':self.res,
                    'PSF':self.PSF
                    }
            
        self.fp = FocalPlane(self.args)

        if targ!=None:
            res = self.fp.get_fp(dit*u.s,nexps,self.pmat,self.rmat,Target=targ,
                                 bg_off = bkg_off,extraction=extraction,shot_off=shot_off,
                                 verbose=verbose,medium=self.med,vortex=vortex)
            if vortex == False:
                img_seq,IFScube_seq = res
            else: 
                img_seq,img_seq_nc,IFScube_seq,IFScube_seq_nc = res


        elif cube!=None:        
            img_seq,IFScube_seq = self.fp.get_fp(dit*u.s,nexps,self.pmat,self.rmat,cube=cube,
                                                 bg_off = bkg_off,extraction=extraction,shot_off=shot_off,
                                                 verbose=verbose,medium=self.med,vortex=vortex)

        

        targ_bg = Target(np.linspace(1.8,5.4,1000),np.zeros(1000))
        img_seq_bg,IFScube_seq_bg = self.fp.get_fp(dit*u.s,nexps,self.pmat,self.rmat,Target=targ_bg,
                                 bg_off = bkg_off,extraction=extraction,shot_off=shot_off,
                                 verbose=verbose,medium=self.med,vortex=False)

        if bkgsub==True:
            if vortex==False:
                return img_seq-img_seq_bg, IFScube_seq-IFScube_seq_bg, self.rlams
            else:
                return img_seq-img_seq_bg, IFScube_seq-IFScube_seq_bg, img_seq_nc-img_seq_bg, IFScube_seq_nc-IFScube_seq_bg, self.rlams

        if bkgsub==False:
            if vortex==False:
                return img_seq, img_seq_bg, IFScube_seq, IFScube_seq_bg, self.rlams
            else:
                return img_seq, img_seq_bg, IFScube_seq, IFScube_seq_bg, img_seq_nc, img_seq_bg, IFScube_seq_nc, IFScube_seq_bg, self.rlams



 
    def point_snr_cube(self,targ=None,cube=None,
             inst_emissivities = [0.4],inst_temps = [277*u.K],
             dit=1,nexps=1,extraction='optimal',
             vapor = 1.0, airmass = 1.0, change_psfs=False,
             verbose=False,skytrans_off=False,bkg_off=False):
 
        self.skybg = SkyBG(vapor,airmass)
        self.skytrans = SkyTrans(vapor,airmass)
        if skytrans_off==True: self.skytrans.y = np.ones(self.skytrans.y.shape)*u.dimensionless_unscaled
        self.atmodisp = AtmoDispersion(90,20,600)

        self.inst = InstTransEm(inst_emissivities, inst_temps)
        self.qe = QE()


        self.args = {
                    'SkyBG':self.skybg,
                    'SkyTrans':self.skytrans,
                    'InstTransEm':self.inst,
                    'Filter':self.filt,
                    'QE':self.qe,
                    'ProjMat':self.pmat,
                    'ProjLams':self.plams,
                    'RectMat':self.rmat,
                    'C2RectMat':self.c2rmat,
                    'RectLams':self.rlams,
                    'ResMode':self.res,
                    'PSF':self.PSF
                    }
            
        self.fp = FocalPlane(self.args)

        if targ!=None:
            res = self.fp.get_fp(dit*u.s,nexps,self.pmat,self.rmat,Target=targ,
                                 extraction=extraction,shot_off=True,
                                 verbose=verbose,medium=self.med)
            
            img_seq,IFScube_seq = res


        elif cube!=None:        
            img_seq,IFScube_seq = self.fp.get_fp(dit*u.s,nexps,self.pmat,self.rmat,cube=cube,
                                                 extraction=extraction,shot_off=True,
                                                 verbose=verbose,medium=self.med)

        

        targ_bg = Target(np.linspace(1.8,5.4,1000),np.zeros(1000))
        img_seq_bg,IFScube_seq_bg = self.fp.get_fp(dit*u.s,nexps,self.pmat,self.rmat,Target=targ_bg,
                                 extraction=extraction,shot_off=True,
                                 verbose=verbose,medium=self.med,vortex=False)

        bkgsub_cube = IFScube_seq-IFScube_seq_bg
        SNRcube_seq = []
        SNRlist_seq = []
        for nn in range(nexps):
            SNRcube = calc_SNR_cube(bkgsub_cube[nn],IFScube_seq_bg[nn])
            SNRcube_seq.append(SNRcube)
            if self.med==True:
                SNRlist = calc_SNR_lam_ap_med(bkgsub_cube[nn],IFScube_seq_bg[nn],self.rlams)
            else:
                SNRlist = calc_SNR_lam_ap(bkgsub_cube[nn],IFScube_seq_bg[nn],self.rlams)
            SNRlist_seq.append(SNRlist_seq)
        return SNRcube, SNRlist, self.rlams
            