from astropy import units as u
import numpy as np
import astropy.io.fits as pyfits
from .io import *
import astropy.constants as const
import astropy.units as u

def phoenix_star(T_s = 3800,logg = 4.5,zz = 0.0,rstar = 1.0,dstar = 20,
                 Hmag = None, Lmag=None, Kmag=None, Mmag=None,
                 phoenixdir='data/PHOENIX_HiRes/'):
    """
    inputs:
        T_s - effectve temperature in K
        logg - log surface gravity in cgs
        zz - metallicity
        rstar - stellar radius in R_sun
        dstar - distance to the system in pc

    returns:
        wav - list of wavelengths in microns
        fluxs - specific intensity in erg/s/cm^2/micron
    """
    logg = '%.2f' % logg
    if T_s < 10000: T_s = '0'+str(int(T_s))

    #specstar = pyfits.getdata('PHOENIX_HiRes/lte'+str(T_s)+'-'+str(logg)+'-'+str(zz)+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits') #flux units are erg/s/cm^2/cm
    specstar = pyfits.getdata(phoenixdir+'lte'+str(T_s)+'-'+str(logg)+'-'+str(zz)+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits') #flux units are erg/s/cm^2/cm
    wav = pyfits.getdata(phoenixdir+'/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits') #units are angstrom
    wav = wav / 1.0e4 ##now units are um
    rstar = rstar*u.R_sun.to(u.cm) 
    dstar = dstar*u.pc.to(u.cm)
    I_lam = specstar * (rstar / dstar)**2 * 1.0e-4 ##to convert from /cm to /um
    
    Lflux = I_lam[np.where(np.abs(wav-3.8)==np.min(np.abs(wav-3.8)))]
    Kflux = I_lam[np.where(np.abs(wav-2.19)==np.min(np.abs(wav-2.19)))]
    Mflux = I_lam[np.where(np.abs(wav-5.0)==np.min(np.abs(wav-5.0)))]
    Hflux = I_lam[np.where(np.abs(wav-1.6)==np.min(np.abs(wav-1.6)))]

    Lmag_t = -2.5*np.log10(Lflux/(Jy_to_flam(288.0,3.8)))
    Mmag_t = -2.5*np.log10(Mflux/(Jy_to_flam(158.0,5.0)))
    Kmag_t = -2.5*np.log10(Kflux/(Jy_to_flam(653.0,2.19)))
    Hmag_t = -2.5*np.log10(Hflux/(Jy_to_flam(1040.0,1.6)))

    """
    print(Lflux)
    print(Jy_to_flam(288,3.8))
    print(Lflux/Jy_to_flam(288,3.8))
    print(Lmag_t)
    print(Lmag)
    print(10**((-Lmag-Lmag_t)/2.5))
    print(Lflux*(10**((-Lmag-Lmag_t)/2.5)))
    """


    if Lmag!=None:
        targ = Target(wav,I_lam*(10**(-(Lmag-Lmag_t)/2.5)))

        Ltest = targ.y.value[np.where(np.abs(targ.x.value-3.8)==np.min(np.abs(targ.x.value-3.8)))]
        print(Ltest)

        
    elif Mmag!=None:
        targ = Target(wav,I_lam*(10**(-(Mmag-Mmag_t)/2.5)))
    
    elif Kmag!=None:
        targ = Target(wav,I_lam*(10**(-(Kmag-Kmag_t)/2.5)))

    elif Hmag!=None:
        targ = Target(wav,I_lam*(10**(-(Hmag-Hmag_t)/2.5)))
    
    else: targ = Target(wav,I_lam)
        
    Ltest = targ.y.value[np.where(np.abs(targ.x.value-3.8)==np.min(np.abs(targ.x.value-3.8)))]
    print(Ltest)
    return targ

def sonora_planet(T_p=300,sg=100,rp=1.0,d=10.0,
                  Hmag=None, Kmag=None, Lmag=None, Mmag=None,
                  sonoradir = 'data/sonora_2018/'):
    rjup_cm = 6.9911e9
    rplan_cm = rp*rjup_cm
    pc_cm = 3.086e18
    a = np.loadtxt(sonoradir+'spectra/sp_t'+str(int(T_p))+'g'+str(sg)+'nc_m0.0.gz',skiprows=2) 
    #microns, fnu in cgs through surface (erg / s /cm^2 / Hz), lam in microns
    a = a[::-1]
    a_lo = a[np.where(a[:,0] > 1.0)]
    a_sub = a_lo[np.where(a_lo[:,0] < 5.5)]
    lplan_um = a_sub[:,0]
    
    a_dist = a_sub.copy()
    a_dist[:,1] = (rplan_cm / (d*pc_cm))**2 * a_dist[:,1] * np.pi
    
    ###convert fnu to flambda
    C = 2.998e10 ###cgs speed of light
    flam_tmp = a_dist[:,1] * C / (a_dist[:,0]*1.0e-4)**2 ###now in erg / s / cm^2 / cm
    I_lam = flam_tmp * 1.0e-4 ###now in erg / s / cm^2 / um
    wav = a_dist[:,0]

    Lflux = I_lam[np.where(np.abs(wav-3.8)==np.min(np.abs(wav-3.8)))]
    Kflux = I_lam[np.where(np.abs(wav-2.19)==np.min(np.abs(wav-2.19)))]
    Mflux = I_lam[np.where(np.abs(wav-5.0)==np.min(np.abs(wav-5.0)))]
    Hflux = I_lam[np.where(np.abs(wav-1.6)==np.min(np.abs(wav-1.6)))]

    Lmag_t = -2.5*np.log10(Lflux/(Jy_to_flam(288.0,3.8)))
    Mmag_t = -2.5*np.log10(Mflux/(Jy_to_flam(158.0,5.0)))
    Kmag_t = -2.5*np.log10(Kflux/(Jy_to_flam(653.0,2.19)))
    Hmag_t = -2.5*np.log10(Hflux/(Jy_to_flam(1040.0,1.6)))

    if Lmag!=None:
        targ = Target(wav,I_lam*(10**(-(Lmag-Lmag_t)/2.5)))

    elif Mmag!=None:
        targ = Target(wav,I_lam*(10**(-(Mmag-Mmag_t)/2.5)))

    elif Kmag!=None:
        targ = Target(wav,I_lam*(10**(-(Kmag-Kmag_t)/2.5)))

    elif Hmag!=None:
        targ = Target(wav,I_lam*(10**(-(Hmag-Hmag_t)/2.5)))
    else: targ = Target(wav,I_lam)
    
    return targ


def planet_and_bkg(T_p=300,sg=100,rp=1.0,d=10.0,Lmag=None,Mmag=None,Kmag=None):
    target = sonora_planet(T_p=T_p,sg=sg,rp=rp,d=d,Kmag=Kmag,Lmag=Lmag,Mmag=Mmag)        
    targ_bg = Target(wav,np.zeros(wav.shape))
    return targ, targ_bg


def star_and_bkg(T_s=3800,logg=4.5,zz=0.0,rstar=1.0,dstar=20,
                 Kmag=None,Lmag=None,Mmag=None):
    target = phoenix_star(T_s=T_s,logg=logg,zz=zz,rstar=rstar,dstar=dstar,Kmag=Kmag,Lmag=Lmag,Mmag=Mmag)
    targ_bg = Target(wav,np.zeros(wav.shape))
    return targ, targ_bg


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    angle1 = np.deg2rad(angle)
    qx = ox + np.cos(angle1) * (px - ox) - np.sin(angle1) * (py - oy)
    qy = oy + np.sin(angle1) * (px - ox) + np.cos(angle1) * (py - oy)
    return qy, qx


def ADI_scene_targs_lowres(psf_seq,star,lams,
                            planet=None,
                            PAlist=np.linspace(-45,45,90),p_sep=350.0, p_PA=45.0,
                            psf_seq_c=None):

    star_new = star.resample(lams)

    if planet!=None:
        planet_new = planet.resample(lams)
        seps=np.array([p_sep/20.0])
        position_angles=np.deg2rad([p_PA])
        posns = np.array([54+seps*np.cos(-position_angles), 54+seps*np.sin(-position_angles)]).T
        coords = rotate((54,54),posns[0],PAlist)

    scene = np.zeros([len(PAlist),len(star_new), 108, 108])
    scene_conv = np.zeros(scene.shape)

    if psf_seq_c==False:
        for i in range(len(scene)):
            if planet!= None: scene[i,:,int(coords[1][i]),int(coords[0][i])] = planet_new
            scene[i,:,54,54] = star_new
            for j in range(len(scene[i])):
                FTPSF = np.fft.fft2(np.fft.fftshift(psfs[i,j]))
                FTSCENE = np.fft.fft2(np.fft.fftshift(scene[i,j]))
                FTCONV = FTPSF*FTSCENE
                convim = np.fft.ifftshift(np.fft.ifft2(FTCONV))
                scene_conv[i][j] = np.real(convim)
        scene_conv = np.array(scene_conv)*u.erg/u.cm/u.cm/u.s/u.um
        return scene, scene_conv
    else:
        scene_conv_s = np.zeros(scene.shape)
        for i in range(len(scene)):
            sc_star = scene[i].copy()
            sc_star[:,54,54] = star_new
            sc_planet = scene[i].copy()
            if planet!=None:
                sc_planet[:,int(coords[1][i]),int(coords[0][i])] = planet_new
            for j in range(len(scene[i])):
                FTPSF = np.fft.fft2(np.fft.fftshift(psfs[i,j]))
                FTPSF_C = np.fft.fft2(np.fft.fftshift(psfs_coron[i,j]))
                FTSCENE_S = np.fft.fft2(np.fft.fftshift(sc_star[j]))
                FTSCENE_P = np.fft.fft2(np.fft.fftshift(sc_planet[j]))
                FTCONV_S = FTSCENE_S * FTPSF_C
                FTCONV_P = FTSCENE_P * FTPSF
                convim_s = np.fft.ifftshift(np.fft.ifft2(FTCONV_S))
                convim_p = np.fft.ifftshift(np.fft.ifft2(FTCONV_P))
                scene_conv[i][j] = np.real(convim_s + convim_p)

                convim_sn = np.fft.ifftshift(np.fft.ifft2(FTPSF*FTSCENE_S))
                scene_conv_s[i][j] = np.real(convim_sn)
            scene[i] = sc_star+sc_planet

        scene_conv = np.array(scene_conv)*u.erg/u.cm/u.cm/u.s/u.um
        scene_conv_s = np.array(scene_conv_s)*u.erg/u.cm/u.cm/u.s/u.um
        return scene, scene_conv, scene_conv_s

def Jy_to_flam(flux_Jy,lam_um):
    fnu_cgs = flux_Jy*1.0e-23
    c_cgs = const.c.to('cm/s').value
    lams_cgs = lam_um/1.0e4
    flam_cgs = fnu_cgs*c_cgs/(lams_cgs)**2
    flam_um = flam_cgs/1.0e4
    return flam_um

def flat_mJy_target(flux_mJy):
    #spec_in is input spectrum in mJy
    #lamsin is input wavelength in microns

    lamsin = np.linspace(1.7,5.5,3000)
    spec_in = np.ones(lamsin.shape)*flux_mJy
    fnu_cgs = spec_in*1.0e-26 #now in erg/s/cm^2/Hz
    c_cgs = const.c.to('cm/s').value
    lams_cgs = lamsin/1.0e4
    flam_cgs = fnu_cgs*c_cgs/(lams_cgs)**2
    flam_um = flam_cgs/1.0e4
    targ = Target(lamsin,flam_um)
    targ_bg = Target(lamsin,np.zeros(flam_um.shape))
    return targ, targ_bg

def flat_Lmag_target(Lmag):
    #spec_in is input spectrum in mJy
    #lamsin is input wavelength in microns

    lamsin = np.linspace(1.7,5.5,3000)
    flux_mJy = 10.**(-Lmag/2.5)*288.*1000.
    spec_in = np.ones(lamsin.shape)*flux_mJy
    fnu_cgs = spec_in*1.0e-26 #now in erg/s/cm^2/Hz
    c_cgs = const.c.to('cm/s').value
    lams_cgs = lamsin/1.0e4
    flam_cgs = fnu_cgs*c_cgs/(lams_cgs)**2
    flam_um = flam_cgs/1.0e4
    targ = Target(lamsin,flam_um)
    targ_bg = Target(lamsin,np.zeros(flam_um.shape))
    return targ, targ_bg

def calc_SNR_cube(signal_cube,bkg_cube):
    snrcube = signal_cube/np.sqrt(signal_cube+bkg_cube)
    return snrcube

def calc_SNR_lam_ap(signal_cube,bkg_cube,rlams,yc=54,xc=54):
    xs = np.linspace(0,107,108)
    ys = np.linspace(0,107,108)
    dists = np.array([[np.sqrt((x-xc)**2+(y-yc)**2) for x in xs] for y in ys])
    snrs = []
    for ll in range(len(rlams)):
        lam = rlams[ll].value
        fwhm = lam*1.0e-6/10.0*206265./0.02
        toex = np.where(dists < fwhm)
        sig = np.sum(signal_cube[ll][toex])
        noise = np.sqrt(sig + np.sum(bkg_cube[ll][toex]))
        snrs.append(sig/noise)
    return np.array(snrs)


def calc_SNR_lam_ap_med(signal_cube,bkg_cube,rlams,yc=54,xc=54):
    xs = np.linspace(0,16,17)
    ys = np.linspace(0,17,18)
    dists = np.array([[np.sqrt((x-xc)**2+(y-yc)**2) for x in xs] for y in ys])
    #plt.imshow(dists)
    #plt.scatter(8,9)
    #plt.show()
    #stop
    snrs = []
    for ll in range(len(rlams)):
        lam = rlams[ll].value
        fwhm = lam*1.0e-6/10.0*206265./0.02
        toex = np.where(dists < fwhm)
        sig = np.sum(signal_cube[ll][toex])
        noise = np.sqrt(sig + np.sum(bkg_cube[ll][toex]))
        snrs.append(sig/noise)
    return np.array(snrs)