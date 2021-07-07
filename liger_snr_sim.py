#!/usr/bin/env python

# SNR equation:
# S/N = S*sqrt(T)/sqrt(S + npix(B + D + R^2/t))
# t = itime per frame
# T = sqrt(Nframes*itime)
# S, B, D -> electrons per second
# R -> read noise (electrons)
import os
from math import log10,ceil,sqrt,log
from collections import OrderedDict
import numpy as np
from scipy import integrate,interpolate
from astropy.io import fits
from astropy.modeling import models
import pandas as pd
from photutils import aperture_photometry
from photutils import CircularAperture
import plotly.graph_objects as go
import plotly.express as px
# Liger interal packages
from background_specs import background_specs3
from get_liger_psf import get_liger_psf
from misc_funcs import extrap1d, binnd, get_filterdat, frebin2d

# constants
c_km = 2.9979E5      # km/s
c = 2.9979E10       # cm/s
h = 6.626068E-27    # cm^2*g/s
k = 1.3806503E-16   # cm^2*g/(s^2*K)
Ang = 1E-8          # cm
mu = 1E-4           # cm

def LIGER_ETC(filter = "K", mag = 21.0, flambda=1.62e-19, fint=4e-17, itime = 1.0,
             nframes=1, snr=10.0, aperture=None, gain=3.04,
             readnoise=5., darkcurrent=0.002, scale=0.014,
             resolution=4000, collarea=78.5, positions=[0, 0],
             bgmag=None, efftot=None, mode="imager", calc="snr",
             spectrum="Vega", specinput=None, lam_obs=2.22, line_width=200.,
             png_output=None, source='point_source', profile=None,
             source_size=0.2, csv_output=None, fov=[0.56, 2.24],
             psf_loc=[8.8, 8.8], psf_time=1.4, verb=1, psf_input=False,
             simdir='~/data/Liger/sim/', psfdir='~/data/Liger/sim/', test=0):
    """
    :param filter: broadband filter to use (default: 'K')
    :param mag: magnitude of the point source
    :param itime: integration time per frame (default: 900 s)
    :param nframes: number of observations (default: 1)
    :param snr: signal-to-noise of source
    :param aperture: aperture radius in arcsec
    :param gain: gain in e-/DN
    :param readnoise: read noise in e-, estimated from NIRC2 manual for 64 samples in MCDS mode
    :param darkcurrent: dark current noise in e-/s, estimated from NIRC2 manual
    :param scale: pixel scale (default: 0.004"), sets the IFS mode
    :param collarea: collecting area (m^2) (Keck 78.5)
    :param positions: position of point source
    :param bgmag : the background magnitude (default: sky background
                    corresponding to input filter)
    :param efftot: total throughput
    :param verb: verbosity level
    :param fov: field of view of image
    :param efftot: total throughput
    :param spectrum: spectral shape to use
    :param specinput: the input spectrum in [wavelength, flux] if not one of the 3 main shapes
    :param lam_obs: wavelength for observation for emission line
    :param line_width: emission line width
    :param mode: either "imager" or "ifs"
    :param calc: either "snr" or "exptime"
    :param source: either "point_source" or "extended" if "extended, "profile" is used
    :param profile: if source is "extended" this contains the array of a brightness distribution.
                    If this is None and source is "extended" the profile assumed top-hat.
    :return: Ordered dictionary containing relevant values to calculation.
    """

    #fixed radius 0.2 arc sec
    radius = 0.2
    radius /= scale
    ## Saturation Limit
    sat_limit = 90000
    if spectrum is not None:
        if spectrum.lower() == "vega":
           spectrum = "vega_all.fits"
    ##### READ IN FILTER INFORMATION
    filterdat = get_filterdat(filter, simdir, mode=mode)
    modes = pd.read_csv(simdir + 'info/Liger_modes.csv', header=0)
    filts = np.array([fil.lower() for fil in modes['Filter']])
    wavs = np.where(filts == filter.lower())[0]
    lmin = filterdat["lambdamin"]/10.
    lmax = filterdat["lambdamax"]/10.
    # lmin = lmin.values
    # lmax = lmax.values
    if not isinstance(lmin, str) and not isinstance(lmin, float):
        lmin = lmin[0]
    if not isinstance(lmax, str) and not isinstance(lmax, float):
        lmax = lmax[0]
    if '*' in str(lmin): lmin = str(lmin).replace('*', '')
    if '*' in str(lmax): lmax = str(lmax).replace('*', '')
    lambdamin = float(lmin)*10.
    lambdamax = float(lmax)*10.
    lambdac = np.mean([lambdamin, lambdamax])

    #various scales of lambda/D radius according to scale
    if aperture is None:
        if scale==0.02:
             radiusl=1.*lambdac*206265/(1.26e11)
        elif scale==0.035 :
             radiusl=1.5*lambdac*206265/(1.26e11)
        elif scale==0.05 :
             radiusl=20*lambdac*206265/(1.26e11)
        else:
             radiusl=40*lambdac*206265/(1.26e11)
    else:
        radiusl = aperture
    sizel = 2. * radiusl
    radiusl /= scale

    ## Determine the length of the cube, dependent on filter
    ## (supporting only BB for now)
    wi = lambdamin # Ang
    wf = lambdamax # Ang
    dxspectrum = int(ceil( log10(wf/wi)/log10(1.0+1.0/(resolution*2.0)) ))
    ## resolution times 2 to get nyquist sampled
    crval1 = wi/10.                      # nm
    cdelt1 = ((wf-wi) / dxspectrum)/10.  # nm/channel
    # Throughput calculation
    if efftot is None:
        teltot = 0.80
        aotot = 0.65

        wav  = [830,900,2000,2200,2300,2412] # nm
        #return(fov, type(fov), scale, type(scale))
        imagesize = np.array(fov)/scale

        if mode == "imager":
           tput = [0.631,0.772,0.772,0.813,0.763,0.728] # imager
           if verb > 1: print('Liger imager selected!!!')
        elif scale == 0.014 or scale == 0.031:
          tput = [0.340,0.420,0.420,0.490,0.440,0.400] # IFS lenslet
          if verb > 1: print('Liger lenslet selected!!!')
        else:
            tput = [0.343, 0.465, 0.465, 0.514, 0.482, 0.451]
        w = (np.arange(dxspectrum)+1)*cdelt1 + crval1  # compute wavelength
        #####################################################################
        # Interpolating the Liger throughputs from the PDR-1 Design Description
        # Document (Table 7, page 54)
        #####################################################################
        R = interpolate.interp1d(wav,tput,fill_value='extrapolate')
        eff_lambda = [R(w0) for w0 in w]

        ###############################################################
        # MEAN OF THE INSTRUMENT THROUGHPUT BETWEEN THE FILTER BANDPASS
        ###############################################################
        instot = np.mean(eff_lambda)

        efftot = instot*teltot*aotot
    if verb > 1: print( ' ')
    if verb > 1: print( 'Total throughput (Keck+AO+Liger) = %.3f' % efftot)

    if bgmag:
       backmag = bgmag
    else:
       backmag = filterdat["backmag"] #background between OH lines
       imagmag = filterdat["imagmag"] #integrated BB background
       if mode == "imager": backmag = imagmag ## use the integrated background if specified
    zp = filterdat["zp"]

    if mode.lower() == "ifs":
        psf_wvls = np.array([840, 928, 1026, 988, 1092, 1206, 1149, 1270, 1403, 1474,
                    1629, 1810, 1975, 2182, 2412]) # nm
    if mode.lower() == "imager":
        psf_wvls = np.array([830, 876, 925, 970, 1019, 1070, 1166, 1245, 1330, 1485,
                    1626, 1781, 2000, 2191, 2400]) # nm
    if psf_input is None:
        psf_file, exten_no = get_liger_psf(mode, filter, psf_loc)
        if verb > 1: print('PSF FILE:', psf_file)
        psf_file = os.path.expanduser(psfdir + psf_file)
        pf = fits.open(psf_file)
        psfimage = pf[exten_no].data
        scl = pf[exten_no].header['dp']
        x_im_size, y_im_size = psfimage.shape
        #return dict([('psfsize: ', psfimage.shape)]), [], []
        # image = binnd(np.array(psfimage), [int(ceil(x_im_size * float(scl) / float(scale))),
        #                                   int(ceil(y_im_size * float(scl) / float(scale)))], 'sum')
        image = frebin2d(np.array(psfimage), (int(ceil(x_im_size * float(scl) / float(scale))),
                                           int(ceil(y_im_size * float(scl) / float(scale)))))
        image = image * np.sum(psfimage) / np.sum(image)
    else:
        image = psf_input
    # truncate PSF with image size goes down to 1e-6
    # check in log scale
    # position of center of PSF
    x_im_size,y_im_size = image.shape
    hw_x = x_im_size/2
    hw_y = y_im_size/2
    image /= image.sum()
    psf_extend = np.array(image)
    if source=='extended':
        if profile is None:
            window=300
            obj=np.ones([1500,1500])
            centerx=psf_extend.shape[0]/2
            centery=psf_extend.shape[1]/2
            #psf_extend=psf_extend[centerx-(window/2):centerx+(window/2),centery-(window/2):centery+(window/2)]
            image=np.ones([1500,1500])
        else:
            from astropy.convolution import convolve
            xc, yc = positions
            # image coordinates
            hwbox = np.min([x_im_size/4, np.min(profile.size)-2])
            xp = xc + hw_x
            yp = yc + hw_y
            subimage = image[int(yp - hwbox):int(yp + hwbox + 1), int(xp - hwbox):int(xp + hwbox + 1)]
            image = convolve(profile, subimage)
            image /= image.sum()
    x_im_size, y_im_size = image.shape
    hw_x = x_im_size/2
    hw_y = y_im_size/2
    #  mag = ABmag - 0.91 ; Vega magnitude
    ##########################################
    # convert AB to Vega and vice versa
             # band  eff     mAB - mVega
    ABconv = [["i",  0.7472, 0.37 ],
              ["z",  0.8917, 0.54 ],
              ["Y",  1.0305, 0.634],
              ["J",  1.2355, 0.91 ],
              ["H",  1.6458, 1.39 ],
              ["Ks", 2.1603, 1.85 ]]
    ABwave  = [i[1] for i in ABconv]
    ABdelta = [i[2] for i in ABconv]
    if verb > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ABwave, y=ABdelta))
        fig.update_layout(xaxis_title="Wavelength (microns)", yaxis_title="m$_{\\rm AB}$ - m$_{\\rm Vega}$")
        fig.show()
    R_i = interpolate.interp1d(ABwave,ABdelta)
    R_x = extrap1d(R_i)
    delta = R_x([lambdac/1e4])
    # delta = mAB - mVega
    ##########################################
    if specinput is not None:
        specwave = specinput[0]
        spec = specinput[1]
        flux_phot = np.trapz(spec, x=specwave)
    elif mag is not None:
        # convert to flux density (flambda)
        ABmag = mag + delta
        fnu = 10**(-0.4*(ABmag + 48.60))                 # erg/s/cm^2/Hz
        flambda = fnu*Ang/((lambdac*Ang)**2/c)
        flambda=flambda[0]
        flux_phot = zp*10**(-0.4*mag) # photons/s/m^2
        if source=='extended':
            flux_phot= flux_phot*(scale**2)
    elif flambda is not None:
        # convert to Vega mag
        fnu = flambda/(Ang/((lambdac*Ang)**2/c))
        ABmag = -2.5* log10(fnu) - 48.60
        mag = ABmag - delta
        flux_phot = zp*10**(-0.4*mag) # photons/s/m^2
        if source=='extended':
            flux_phot= flux_phot*(scale**2)
    elif fint is not None:
        # Compute flux_phot from flux
        E_phot = (h*c)/(lambdac*Ang)
        flux_phot=1e4*fint/E_phot
        if source=='extended':
            flux_phot= flux_phot*(scale**2)
    else:
        print('No Flux values provided. Returning.')
        return '', '', ''
    #########################################################################
    #########################################################################
    # comparison:
    #########################################################################
    # http://ssc.spitzer.caltech.edu/warmmission/propkit/pet/magtojy/
    # input: Johnson
    #        20 K-banda
    #        2.22 micron
    # output:
    #        6.67e-29 erg/s/cm^2/Hz
    #        4.06e-19 erg/s/cm^2/Ang
    #########################################################################
    #################################################################
    # tests
    # #################################################################
    # if test:
    #     fnu = 10**(-0.4*(ABmag + 48.60))                 # erg/s/cm^2/Hz
    #     flambda = fnu*Ang/((lambdac*Ang)**2/c)
    #     E_phot = (h*c)/(lambdac*Ang) # erg
    # ########################################################################
    if verb > 1:
        # Vega test spectrum
        if spectrum == "vega_all.fits":
            ext = 0
            spec_file = os.path.expanduser(simdir + "/model_spectra/" + spectrum)
            pf = fits.open(spec_file)
            spec = pf[ext].data
            head = pf[ext].header
            cdelt1 = head["cdelt1"]
            crval1 = head["crval1"]

            nelem = spec.shape[0]
            specwave = (np.arange(nelem))*cdelt1 + crval1  # Angstrom
            #spec /= 206265.**2

        else:
            spectrum = "spec_vega.fits"
            ext = 0
            spec_file = os.path.expanduser(simdir + "/model_spectra/" + spectrum)
            pf = fits.open(spec_file)
            specwave = pf[ext].data[0,:] # Angstrom
            spec = pf[ext].data[1,:]     # erg/s/cm^2/Ang
            nelem = spec.shape[0]

        E_phot = (h*c)/(specwave*Ang) # erg
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=specwave, y=spec/E_phot)) # photons/s/cm^2/Ang
        fig.update_layout(title="Vega photon spectrum", xaxis_title="Wavelength (Angstroms)",
                          yaxis_title="Flux (photons cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)")
        # ABnu = STlamb @ 5492.9 Ang
        STlamb = 3.63E-9*np.ones((nelem)) # erg/cm^2/s/Ang   STlamb = 0
        ABnu = 3.63E-20*np.ones((nelem))  # erg/cm^2/s/Hz    ABnu = 0
        fig.add_trace(go.Scatter(x=specwave, y=STlamb/E_phot))
        fig.add_trace(go.Scatter(x=specwave, y=ABnu/E_phot*(c/(specwave*Ang)**2)*Ang))
        fig.show()

    #print("Calculated from Liger zeropoints")
    E_phot = (h*c)/(lambdac*Ang) # erg
    flux = flux_phot*E_phot*(1./(100*100)) # erg/s/cm^2
    # convert from m**2 to cm**2)
    #########################################################################
    #########################################################################
    if mode.lower() == "ifs":
        #hwbox = 25
        if aperture/scale > 25:
            hwbox = aperture/scale + 1
        elif x_im_size <= 100:
            hwbox = x_im_size/2
        else:
            hwbox = 50
    elif mode == "imager":
        #hwbox = 239
        hwbox = x_im_size/4
        hwbox = 50
    else:
        print('Mode not provided. Returning.')
        return '', '', ''
    # write check for hwbox boundary

    if verb > 1: print(image.shape)
    # center coordinates
    #xp,yp = positions
    xc,yc = positions
    # image coordinates
    xp = xc + hw_x
    yp = yc + hw_y
    # the shape of the PSF image
    # subimage coordinates
    xs = int(xc + hwbox)
    ys = int(yc + hwbox)
    subimage = image[int(yp-hwbox):int(yp+hwbox+1),int(xp-hwbox):int(xp+hwbox+1)]
    # normalize by the full PSF image
    print(np.shape(subimage))
    print(scale)
    print(np.shape(image))

    # to define apertures used throughout the calculations
    radii = np.arange(1,50,1) # pixels
    apertures = [CircularAperture([xs,ys], r=r) for r in radii]
    aperture = CircularAperture([xs,ys], r=radius)
    mask = aperture.to_mask(method='exact')
    #Second Aperture lambda dependent
    aperturel = CircularAperture([xs,ys], r=radiusl)
    maskl = aperturel.to_mask(method='exact')

    ###########################################################################
    ###########################################################################
    # IFS MODE
    ###########################################################################
    ###########################################################################
    if mode.lower() == "ifs":
        bkgd = background_specs3(resolution*2.0, filter, convolve=True, simdir=simdir)
        ohspec = bkgd.backspecs[0,:]
        cospec = bkgd.backspecs[1,:]
        bbspec = bkgd.backspecs[2,:]
        ohspectrum = ohspec*scale**2.0  ## photons/um/s/m^2
        contspectrum = cospec*scale**2.0
        bbspectrum = bbspec*scale**2.0
        backtot = ohspectrum + contspectrum + bbspectrum
        if verb >1:
           print('mean OH: ', np.mean(ohspectrum))
           print('mean continuum: ', np.mean(contspectrum))
           print('mean bb: ', np.mean(bbspectrum))
           print('mean background: ', np.mean(backtot))
        backwave = bkgd.waves/1e4
        wave = np.linspace(wi/1e4,wf/1e4,dxspectrum)
        backtot_func = interpolate.interp1d(backwave,backtot,fill_value='extrapolate')
        backtot = backtot_func(wave)
        if specinput is not None:
            specwave = specinput[0]
            spec = specinput[1]
        elif spectrum.lower() == "flat":
            dxspectrum2 = int(ceil(log10(wf / wi) / log10(1.0 + 1.0 / (resolution * 3.0))))
            specwave = np.linspace(wi / 1e4 - 3. * (wave[1] - wave[0]), wf / 1e4 + 3. * (wave[-1] - wave[-2]),
                                   dxspectrum2)
            spec_temp = np.ones(dxspectrum2)
            intFlux = integrate.trapz(spec_temp, specwave)
            intNorm = flux_phot/intFlux
            spec = spec_temp
        elif spectrum.lower() == "emission":
            dxspectrum2 = int(ceil(log10(wf / wi) / log10(1.0 + 1.0 / (resolution * 3.0))))
            specwave = np.linspace(wi / 1e4 - 3. * (wave[1] - wave[0]), wf / 1e4 + 3. * (wave[-1] - wave[-2]),
                                   dxspectrum2)
            lam_width=lam_obs/c_km*line_width
            instwidth = (lam_obs/resolution)
            width = np.sqrt(instwidth**2+lam_width**2)
            A = flux_phot/(width*np.sqrt(2*np.pi))  #  photons/s/m^2/micron
            spec = A*np.exp(-0.5*((specwave - lam_obs)/width)**2.)
            intFlux = integrate.trapz(spec, specwave)
            intNorm = flux_phot/intFlux
        elif spectrum == "vega_all.fits":
            ext = 0
            spec_file = os.path.expanduser(simdir + "/model_spectra/" + spectrum)
            pf = fits.open(spec_file)
            spec = pf[ext].data
            head = pf[ext].header
            cdelt1 = head["cdelt1"]
            crval1 = head["crval1"]
            nelem = spec.shape[0]
            specwave = (np.arange(nelem))*cdelt1 + crval1  # Angstrom
            specwave /= 1e4 # -> microns
        elif spectrum == "spec_vega.fits":
            ext = 0
            spec_file = os.path.expanduser(simdir + "/model_spectra/" + spectrum)
            pf = fits.open(spec_file)
            specwave = pf[ext].data[0,:] # Angstrom
            spec = pf[ext].data[1,:]     # erg/s/cm^2/Ang
            nelem = spec.shape[0]
            E_phot = (h*c)/(specwave*Ang) # erg
            spec *= 100*100*1e4/E_phot # -> photons/s/m^2/um
            specwave /= 1e4 # -> microns
        else:
            spectrum = 'flat' #assume this if not provided
            spec_temp = np.ones(dxspectrum)
            intFlux = integrate.trapz(spec_temp, wave)
            intNorm = flux_phot / intFlux
            specwave = wave
            spec = spec_temp
        if verb > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=specwave, y=spec))
            fig.show()
        ################################################
        # convolve with the resolution of the instrument
        ################################################
        delt = 2.0*(wave[1]-wave[0])/(specwave[1]-specwave[0])
        if delt > 1:
           stddev = delt/2*sqrt(2*log(2))
           psf_func = models.Gaussian1D(amplitude=1.0, stddev=stddev)
           x = np.arange(4*int(delt)+1)-2*int(delt)
           psf = psf_func(x)
           psf /= psf.sum() # normalize
           spec = np.convolve(spec, psf,mode='same')
        spec_func = interpolate.interp1d(specwave,spec)
        spec_temp = spec_func(wave)
        intFlux = integrate.trapz(spec_temp,wave)
        intNorm = flux_phot/intFlux
        # essentially the output of mkpointsourcecube
        cube = (subimage[np.newaxis]*spec_temp[:,np.newaxis,np.newaxis]).astype(np.float32)
        # photons/s/m^2/um
        cube = intNorm*cube
        if verb > 2:
            # [electrons]
            hdu = fits.PrimaryHDU(cube)
            hdul = fits.HDUList([hdu])
            hdul.writeto('cube.fits',clobber=True)
        # convert the signal and the background into photons/s observed
        # with TMT
        observedCube = cube*collarea*efftot    # photons/s/um
        backtot = backtot*collarea*efftot       # photons/s/um
        # get photons/s per spectral channel, since each spectral
        # channel has the same bandwidth
        if verb > 1:
            print("Observed cube sum = %.2e photons/s/um" % observedCube.sum())
            print("Background cube sum = %.2e photons/s/um" % backtot.sum())
        observedCube = observedCube*(wave[1]-wave[0])
        backtot = backtot*(wave[1]-wave[0])
        if verb > 1:
            print("dL = %f micron" % (wave[1]-wave[0]))
            print("Observed cube sum = %.2e photons/s" % observedCube.sum())
            print("Background cube sum = %.2e photons/s" % backtot.sum())
            print()
        ##############
        # filter curve
        ##############
        # not needed until actual filters are known
        if verb > 1:
            fig = go.Figure()
            p = fig
            #fig.add_trace(go.Scatter(x=wave, filter_tput*cube[:,ys,xs])
            fig.add_trace(go.Scatter(x=wave, y=cube[:,ys,xs],line=dict(color="black")))
            fig.add_trace(go.Scatter(x=wave, y=np.sum(cube, axis=(1,2)), line=dict(color="blue")))
            fig.show()
        if verb > 1:
            print('n wavelength channels: ', len(wave))
            print('channel width (micron): ', wave[1]-wave[0])
            print('mean flux input cube center (phot/s/m^2/micron): %.2e' % np.mean(cube[:, ys, xs]))
            print('mean counts/spectral channel input cube center (phot/s): %.2e' % np.mean(observedCube[:, ys, xs]))
            print('mean background (phot/s): ', np.mean(backtot))
        backgroundCube = np.broadcast_to(backtot[:,np.newaxis,np.newaxis],cube.shape)
        if verb > 1: print(backgroundCube.shape)
        ### Calculate total noise number of photons from detector
        darknoise = darkcurrent       ## electrons/s
        readnoise = readnoise**2.0/itime  ## scale read noise
        # total noise per pixel
        noisetot = darknoise + readnoise
        noise = noisetot
        ### Combine detector noise and background (sky+tel+AO)
        noisetotal = noise + backgroundCube
        totalObservedCube = observedCube * itime * nframes + backgroundCube * itime * nframes + \
                            darkcurrent * itime * nframes + readnoise ** 2.0 * nframes
        if totalObservedCube.max() > sat_limit:
            sat_pix = np.max(totalObservedCube[np.where(totalObservedCube > sat_limit)[0]])
            saturated = len(np.where(totalObservedCube > sat_limit)[0])
        else:
            sat_pix = ''
            saturated = ''
        totalObservedCube = totalObservedCube.astype(np.float64)
        ####################################################
        # Case 1: find s/n for a given exposure time and mag
        ####################################################
        if calc == "snr":
            if verb > 1: print("Case 1: find S/N for a given exposure time and mag")

            signal = observedCube*np.sqrt(itime*nframes)  # photons/s
            # make a background cube and add noise
            # noise = sqrt(S + B + R^2/t)
            noiseCube = np.sqrt(observedCube+noisetotal)
            # SNR cube  = S*sqrt(itime*nframes)/sqrt(S + B+ R^2/t)
            snrCube = signal/noiseCube
            if verb > 2:
                hdu = fits.PrimaryHDU(snrCube)
                hdul = fits.HDUList([hdu])
                hdul.writeto('snrCube.fits',clobber=True)
            peakSNR = str("%0.1f" %np.max(snrCube))
            medianSNR = str("%0.1f" %np.median(snrCube))
            meanSNR = str("%0.1f" %np.mean(snrCube))
            medianSNRl = ""
            meanSNRl = ""
            minexptime = ""
            medianexptimel = ""
            meanexptimel = ""
            flatarray=np.ones(snrCube[0,:,:].shape)
            maskimg= mask.multiply(flatarray)
            masklimg= maskl.multiply(flatarray)
            fluxval = []
            flux_aper = []
            noiseval = []
            noise_aper = []
            apert = CircularAperture([xs, ys], radiusl)
            apermask = apert.to_mask(method='exact')
            cubesize = np.shape(snrCube)
            onesimg = np.ones((cubesize[1], cubesize[2]))
            onesmask = apermask.multiply(onesimg)
            for i in range(dxspectrum):
                fluxslice = apermask.multiply(observedCube[i, :, :])
                fluxslice[np.where(onesmask == 0)] += np.min(fluxslice[np.where(onesmask > 0)])
                fluxval.append(np.sum(fluxslice))
                flux_aper.append(fluxslice)
                noiseslice = apermask.multiply(noisetotal[i, :, :])
                noiseslice[np.where(onesmask == 0)] += np.min(noiseslice[np.where(onesmask > 0)])
                noiseval.append(np.sum(noiseslice))
                noise_aper.append(noiseslice)
            fluxval = np.array(fluxval)
            noiseval = np.array(noiseval)
            flux_aper = np.array(flux_aper)
            noise_aper = np.array(noise_aper)
            flux_int = np.trapz(fluxval, x=wave)
            noise_int = np.trapz(noiseval, x=wave)
            snr_int = flux_int*np.sqrt(itime*nframes)/np.sqrt(flux_int+noise_int)
            totalSNRl = str("%0.1f" %snr_int)                # integrated aperture SNR at pre-defined fixed aperture
            totalexptimel = ""                               # integrated aperture exptime at pre-defined fixed aperture
            snr_aper = flux_aper * np.sqrt(itime*nframes) / np.sqrt(flux_aper + noise_aper)
            snr_cutout = []
            snr_cutout_aper = []
            snr_cutout_aperselect = []
            snr_cutoutl = []
            snr_cutout_aperl = []
            snr_cutout_aperlselect = []
            for n in range(dxspectrum):
                snr_cutout.append(mask.cutout(snrCube[n,:,:]))
                snr_cutout_tmp = mask.multiply(snrCube[n,:,:])
                snr_cutout_tmp_select=snr_cutout_tmp[np.where(maskimg>0)]
                snr_cutout_aper.append(snr_cutout_tmp)
                snr_cutout_aperselect.append(snr_cutout_tmp_select)
            for n in range(dxspectrum):
                snr_cutoutl.append(maskl.cutout(snrCube[n,:,:]))
                snr_cutout_tmp = maskl.multiply(snrCube[n,:,:])
                snr_cutout_tmp_select=snr_cutout_tmp[np.where(masklimg>0)]
                snr_cutout_aperl.append(snr_cutout_tmp)
                snr_cutout_aperlselect.append(snr_cutout_tmp_select)

            if verb > 1: print(np.shape(snr_aper))
            if verb > 1: print(np.shape(fluxval*np.sqrt(itime*nframes)/np.sqrt(fluxval+noiseval)))

            ###########################
            # summation of the aperture
            ###########################
            data_cutout_aperl = []
            noise_cutout_aperl = []
            for n in range(dxspectrum):
                data_cutout_tmp = maskl.multiply(signal[n,:,:])
                noise_cutout_tmp = maskl.multiply(noiseCube[n,:,:])
                data_cutout_aperl.append(data_cutout_tmp)
                noise_cutout_aperl.append(noise_cutout_tmp)
            ###########################
            # summation of the aperture
            ###########################
            snr_chl = fluxval*np.sqrt(itime*nframes)/np.sqrt(fluxval + noiseval)
            if verb > 1: print('S/N (aperture = %.4f") = %.4f' % (sizel, snr_int))
            ###############
            # Main S/N plot
            ###############
            if verb > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=wave, y=snr_chl, line=dict(color="black"),
                                               name="Total Flux [Aperture Diameter: "+"{:.3f}".format(sizel)+'"]'))
                ############
                # inset plot
                ############
                fig.add_trace(go.Scatter(x=wave, y=snrCube[:, ys, xs], name="Peak Flux"))
                fig.add_trace(go.Scatter(x=wave, y=np.mean(flux_aper, axis=(1, 2)) * np.sqrt(itime* nframes) /
                              np.sqrt(np.mean(flux_aper, axis=(1, 2)) + np.mean(noise_aper, axis=(1, 2))),
                # l3, = fig.add_trace(go.Scatter(x=wave, np.mean(snr_aper, axis=(1, 2)),
                              name="Mean Flux [Aperture Diameter: "+"{:.3f}".format(sizel)+'"]'))
                fig.add_trace(go.Scatter(x=wave, y=np.median(snr_aper, axis=(1, 2)),
                              name="Median Flux [Aperture Diameter: "+"{:.3f}".format(sizel)+'"]'))
                # leg = p.legend([l1,l2,l3,l4], ["Total Flux [Aperture : "+"{:.3f}".format(sizel)+'"]',
                #                                "Peak Flux", "Mean Flux [Aperture : "+"{:.3f}".format(sizel)+'"]',
                #                                "Median Flux [Aperture : "+"{:.3f}".format(sizel)+'"]'],
                #                loc=1,numpoints=1,prop={'size': 6})
                fig.update_layout(xaxis_title="Wavelength (microns)", yaxis_title="SNR",
                                  plot_bgcolor='rgba(0,0,0,0)', legend=dict(x=0.45, y=0.15))
                                              #float(np.max(wave) - 0.9 * (np.max(wave)-np.min(wave))), y =2.5))
                                              #y=float(np.max(snr_chl) - 0.9 * (np.max(snr_chl)-np.min(snr_chl)))))
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                if png_output:
                    fig.write_image(png_output)
                if csv_output:
                    csvarr = np.array([wave,snrCube[:,ys,xs], np.median(snr_aper, axis=(1,2)),
                                     np.mean(snr_aper, axis=(1,2)), snr_chl]).T
                else:
                    csvarr=""
            # model + background + noise
            # [electrons]
            simCube_tot = np.random.poisson(lam=totalObservedCube, size=totalObservedCube.shape).astype("float64")
            # divide back by total integration time to get the simulated image
            simCube = simCube_tot/(itime*nframes) # [electrons/s]
            simCube_DN = simCube_tot/gain # [DNs]
            if verb > 2:
                # [electrons]
                hdu = fits.PrimaryHDU(simCube_tot)
                hdul = fits.HDUList([hdu])
                hdul.writeto('simCube_tot.fits',clobber=True)
                # [electrons/s]
                hdu = fits.PrimaryHDU(simCube)
                hdul = fits.HDUList([hdu])
                hdul.writeto('simCube.fits',clobber=True)
                # [DNs]
                hdu = fits.PrimaryHDU(simCube_DN)
                hdul = fits.HDUList([hdu])
                hdul.writeto('simCube_DN.fits',clobber=True)
        #######################################################
        # Case 2: find integration time for a given s/n and mag
        #######################################################
        elif calc == "exptime":
            if verb > 1: print("Case 2: find integration time for a given S/N and mag")
            # snr = observedCube*np.sqrt(itime*nframes)/np.sqrt(observedCube+noisetotal)
            # itime * nframes =  (snr * np.sqrt(observedCube+noisetotal)/observedCube)**2
            totime =  (snr * np.sqrt(observedCube+noisetotal)/observedCube)**2
            peakSNR = ""
            medianSNR = ""
            meanSNR = ""
            medianSNRl = ""
            meanSNRl = ""
            minexptime = str("%0.1f" %np.min(totime))
            medianexptimel = str("%0.1f" %np.median(totime))
            meanexptimel = str("%0.1f" %np.mean(totime))
            totalSNRl = ""                              # integrated aperture SNR at pre-defined fixed aperture
            flatarray=np.ones(totime[0,:,:].shape)
            maskimg= mask.multiply(flatarray)
            masklimg= maskl.multiply(flatarray)
            apert = CircularAperture([xs, ys], radiusl)
            apermask = apert.to_mask(method='exact')
            cubesize = np.shape(observedCube)
            onesimg = np.ones((cubesize[1], cubesize[2]))
            non_zero = apermask.multiply(onesimg)
            fluxval = []
            flux_aper = []
            noiseval = []
            noise_aper = []
            for i in range(dxspectrum):
                fluxslice = apermask.multiply(observedCube[i, :, :])
                fluxslice[np.where(non_zero == 0)] += np.min(fluxslice[np.where(non_zero > 0)])
                fluxval.append(np.sum(fluxslice))
                flux_aper.append(fluxslice)
                noiseslice = apermask.multiply(noisetotal[i, :, :])
                noiseslice[np.where(non_zero == 0)] += np.min(noiseslice[np.where(non_zero > 0)])
                noiseval.append(np.sum(noiseslice))
                noise_aper.append(noiseslice)
            fluxval = np.array(fluxval)
            noiseval = np.array(noiseval)
            flux_aper = np.array(flux_aper)
            noise_aper = np.array(noise_aper)
            flux_int = np.trapz(fluxval, x=wave)
            noise_int = np.trapz(noiseval, x=wave)
            itime_aper = (snr * np.sqrt(flux_aper + noise_aper) / flux_aper)**2
            intflux_itime = (snr * np.sqrt(flux_int + noise_int) / flux_int)**2
            totalSNRl = ""
            totalexptimel = str("%0.1f" %intflux_itime) # integrated aperture exptime at pre-defined fixed aperture
            totime_cutout = []
            totime_cutout_aper = []
            totime_cutout_aperselect = []
            totime_cutoutl = []
            totime_cutout_aperl = []
            totime_cutout_aperlselect = []
            for n in range(dxspectrum):
                totime_cutout.append(mask.cutout(totime[n,:,:]))
                totime_cutout_tmp = mask.multiply(totime[n,:,:])
                totime_cutout_tmp_select=totime_cutout_tmp[np.where(maskimg > 0)]
                totime_cutout_aper.append(totime_cutout_tmp)
            for n in range(dxspectrum): 
                totime_cutout_aperselect.append(totime_cutout_tmp_select)
                totime_cutoutl.append(maskl.cutout(totime[n,:,:]))
                totime_cutout_tmp = maskl.multiply(totime[n,:,:])
                totime_cutout_tmp_select=totime_cutout_tmp[np.where(masklimg > 0)]
                totime_cutout_aperl.append(totime_cutout_tmp)
                totime_cutout_aperlselect.append(totime_cutout_tmp_select)
            totime_cutout = np.array(totime_cutout)
            totime_cutout_aper = np.array(totime_cutout_aper)
            totime_cutoutl = np.array(totime_cutoutl)
            totime_cutout_aperl = np.array(totime_cutout_aperl)
            totime_cutout_aperselect = np.array(totime_cutout_aperselect,dtype=object)
            totime_cutout_aperlselect = np.array(totime_cutout_aperlselect,dtype=object)
            ############################
            # exposure time for aperture 
            ############################
            data_cutout_aperl = []
            noise_cutout_aperl = []
            for n in range(dxspectrum):
                data_cutout_tmp = maskl.multiply(observedCube[n,:,:])
                noise_cutout_tmp = maskl.multiply(noisetotal[n,:,:])
                data_cutout_aperl.append(data_cutout_tmp)
                noise_cutout_aperl.append(noise_cutout_tmp)
            ###########################
            # summation of the aperture
            ###########################
            totime_chl = (snr * np.sqrt(fluxval + noiseval) / fluxval) ** 2
            if verb > 1:
                print("Min time (peak flux) = %.4f seconds" % np.min(totime))
                print("Median time (median aperture flux) = %.4f seconds" % np.median(itime_aper, axis=(1, 2)))
                print("Mean time (mean aperture flux) = %.4f seconds" % np.mean(itime_aper, axis=(1, 2)))
                print('Time (aperture = %.4f") = %.4f' % (sizel, totalexptimel))
            ####################
            # Main exposure plot
            ####################
            if verb > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=wave, y=totime_chl, line=dict(color="black"),
                                               name="Total Flux [Aperture : "+"{:.3f}".format(sizel)+'"]'))
                fig.update_layout(xaxis_title="Wavelength (microns)", yaxis_title="Total Integration time (seconds)",
                                  plot_bgcolor='rgba(0,0,0,0)', legend=dict(x=0.45, y=0.15))
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                ############
                # inset plot
                ############
                fig.add_trace(go.Scatter(x=wave, y=totime[:,ys,xs], name="Peak Flux"))
                fig.add_trace(go.Scatter(x=wave, y=(snr * np.sqrt(np.mean(flux_aper, axis=(1,2)) + np.mean(noise_aper, axis=(1,2)))
                                     / np.mean(flux_aper, axis=(1,2)))**2,
                              name="Mean Flux  [Aperture : "+"{:.3f}".format(sizel)+'"]', visible='legendonly'))
                fig.add_trace(go.Scatter(x=wave, y=np.median(itime_aper, axis=(1, 2)),
                              name="Median Flux [Aperture : "+"{:.3f}".format(sizel)+'"]', visible='legendonly'))
                # leg = p.legend([l1, l2, l3, l4], ["Total Flux [Aperture : "+"{:.3f}".format(sizel)+'"]',
                #           "Peak Flux","Mean Flux  [Aperture : "+"{:.3f}".format(sizel)+'"]',
                #           "Median Flux [Aperture : "+"{:.3f}".format(sizel)+'"]'], loc=1,numpoints=1,prop={'size': 6})
                if png_output:
                    fig.savefig(png_output, dpi=200)
                if csv_output:    
                    csvarr = np.array([wave, totime[:,ys,xs], np.median(itime_aper, axis=(1,2)), 
                                       np.mean(itime_aper, axis=(1,2)), totime_chl]).T
                    # np.savetxt(csv_output, csvarr, delimiter=',', header="Wavelength(microns),Int_Time_PeakFlux(s),
                    # Int_Time_MedianFlux(s),Int_Time_MeanFlux(s),Int_Time_Total_Aperture_Flux(s)", comments="",
                    # fmt='%.4f')
                else:
                    csvarr = ""
            if verb > 1:
                fig = go.Figure()
                px.imshow(totime[0,:,:])
                fig.show()
    ###########################################################################
    ###########################################################################
    # IMAGER MODE
    ###########################################################################
    ###########################################################################
    else:
        csvarr = ""
        # Scale by the zeropoint flux
        subimage *= flux_phot
        ############################# NOISE ####################################
        # Calculate total background number of photons for whole tel aperture
        #efftot = effao*efftel*effLiger #total efficiency
        if verb > 1: print('background magnitude: ', backmag)
        phots_m2 = (10**(-0.4*backmag)) * zp # phots per sec per m2
        #print(phots_m2)
        # divide by the number of spectral channels if it's not an image
        phots_tel = phots_m2 * collarea     # phots per sec for TMT
        #phots_int = phots_tel               # phots per sec
        #phots_chan = phots_int
        # phots from background per square arcsecond through the telescope
        phots_back = efftot*phots_tel
        #background = sqrt(phots_back*scale*scale) #photons from background per spaxial^2
        background = phots_back*scale*scale #photons from background per spaxial^2
        ###################################################################################################
        ### Calculate total noise number of photons from detector
        darknoise = darkcurrent       ## electrons/s
        readnoise = readnoise**2.0/itime  ## scale read noise
                                        # total noise per pixel
        noisetot = darknoise + readnoise
        noise = noisetot
        ### Combine detector noise and background (sky+tel+AO)
        noisetotal = noise + background
        if verb > 1:
            print('detector noise (e-/s): ', noise)
            print('total background noise (phot/s):', background)
            print('Total Noise (photons per pixel^2)= ', noisetotal)
            print('  ')
        ###################################################################################################
        ## put in the TMT collecting area and efficiency
        tmtImage = subimage*collarea*efftot

        ####################################################
        # Case 1: find s/n for a given exposure time and mag
        ####################################################
        # model + background
        totalObserved = tmtImage * itime * nframes + background * itime * nframes + darkcurrent * itime * nframes + readnoise * itime * nframes
        totalObserved_itime = tmtImage * itime + background * itime + darkcurrent * itime + readnoise * itime
        # model + background + noise
        # [electrons]
        simImage_tot = np.random.poisson(lam=totalObserved, size=totalObserved.shape).astype("float64")
        simImage_tot_itime = np.random.poisson(lam=totalObserved_itime, size=totalObserved_itime.shape).astype(
            "float64")
        # divide back by total integration time to get the simulated image
        simImage = simImage_tot / (itime * nframes)  # [electrons/s]
        simImage_DN = simImage_tot_itime / gain  # [DNs]
        simImage_sat = totalObserved_itime + 1.96 * np.sqrt(totalObserved_itime)
        imsize = np.shape(simImage)
        logscale = lambda a, im: np.log10(a * (im / np.max(im)) + 1.0) / (np.log10(a))
        # fig = px.imshow(logscale(1000.,simImage_tot[int(imsize[0] / 2. - imsize[0] / 7.):int(imsize[0] / 2. + imsize[0] / 7.),
        #          int(imsize[1] / 2. - imsize[0] / 7.):int(imsize[1] / 2. + imsize[1] / 7.)]),
        #          origin='lower', labels=dict(color='log10( 1000 * Image / max(image) + 1) / 100'))
        fig = px.imshow(np.log10(simImage_tot[int(imsize[0] / 2. - imsize[0] / 7.):int(imsize[0] / 2. + imsize[0] / 7.),
                 int(imsize[1] / 2. - imsize[0] / 7.):int(imsize[1] / 2. + imsize[1] / 7.)]))
        fig.update_layout(
            xaxis_title='Simulated Image (Central ' + str(100. / 7. * scale)[0:4] + ' arcseconds)',
            coloraxis_colorbar=dict(
                title="log10( Electrons )"
            ))
        if simImage_sat.max() > sat_limit:
            sat_pix = np.max( simImage_sat[np.where(simImage_sat > sat_limit)[0]] )
            saturated = len(np.where(simImage_sat > sat_limit)[0])
        else:
            saturated = ''
            sat_pix = ''
        if calc == "snr":
            signal = tmtImage*np.sqrt(itime*nframes)  # photons/s
            noisemap = np.sqrt(tmtImage+noisetotal)
            snrMap = signal/noisemap
            if verb > 1:
                print("Case 1: find S/N for a given exposure time and mag")
                fig = go.Figure()
                X = snrMap.flatten()
                x0 = np.min(X)
                x1 = np.max(X)
                bins = 50
                px.histogram(bins, x=X, range=(x0,x1), histtype='stepfilled',
                                        color="yellow", alpha=0.3)
                fig.update_layout(xaxis_title="Signal/Noise", yaxis_title="Number of pixels")
                fig.update_yaxes(type="log")
                fig.show()
                fig = px.imshow(snrMap)
                fig.show()
            if verb > 2:
                hdu = fits.PrimaryHDU(snrMap)
                hdul = fits.HDUList([hdu])
                hdul.writeto('snrImage.fits',clobber=True)
                hdu = fits.PrimaryHDU(totalObserved)
                hdul = fits.HDUList([hdu])
                hdul.writeto('new2.fits',clobber=True)
                # [electrons]
                hdu = fits.PrimaryHDU(simImage_tot)
                hdul = fits.HDUList([hdu])
                hdul.writeto('simImage_tot.fits',clobber=True)
                # [electrons/s]
                hdu = fits.PrimaryHDU(simImage)
                hdul = fits.HDUList([hdu])
                hdul.writeto('simImage.fits',clobber=True)
                # [DNs]
                hdu = fits.PrimaryHDU(simImage_DN)
                hdul = fits.HDUList([hdu])
                hdul.writeto('simImage_DN.fits',clobber=True)
            # Sky background counts
            data_cutout = mask.cutout(snrMap)
            data_cutoutl = maskl.cutout(snrMap)
            data_cutout_aper = mask.multiply(snrMap) # in version 0.4 of photutils
            data_cutout_aperl = maskl.multiply(snrMap) # in version 0.4 of photutils
            flatarray=np.ones(snrMap.shape)
            maskimg = mask.multiply(flatarray)
            masklimg = maskl.multiply(flatarray)
            if verb > 1:
                print("Peak S/N = %.4f" % np.max(snrMap))
                print("Median S/N = %.4f" % np.median(data_cutout_aper))
                print("Mean S/N = %.4f" % np.mean(data_cutout_aper))
                fig = px.imshow(data_cutout_aper)
                fig.show()
            ###########################
            # summation of the aperture
            ###########################
            data_cutout_aper = mask.multiply(tmtImage)  # in version 0.4 of photutils
            data_cutout_aperl = maskl.multiply(tmtImage)  # in version 0.4 of photutils
            noise_cutout_aperl = maskl.multiply(flatarray) * noisetotal
            aper_totsuml = data_cutout_aperl.sum()+noise_cutout_aperl.sum()
            aper_suml = data_cutout_aperl.sum()
            #Change to more classical sum for SNR rather than using phot_table
            snr_int=(aper_suml*np.sqrt(nframes*itime))/(np.sqrt(aper_totsuml))
            if verb > 1:
                print('S/N (aperture = %.4f") = %.4f' % (sizel, snr_int))
                phot_table = aperture_photometry(signal, apertures, error=noisemap)
                dn     = np.array([phot_table["aperture_sum_%i" % i] for i in range(len(radii))])
                dn_err = np.array([phot_table["aperture_sum_err_%i" % i] for i in range(len(radii))])
                fig = go.Figure()
                fig.update_layout(xaxis_title="Aperture radius [pixels]", yaxis_title="Counts [photons/s/aperture]")
                fig.show()
            peakSNR = str("%0.2f" % np.max(snrMap))
            medianSNR = str("%0.2f" % np.median(data_cutout_aper[np.where(maskimg>0)]))
            meanSNR = str("%0.2f" % np.mean(data_cutout_aper[np.where(maskimg>0)]))
            medianSNRl = str("%0.2f" % np.median(data_cutout_aperl[np.where(masklimg>0)]))
            meanSNRl = str("%0.2f" % np.mean(data_cutout_aperl[np.where(masklimg>0)]))      
            totalSNRl = str("%0.2f" % snr_int)                      # integrated aperture SNR at pre-defined fixed aperture
            minexptime = ""
            medianexptime = ""
            meanexptime = ""
            medianexptimel = ""
            meanexptimel = ""       
            totalexptimel = ""                                      # integrated aperture exptime at pre-defined fixed aperture
            if verb > 1:
                print("Peak S/N = %.4f" % np.max(snrMap))
                print("Median S/N = %.4f" % np.median(data_cutout_aper))
                print("Mean S/N = %.4f" % np.mean(data_cutout_aper))
                fig = px.imshow(data_cutout_aper)
                fig.show()
        #######################################################
        # Case 2: find integration time for a given s/n and mag
        #######################################################
        elif calc == "exptime":
            if verb > 1:
                print("Case 2: find integration time for a given S/N and mag")
            totime = (snr * np.sqrt(tmtImage+noisetotal)/tmtImage)**2
            data_cutout = mask.cutout(totime)
            data_cutoutl = maskl.cutout(totime)
            data_cutout_aper = mask.multiply(totime)
            data_cutout_aperl = maskl.multiply(totime)
            flatarray = np.ones(totime.shape)
            maskimg = mask.multiply(flatarray)
            masklimg = maskl.multiply(flatarray)
            minexptime = str("%0.1f" %np.min(totime))
            medianexptime = str("%0.1f" %np.median(data_cutout_aper[np.where(maskimg>0)]))
            meanexptime = str("%0.1f" %np.mean(data_cutout_aper[np.where(maskimg>0)]))
            medianexptimel = str("%0.1f" %np.median(data_cutout_aperl[np.where(masklimg>0)]))
            meanexptimel = str("%0.1f" %np.mean(data_cutout_aperl[np.where(masklimg>0)]))
            peakSNR = ""
            medianSNR = ""
            meanSNR = ""
            medianSNRl = ""
            meanSNRl = ""
            totalSNRl = ""                          # integrated aperture SNR at pre-defined fixed aperture
            if verb > 1:
                print("Min time (peak flux) = %.4f seconds" % np.min(totime))
                print("Median time (median aperture flux) = %.4f seconds" % np.median(data_cutout_aper))
                print("Mean time (mean aperture flux) = %.4f seconds" % np.mean(data_cutout_aper))
                print(totime.shape)
                fig = go.Figure()
                p = fig
                fig = px.imshow(totime[0,:])
                fig.show()
            flatarray=np.ones(tmtImage.shape)
            # exposure time for aperture
            data_cutout = mask.cutout(tmtImage)
            data_cutoutl = maskl.cutout(tmtImage)
            data_cutout_aper = mask.multiply(tmtImage)
            data_cutout_aperl = maskl.multiply(tmtImage)
            noise_cutout_aperl = maskl.multiply(flatarray)*noisetotal
            
            aper_totsuml = data_cutout_aperl.sum()+noise_cutout_aperl.sum()
            aper_suml = data_cutout_aperl.sum()
            ###########################
            # summation of the aperture
            ###########################
            totimel =  (snr * np.sqrt(aper_totsuml)/aper_suml)**2
            if verb > 1: print('Time (aperture = %.4f") = %.4f' % (sizel, totimel))
            totalexptimel = str("%0.1f" %totimel) # integrated aperture exptime at pre-defined fixed aperture
    if calc == "exptime":
        inputstr = 'Input SNR'
        inputvalue = str(snr)
    else:
        inputstr = 'Input integration time [s]'
        inputvalue = str(nframes*itime)

    if mode == 'imager':
        saturatedstr = str(saturated)
        resolutionstr = ''
        aperture_str = ''
    else:
        saturatedstr = str(saturated)
        resolutionstr = str(resolution)
        aperture_str = "{:.3f}".format(sizel)+'"'

    if source == "extended":
        magadd = '[per square arcsecond]'
        if saturated > 0:
            saturatedstr = 'True'
        else:
            saturatedstr = 'False'
    else:
        magadd = ''
        
    if mag is not None:
        magstr = str(mag)
    if fint is not None:
        fintstr = str("%0.4e" %fint)
    if flambda is not None:
        flambdastr = str("%0.4e" %flambda)
    # (inputstr, inputvalue), ('Filter', str(filter)),
    # ('Central Wavelength [microns]', "{:.3f}".format(lambdac * .0001)),
    # ('Resolution', resolutionstr), ('Magnitude of Source [Vega]' + magadd, magstr),
    # ("Flux density of Source [erg/s/cm^2/Ang]", flambdastr),
    # ("Integrated Flux over bandpass [erg/s/cm^2]", fintstr), ("Aperture Size", aperture_str),

    jsondict = OrderedDict([('Peak Value of SNR', peakSNR),
                            ('SNR for Integrated Total Flux (Aperture Radius = ' + "{:.3f}".format(sizel/2.) + '")', totalSNRl),
                            ('Median Value of SNR (Aperture Radius = ' + "{:.3f}".format(sizel/2.) + '")', medianSNRl),
                            ('Mean Value of SNR (Aperture Radius = ' + "{:.3f}".format(sizel/2.) + '")', meanSNRl),
                            ('Median Value of SNR (Aperture Radius = 0.2")', medianSNR),
                            ('Mean Value of SNR (Aperture Radius = 0.2")', meanSNR),
                            ('Total integration time [s] to achieve SNR for '
                             'Peak Flux ', minexptime), (
                            'Total integration time [s] to achieve SNR for '
                            'Median Flux (Aperture Radius = ' + "{:.3f}".format(sizel/2.) + '")',
                            medianexptimel), (
                            'Total integration time [s] to achieve SNR for '
                            'Mean Flux (Aperture Radius = ' + "{:.3f}".format(sizel/2.) + '")',
                            meanexptimel), (
                            'Total integration time [s] to achieve SNR for '
                            'Total Integrated Flux (Aperture Radius = ' + "{:.3f}".format(sizel/2.) + '")',
                            totalexptimel), ('Saturated Pixels', saturatedstr), ('Peak Saturated Pixel Electrons',
                                                                                 str(sat_pix))])
    return jsondict, fig, csvarr







