#!/usr/bin/env python


#NAME:
#	BACKGROUND_SPECS3.PRO
#
#PURPOSE:
#	Produce background spectra (OH, continuum sky, and thermal) 
#	given a resolving power and temperature 
#	Typically ran outside of simulator for testing
#	
#INPUT:
#	RES: resolving power
#	FILTER: Filter to scale by
#
#KEYWORDS:
#	T_tel: Temperature of telescope (K)
#       T_atm: Temperature of atmosphere (K)
#	T_aos: Temperature of adaptive optic system (K)
#	T_zod: Temperature of zodical emission (K)#
#	Em_tel: Emissivity of telescope 
#       Em_atm: Emissivity of atmosphere 
#	Em_aos: Emissivity of adaptive optic system 
#	Em_zod: Emissivity of zodical emission#
#
#       convolve - convolve the OH spectrum to the correct spectral
#                  resolution (default is set)
#       noconvolve - do not convolve the OH spectrum
#       collarea - collecting area, by default set to TMT
#       nobbnorm - don't normalize the BB to zero mag (use absolute flux)
#       ohsim    - use simulated OH lines
#
#OUTPUT: Individual spectrum of each background component (tel,atm,ao)
#	and one final spectrum with all background combined with OH emission
#	(fits images) NOTE: will be in units of photons/s/m^2/arcsec/micron
#
#REVISION HISTORY:
# December 2008	 - Shelley Wright: Created.
# Feb 2010 - Modified to use better units and conversions for a given filter
# June 2010 - Added in Zodical emission BB
# 2011-02-08 - T. Do - slight modification to use the integrated flux
#              instead of the total flux, so start out with the zero
#              point in photons/s/m^2
# 2011-10-03 - T. Do - fullbb, fullcont, and fulloh spectra will now
# be returned scaled by the zeropoint of the filter used. This will be
# useful to get the background magnitude scaled by one of the filters
# 2012-04-11 - T. Do - added 'fullback' keyword to return the full
# back ground. Fixed bug where full background is returned instead of
# just within the filter range
# 2012-11-06 - Changed the procedure to use the sky background from
#              Gemini and with blackbody components for the telescope
#              and AO system. Also, by default with convolve the OH
#              lines unless specifically set not to with
#              'noconvolve'. Old 'convolve' keyword is left in place
#              for convenience. 
# 2012-11-10 - T. Do - Changed the default TMT telescope and AO system
#              temperature and emissivity to the values in
#              David Anderson's TMT report "SCIENTIFIC IMPACT
#              OF CHANGING THE NFIRAOS OPERATING TEMPERATURE"
# 2015-07-17 - D. Marshall - Fixed bug with 'geminifile' path slashes
#
#-

import os
from misc_funcs import extrap1d
from math import log, log10, ceil, floor, exp, sqrt
import numpy as np
from scipy import integrate, interpolate
import pandas as pd
import matplotlib.pyplot as plt
from astropy.modeling import models
from astropy.io import fits
from misc_funcs import get_filterdat

class background_specs3():

    def __init__(self, resolution, filter, T_tel = 275, T_atm = 258.0,
                 T_aos = 273.0, T_zod=5800.0, Em_tel = 0.20, Em_atm = 0.2,
                 Em_aos=0.35, Em_zod = 3e-14*49.0, fullbb = False, airmass='10', vapor='15',
                 fullcont=False, fulloh = False, fullback = False, no_oh=True,
                 convolve = True, noconvolve = False, ohsim=True, verb = 0,
                 simdir='~/data/osiris/sim/', filteronly = False):
    # ignore the atmospheric BB and Zodiacal light for now since
    #we're going to be using Gemini
    ##some parameters
    #h=6.626e-34 #J*s
    #c=3.0e8     #m/s^2
    #k=1.38e-23  #J/K
    #hc=1.986e-16# Planck constant times speed of light (erg cm)
    
        h = 6.626e-27 #erg*s
        c = 3.0e10    #cm s^-2
        k = 1.38e-16  #boltman's constant (erg/K)

        ## Need to define zeropoint ###
        ### READ IN DATA FOR GIVEN FILTER FOR PSF (BROADBAND) 

        ##### READ IN FILTER INFORMATION
        filterdat = get_filterdat(filter, simdir, mode='IFS')
        modes = pd.read_csv(simdir + 'info/Liger_modes.csv', header=0)
        filts = np.array([fil.lower() for fil in modes['Filter']])
        wavs = np.where(filts == filter.lower())[0]
        lmin = filterdat["lambdamin"] / 10.
        lmax = filterdat["lambdamax"] / 10.
        # lmin = lmin.values
        # lmax = lmax.values
        if not isinstance(lmin, str) and not isinstance(lmin, float):
            lmin = lmin[0]
        if not isinstance(lmax, str) and not isinstance(lmax, float):
            lmax = lmax[0]
        if '*' in str(lmin): lmin = str(lmin).replace('*', '')
        if '*' in str(lmax): lmax = str(lmax).replace('*', '')
        lambdamin = float(lmin) * 10.
        lambdamax = float(lmax) * 10.
        lambdac = np.mean([lambdamin, lambdamax])

        dxspectrum = 0
        if filteronly:
            wi = lambdamin
            wf = lambdamax
        else:
            wi = 8000.  #Angstroms
            wf = 25000. #Angstroms

        sterrad = 2.35e-11 # sterradians per square arcsecond
        
        ## CREATE THE SIZE OF THE SPECTRUM
        #determine length in pixels of complete spectrum(9000 to 25000), dxspectrum
        dxspectrum = int(ceil(log10(wf/wi)/log10(1.0+1.0/resolution) ) )
        #print(dxspectrum)
        wave = np.linspace(wi,wf,dxspectrum)
        ## READ IN OH Lines ( the resolving power of this OH sky lines is around R=2600)
        #openr,ohlines_file,/get_lun,simdir+"/info/ohlineslist.dat"
        #woh=0.0 #initialize woh for search in ohlineslist.dat
        #woh is last wavelength read in .dat
        # Generate arrays for loop below
        bbtel = np.zeros(dxspectrum)	#telescope blackbody spectrum
        bbaos = np.zeros(dxspectrum)	#AO blackbody spectrum
        bbatm = np.zeros(dxspectrum)	#ATM blackbody spectrum
        bbspec = np.zeros(dxspectrum)	#TOTAL blackbody spectrum
        bbzod = np.zeros(dxspectrum)
        ohspec = np.zeros(dxspectrum)	#OH lines
        contspec = np.zeros(dxspectrum)	#continuum of sky 
        wavelength = np.zeros(dxspectrum)
        bbspec_gem = np.zeros(dxspectrum)
        ## Loop over the first and last pixel of the complete spectrum
        ## i (pixel along complete spectrum)
        #for i=0L,dxspectrum-1 do begin 
        #print(dxspectrum)
        for i in range(dxspectrum):
            wa=wi*(1.0+1.0/resolution)**i	#wavelength in angstroms corresponding to pixel x
            wamin=wa*(1.0-1.0/(2.0*resolution))	#min wavelength falling in this pixel
            wamax=wa*(1.0+1.0/(2.0*resolution))	#max pixel falling in this pixel
            wavelength[i] = wa

            ## Generate thermal Blackbodies (Tel, AO system, atmosphere)
            # erg s^-1 cm^-2 cm^-1 sr^-1
            bbtel[i] = (2*h*c*c/(wa*1e-8)**5) / (exp(h*c/(wa*1e-8*k*T_tel))-1.0)
            bbaos[i] = (2*h*c*c/(wa*1e-8)**5) / (exp(h*c/(wa*1e-8*k*T_aos))-1.0)
            bbatm[i] = (2*h*c*c/(wa*1e-8)**5) / (exp(h*c/(wa*1e-8*k*T_atm))-1.0)
            bbzod[i] = (2*h*c*c/(wa*1e-8)**5) / (exp(h*c/(wa*1e-8*k*T_zod))-1.0)
            # convert to photons s^-1 cm^-2 A^-1 sr-1 (conversion from
            # table)
            # 5.03e7*lambda photons/erg (where labmda is in angstrom)
            bbtel[i] = (bbtel[i] * 5.03e7 * wa) / 1e8
            bbaos[i] = (bbaos[i] * 5.03e7 * wa) / 1e8
            bbatm[i] = (bbatm[i] * 5.03e7 * wa) / 1e8
            bbzod[i] = (bbzod[i] * 5.03e7 * wa) / 1e8
            # now convert to photon s^-1 m^-2 um^-1 sr-2 (switching to meters and microns
            # for vega conversion down below, yes it cancels above but better to see the step)
            bbtel[i] = bbtel[i] * 1e4 * 1e4
            bbaos[i] = bbaos[i] * 1e4 * 1e4
            bbatm[i] = bbatm[i] * 1e4 * 1e4
            bbzod[i] = bbzod[i] * 1e4 * 1e4
            ## Total BB together with emissivities from each component
            ## photons s^-1 m^-2 um^-1 arcsecond^-2
            #bbspec[i] = sterrad*(bbatm[i]*Em_atm + bbtel[i]*Em_tel + bbaos[i]*Em_aos)
            # only use the BB for the AO system and the telescope since
            # the Gemini observations already includes the atmosphere
            bbspec[i] = sterrad * (bbatm[i] * Em_atm + bbtel[i] * Em_tel + bbaos[i] * Em_aos + bbzod[i] * Em_zod)
            bbspec_gem[i] = sterrad * (bbtel[i] * Em_tel + bbaos[i] * Em_aos)
            #bbspec[i] = bbzod[i]*Em_Zod*sterrad
        if ohsim:
            # use the OH line simulator instead of loading the Gemini file
            gemini_transm = simdir + 'skyspectra/mktrans_zm_' + airmass + '_' + vapor + '.dat'
            gem_transm = np.genfromtxt(gemini_transm)
            transm_wav = gem_transm[:, 0]*1e4
            atm_t = gem_transm[:, 1]
            if wi / 1e4 < np.min(transm_wav):
                transm_wav = np.append(wi / 1e4, transm_wav)
                atm_t = np.append(1.0, atm_t)
            transm_func = interpolate.interp1d(transm_wav, atm_t)
            atm_t = transm_func(wavelength)
            for i in range(0, len(wavelength)):
                bbspec[i] = sterrad * (bbatm[i] * (1. - atm_t[i]) + bbtel[i] * Em_tel + bbaos[i] * Em_aos +
                                      bbzod[i] * Em_zod * (atm_t[i]))
                bbspec_gem[i] = sterrad * (bbatm[i] * (1. - atm_t[i]) + bbzod[i] * atm_t[i] * Em_zod)
            ohfile = simdir + 'skyspectra/optical_ir_sky_lines.dat'
            # load in ESO ohlines
            ohlines = np.genfromtxt(ohfile)
            linecenters = np.array(ohlines[:, 0])*1e4
            linestrengths = np.array(ohlines[:, 1])
            goodCenters = np.where((linecenters >= wi) & (linecenters <= wf) & (linestrengths > 0.))
            linecenters = linecenters[goodCenters]
            if no_oh:
                linestrengths = linestrengths[goodCenters] * 0.
            else:
                linestrengths = linestrengths[goodCenters]
            ohflux = np.zeros((len(wavelength)))
            for i in range(0, len(linecenters)):
                ohflux = ohflux + sim_inst_scatter(wavelength, linecenters[i], strength=linestrengths[i])
            ohspec = ohflux*1000.
            delt = 2.0 * (wavelength[1] - wavelength[0])
            if delt > 1:
                # psf = psf_gaussian(fwhm = delt, npixel = 4*fix(delt)+1, ndimen = 1, /normal)

                stddev = delt / 2 * sqrt(2 * log(2))
                psf_func = models.Gaussian1D(amplitude=1.0, stddev=stddev)
                x = np.arange(4 * int(delt) + 1) - 2 * int(delt)
                psf = psf_func(x)
                psf /= psf.sum()  # normaliza
                #ohspec = convol(ohspec, psf, /edge_truncate)
                ohspec = np.convolve(ohspec, psf, mode='same')
                bbspec = np.convolve(bbspec, psf, mode='same')
                contspec = np.convolve(contspec, psf, mode='same')
        else:
            # read in Gemini data
            geminifile = os.path.expanduser(simdir+'skyspectra/mk_skybg_zm_' + airmass + '_' + vapor + '_ph.dat')
            # load gemini file and convert from  ph/sec/arcsec^2/nm/m^2 to  ph/sec/arcsec^2/micron/m^2
            gemini_dat = np.genfromtxt(geminifile)
            geminiSpec = gemini_dat[:,1]*1e3
            backwave = gemini_dat[:,0]
            geminiWave = backwave*10.0 # nm -> Angstroms
            delt = 2.0*(wavelength[1]-wavelength[0])/(geminiWave[1]-geminiWave[0])
            if delt > 1:
                #psf = psf_gaussian(fwhm = delt, npixel = 4*fix(delt)+1, ndimen = 1, /normal)

                stddev = delt/2*sqrt(2*log(2))
                psf_func = models.Gaussian1D(amplitude=1.0, stddev=stddev)
                x = np.arange(4*int(delt)+1)-2*int(delt)
                psf = psf_func(x)
                psf /= psf.sum() # normaliza

                geminiSpec = np.convolve(geminiSpec, psf, mode='same')
            # interpolate Gemini data to that of OSIRIS
            geminiOrig = geminiSpec
            if np.min(geminiWave) > np.min(wavelength):
                geminiWave = np.append(np.array([np.min(wavelength)]), geminiWave)
                geminiSpec = np.append(np.array([0]), geminiSpec)
            R_i = interpolate.interp1d(geminiWave,geminiSpec)
            R_x = extrap1d(R_i)
            geminiSpec = R_x(wavelength)
            # make sure that we have valid values beyond the edge of the Gemini
            # spectrum
            #bad = where((wavelength <= geminiWave[0]) or (geminiSpec <= 0.0), nbad)
            #if nbad > 0:
            #geminiSpec[bad] = np.median(geminiSpec[max(bad)+1:max(bad)+100]) # replace the uninterpolated points to the median of the next 100 points to get some sort of continuum
            # place gemini spectrum as the OH spectrum
            ohspec = geminiSpec
            bbspec = bbspec_gem
        ### Get information on the filter selection used 
        ### Using zeropoints only from broadband filters (for right now)
        ## normalize the spectra to mag/sq arcsec and get total flux for range of desired filter 
        imin = int(ceil( log10(lambdamin/wi)/log10(1.0+1.0/(resolution)) ))
        #pixel of spectrum corresponding to min
        imax = int(floor( log10(lambdamax/wi)/log10(1.0+1.0/(resolution)) ))
        ## Define background spectra to region of desired filter
        ohspec_filt = ohspec[imin:imax+1]
        contspec_filt = contspec[imin:imax+1]
        bbspec_filt = bbspec[imin:imax+1]
        self.waves = wavelength[imin:imax+1] # (feeding plottiing of background)
        # convolve the OH lines to this resolution
        if not noconvolve:
           delt = 2.0*self.waves[1]/(resolution*(self.waves[1]-self.waves[0]))
           #print, 'delt: ', delt
           #psf = psf_gaussian(fwhm = delt, npixel = 4*fix(delt)+1, ndimen = 1, /normal)
           stddev = delt/2*sqrt(2*log(2))
           psf_func = models.Gaussian1D(amplitude=1.0, stddev=stddev)
           x = np.arange(4*int(delt)+1)-2*int(delt)
           psf = psf_func(x)
           psf /= psf.sum() # normaliza
           #ohspec_filt = convol(ohspec_filt, psf, /edge_truncate)
           ohspec_filt = np.convolve(ohspec_filt, psf, mode='same')
        # tot_oh = total(ohspec_filt)   #total integrated relative photon flux for OH spectrum
        # tot_cont = total(contspec_filt) #same for continuum spectrum
        # tot_bb = total(bbspec_filt)
        # tot_oh = int_tabulated(waves/1e4, ohspec_filt)
        # tot_cont = int_tabulated(waves/1e4, contspec_filt)
        # tot_bb = int_tabulated(waves/1e4, bbspec_filt)
        # ## *total #photons/s/um/sq arcsec collected by TMT for 0 mag/sq arcsec is
        # ## normalize to mag/sq.arcsec=0
        # # ohspectrum = (ohspec_filt/tot_oh) * zp * collarea
        # # contspectrum = (contspec_filt/tot_cont) *  zp * collarea
        # # bbspectrum = (bbspec_filt/tot_bb) *  zp * collarea
        # # scale by zeropoint to get #photons/s/m^2/um/arcec^2 for 0 mag/arcsec^2
        # ohspectrum = (ohspec_filt/tot_oh) * zp
        # contspectrum = (contspec_filt/tot_cont) *  zp
        # if keyword_set(nobbnorm) then bbspectrum = bbspec_filt else bbspectrum = (bbspec_filt/tot_bb) *  zp

        # return the background in the specific filter
        self.backspecs = np.zeros((3, ohspec_filt.shape[0]))
        self.backspecs[0, :] = ohspec_filt
        self.backspecs[1, :] = contspec_filt
        self.backspecs[2, :] = bbspec_filt
        ## return the entire background array, scaled by the zero points for
        ## the givien filter
        #fullbb = (bbspec/tot_bb)*zp
        # don't normalize
        if fullback or fulloh or fullcont or fullbb:
           fullbb = bbspec
           fullcont = contspec
           fulloh = ohspec
           self.fullwaves = wavelength
           self.fullback = np.zeros((3, fulloh.shape[0]))
           self.fullback[0, :] = fulloh
           self.fullback[1, :] = fullcont
           self.fullback[2, :] = fullbb
        ## write results
        #writefits,simdir+'info/tmt_oh_spectrum_m0_'+filter+'.fits',ohspectrum
        #writefits,simdir+'info/tmt_cont_spectrum_m0_'+filter+'.fits',contspectrum
        #writefits,simdir+'info/tmt_bb_spectrum_m0_'+filter+'.fits',bbspectrum
        ## output for osiris_sim.pro

def test_background_specs2(simdir='~/data/osiris/sim/'):

    resolution = 16000
    filter = "Kbb"
    bkgd1 = background_specs3(resolution*2.0, filter, T_atm=273., ohsim=False, simdir=simdir, fullback=True)
    bkgd2 = background_specs3(resolution*2.0, filter, T_atm=273., ohsim=True, simdir=simdir, fullback=True)

    fullwaves   = bkgd2.fullwaves/1e4
    fullback1   = bkgd1.fullback
    fullback2   = bkgd2.fullback
    ohspec1     = fullback1[0,:] + fullback1[2,:]
    ohspec2     = fullback2[0,:] + fullback2[2,:]

    backwave    = bkgd1.waves/1e4
    background1 = bkgd1.backspecs
    background2 = bkgd2.backspecs

    # read in Gemini data
    geminifile = os.path.expanduser(simdir+'skyspectra/mk_skybg_zm_16_15_ph.fits')
    ext = 0 
    pf = fits.open(geminifile)
    # load gemini file and convert from  ph/sec/arcsec^2/nm/m^2 to  ph/sec/arcsec^2/micron/m^2
    geminiSpec = pf[ext].data*1e3
    head = pf[ext].header
    cdelt1 = head["cdelt1"]
    crval1 = head["crval1"]

    nelem = geminiSpec.shape[0]
    #print(nelem)
    geminiWave = (np.arange(nelem))*cdelt1 + crval1  # compute wavelength
    #print(geminiWave.shape)
    geminiWave /= 1e4 # nm
    geminiWave /= 1e3 # nm -> microns

    #geminiWave = backwave*10.0 # go from nm to angstroms

    fig = plt.figure()
    p = fig.add_subplot(111)

    p.plot(fullwaves,ohspec1,c="k",zorder=3,label="Full OH spectrum - Gemini")
    p.plot(fullwaves,ohspec2,c="r",zorder=2,label="Full OH spectrum - simulated")
    #p.plot(geminiWave,geminiSpec,c="b",zorder=1,label="Gemini")
    #p.plot(backwave,np.sum(background1,axis=0)-np.sum(background2,axis=0),c="b")
    #p.set_yscale("log")
    p.set_xlim(np.min(fullwaves), np.max(fullwaves))
    p.set_ylim(0, np.max(ohspec1))
    p.legend()

    # p = fig.add_subplot(212)
    # #p.plot(fullwaves,ohspec1,c="k",zorder=3,label="Full OH spectrum")
    # p.plot(fullwaves,np.sum(fullback1,axis=0),c="k")
    #
    # p.set_yscale("log")
    # p.set_xlim(0.8,0.9)

    plt.show()

def sim_ohlines(wav, simdir = ''):
    ohspec = np.zeros((len(wav)))
    return ohspec
    
def generate_line(wave, linecenters, linestrengths):
    outflux = np.zeros(len(wave))
    for i in range(0, len(linecenters)):
        #fwhm = 2.355*sigma
        line = linecenters[i]
        sigma = 0.00004/2.355
        outflux = outflux + linestrengths[i]*np.exp(-((wave - line)**2.)/(2.*(sigma**2.)))/(sigma * np.sqrt(2.*np.pi))
    return outflux

def sim_inst_scatter(wave, waveCenter, nLines=30000., background=1e-8, strength=None):
    omega = waveCenter / (nLines * np.pi * np.sqrt(2.))
    y = omega ** 2 / ((wave - waveCenter) ** 2.0 + omega ** 2.0) + background

    # normalize so the line has the correct total flux
    norm = strength / (omega * np.pi)

    return norm * y

