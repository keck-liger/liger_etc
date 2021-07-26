import numpy as np
from scipy import interpolate
from astropy.io import fits
import os
import pandas as pd
from math import log10,ceil
from numpy.lib.stride_tricks import as_strided

def get_filterdat(filt, simdir='~/osiris/sensitivity/', mode=''):

    filterfile = os.path.expanduser(simdir + "info/filter_info.dat")
    filterall= np.genfromtxt(filterfile, dtype=str)
    names = ["filterread", "lambdamin", "lambdamax", "lambdac",
             "bw", "backmag", "imagmag", "zp"]
    filterdat = dict(zip(names, filterall.T))
    filts = [i.lower().strip() for i in filterdat['filterread']]
    if mode.lower()=='ifs':
        index = np.where(np.array(filts) == filt.lower().strip())[0][0]
    else:
        index = np.where(np.array(filts) == filt.lower().strip())[0][0]
    if not isinstance(index, np.int64) and mode.lower()=='ifs':
        index = np.where(np.array(filts) == filt.lower().strip())[0][0]
    filterdat = dict(zip(names,[float(val) if val[0].isdigit() else str(val) for val in filterall[index]]))
    return filterdat


def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike

def binnd(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    Number of output dimensions must match number of input dimensions.
    Example
    -------
    # >>> m = np.arange(0,100,1).reshape((10,10))
    # >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    # >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened[::2])

    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray

def frebin2d(array, shape):
    '''Function that performs flux-conservative
    rebinning of an array.
    Inputs:
        array: numpy array to be rebinned
        shape: tuple (x,y) of new array size
        total: Boolean, when True flux is conserved
    Outputs:
        new_array: new rebinned array with dimensions: shape
    '''

    # Determine size of input image
    y, x = array.shape

    y1 = y - 1
    x1 = x - 1

    xbox = x / float(shape[0])
    ybox = y / float(shape[1])

    # Otherwise if not integral contraction
    # First bin in y dimension
    temp = np.zeros((int(shape[1]), x), dtype=np.float64)
    # Loop on output image lines
    #    for i in range(0, int(np.round(shape[1],0)), 1):
    for i in range(0, int(shape[1]), 1):
        rstart = i * ybox
        istart = int(rstart)
        rstop = rstart + ybox
        istop = int(rstop)
        if istop > y1:
            istop = y1
        frac1 = rstart - istart
        frac2 = 1.0 - (rstop - istop)

        # Add pixel values from istart to istop an subtract
        # fracion pixel from istart to rstart and fraction
        # fraction pixel from rstop to istop.
        if istart == istop:
            temp[i, :] = (1.0 - frac1 - frac2) * array[istart, :]
        else:
            temp[i, :] = np.sum(array[istart:istop + 1, :], axis=0) \
                         - frac1 * array[istart, :] \
                         - frac2 * array[istop, :]

    temp = np.transpose(temp)

    # Bin in x dimension
    result = np.zeros((int(shape[0]), int(shape[1])), dtype=np.float64)
    # Loop on output image samples
    #    for i in range(0, int(np.round(shape[0],0)), 1):
    for i in range(0, int(shape[0]), 1):
        rstart = i * xbox
        istart = int(rstart)
        rstop = rstart + xbox
        istop = int(rstop)
        if istop > x1:
            istop = x1
        frac1 = rstart - istart
        frac2 = 1.0 - (rstop - istop)
        # Add pixel values from istart to istop an subtract
        # fracion pixel from istart to rstart and fraction
        # fraction pixel from rstop to istop.
        if istart == istop:
            result[i, :] = (1. - frac1 - frac2) * temp[istart, :]
        else:
            result[i, :] = np.sum(temp[istart:istop + 1, :], axis=0) \
                           - frac1 * temp[istart, :] \
                           - frac2 * temp[istop, :]

    return np.transpose(result) / float(xbox * ybox)

def ab2vega(mag, lmean=2000.):
    ABconv = [["i", 0.7472, 0.37],
              ["z", 0.8917, 0.54],
              ["Y", 1.0305, 0.634],
              ["J", 1.2355, 0.91],
              ["H", 1.6458, 1.39],
              ["Ks", 2.1603, 1.85]]
    ABwave = [i[1] for i in ABconv]
    ABdelta = [i[2] for i in ABconv]
    R_i = interpolate.interp1d(ABwave, ABdelta)
    R_x = extrap1d(R_i)
    delta = R_x([lmean / 1e3])
    mag = mag - delta
    return mag

def vega2ab(mag, lmean=2000.):
    ABconv = [["i", 0.7472, 0.37],
              ["z", 0.8917, 0.54],
              ["Y", 1.0305, 0.634],
              ["J", 1.2355, 0.91],
              ["H", 1.6458, 1.39],
              ["Ks", 2.1603, 1.85]]
    ABwave = [i[1] for i in ABconv]
    ABdelta = [i[2] for i in ABconv]
    R_i = interpolate.interp1d(ABwave, ABdelta)
    R_x = extrap1d(R_i)
    delta = R_x([lmean / 1e3])
    mag = mag + delta
    return mag

def gen_spec(spec1, filt='K', wave=None, scale=None, fint=None, flambda=None, mag=None, source=None,
             simdir='~/osiris/sim', lam_obs=None, line_width=None, mode='ifs', teff='6000', logg='0.0', feh='-0.0',
             aom='0.0', temp=6000, specname=None):
    # constants
    c_km = 2.9979E5  # km/s
    c = 2.9979E10  # cm/s
    h = 6.626068E-27  # cm^2*g/s
    k = 1.3806503E-16  # cm^2*g/(s^2*K)
    Ang = 1E-8  # cm
    mu = 1E-4  # cm
    ##### READ IN FILTER INFORMATION
    filterdat = get_filterdat(filt, simdir, mode=mode)
    modes = pd.read_csv(simdir + 'info/liger_modes.csv', header=0)
    filts = np.array([fil.lower() for fil in modes['Filter']])
    wavs = np.where(filts == filt.lower())[0]
    lmin = modes['λ (nm) min'][wavs]
    lmax = modes['λ (nm) max'][wavs]
    lmin = lmin.values
    lmax = lmax.values
    if not isinstance(lmin, str): lmin = lmin[0]
    if not isinstance(lmax, str): lmax = lmax[0]
    if '*' in str(lmin): lmin = lmin.replace('*', '')
    if '*' in str(lmax): lmax = lmax.replace('*', '')
    lmin = float(lmin)
    lmax = float(lmax)
    lmean = np.mean([lmin, lmax])
    zp = filterdat["zp"]
    if mag is not None:
        # convert to flux density (flambda)
        flux_phot = zp * 10 ** (-0.4 * mag)  # photons/s/m^2
        if source == 'extended':
            flux_phot = flux_phot * (scale ** 2)
    elif flambda is not None:
        fnu = flambda / (Ang / ((lmean * 1e-9) ** 2 / c))
        ABmag = -2.5 * log10(fnu) - 48.60
        mag = ab2vega(ABmag, lmean=lmean)
        flux_phot = zp * 10 ** (-0.4 * mag)  # photons/s/m^2
        if source == 'extended':
            flux_phot = flux_phot * (scale ** 2)
    elif fint is not None:
        # Compute flux_phot from flux
        E_phot = (h * c) / (lmean * Ang)
        flux_phot = 1e4 * fint / E_phot
        if source == 'extended':
            flux_phot = flux_phot * (scale ** 2)
    if spec1.lower() == "flat":
        spec_temp = flux_phot * np.ones(len(wave)) / np.trapz(np.ones(len(wave)), x=wave)
    if spec1 == 'Emission':
        lam_obs = lam_obs * 1e-3
        specwave = wave
        lam_width = lam_obs / c_km * line_width
        instwidth = (lam_obs / 4000.)
        width = np.sqrt(instwidth ** 2 + lam_width ** 2)
        A = flux_phot / (width * np.sqrt(2 * np.pi))  # photons/s/m^2/micron
        spec_temp = A * np.exp(-0.5 * ((specwave - lam_obs) / width) ** 2.)
    elif spec1 == 'Vega':
        ext = 0
        spec_file = os.path.expanduser(simdir + "/model_spectra/" + "vega_all.fits")
        pf = fits.open(spec_file)
        spect = pf[ext].data
        head = pf[ext].header
        cdelt1 = head["cdelt1"]
        crval1 = head["crval1"]
        nelem = spect.shape[0]
        specwave = (np.arange(nelem)) * cdelt1 + crval1  # Angstrom
        spec_func = interpolate.interp1d(specwave / 1e4, spect)
        spec_temp = flux_phot * spec_func(wave) / np.trapz(spec_func(wave), x=wave)
    elif spec1 == 'Phoenix Stellar Library Spectrum':
        if aom == '0.0':
            specfile = 'lte0' + teff + '-' + logg +'0' + feh + '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        else:
            specfile = 'lte0' + teff + '-' + logg +'0' + feh + '.Alpha=' + aom + \
                       '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        specdir = simdir + 'model_spectra/phoenix/' + specfile
        if os.path.isfile(specdir):
            wav = fits.open(simdir + 'model_spectra/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data/10.
            e_phot = (h * c) / (wav * 10. * Ang)
            inspec = fits.open(specdir)[0].data / e_phot
            inspec = inspec[np.argmin(np.abs(wav - lmin)):np.argmin(np.abs(wav - lmax))+1]
            wav = wav[np.argmin(np.abs(wav-lmin)):np.argmin(np.abs(wav-lmax))+1]
            spec_func = interpolate.interp1d(wav / 1e3, inspec)
            spec_temp = flux_phot * spec_func(wave)/np.trapz(spec_func(wave), x=wave)
        else: return ('')
    elif spec1 == 'Black Body':
        spec_temp = (2. * (6.626e-34) * ((3e8) ** 2.)) / (((wave * 1e-6) ** 5.) * (
                    np.exp((((6.626e-34) * (3e8)) / ((wave * 1e-6) * temp * 1.380 * 1e-23))) - 1.))
        spec_temp = flux_phot * spec_temp/np.trapz(spec_temp, x=wave)
    if spec1 == 'Stellar Population Spectra - Maraston & Stromback (2011)':
        spec_temp = specname
    if spec1 == 'Upload':
        spec_temp = specname
    return spec_temp

def sersic_profile(r_eff, s_ind, imsizex, imsizey, zp=None, flux=None, sq_mag=None):
    # assume the galaxy is the center of the image
    x, y = np.indices(([imsizex, imsizey]))
    r = np.sqrt((x - imsizex/2.) ** 2 + (y - imsizey/2.) ** 2)
    r = r.astype(np.float) - np.min(r.astype(np.float))
    b = 1.999 * s_ind - 0.327
    if flux is None:
        fluxNorm = 10.0 ** (-sq_mag / 2.5) / zp
        profile = fluxNorm * np.exp(-b * ((r / r_eff) ** (1.0 / s_ind) - 1.0))
    elif sq_mag is None:
        profile = flux * np.exp(-b * ((r / r_eff) ** (1.0 / s_ind) - 1.0))
    else:
        print('No Flux input')
        profile = np.exp(-b * ((r / r_eff) ** (1.0 / s_ind) - 1.0))
    return profile

def readspec(filename):
    if filename[-4:] == '.dat':
        dat = np.genfromtxt(filename, skip_header=1)
        wvl = dat[:, 0]
        simdat = dat[:, 1]
    elif filename[-4:].lower() == 'fits':
        dat = fits.open(filename)
        header = dat[0].header
        print(header)
        simdat = dat[0].data
        crpix = header['CRPIX1']
        cdelt = header['CDELT1']
        crval = header['CRVAL1']
        N = len(simdat)
        wvl = ((np.arange(N) + 1.0) - crpix) * cdelt + crval
    else:
        print('Please enter either a .dat file or a .fits file')
        wvl = None
        simdat = None
    return wvl, simdat

def eround(n):
    answer = round(n)
    if not answer % 2:
        return answer
    if abs(answer + 1 - n) < abs(answer - 1 - n):
        return answer + 1
    else:
        return answer - 1

