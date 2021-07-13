########################################################################################################################
#
#   etc_analytic_psf.py written by Sanchit Sabhlok originally for OSIRIS ETC under development by OIRLab (UCSD)
#   Date - 01/05/2021
#
#   Adapted from the codes speckle.py and wavefront.py written by Tom Murphy
#   Code taken from his website at https://tmurphy.physics.ucsd.edu/astr597/exercises/speckle.html
#
#   Keck pupil image used to generate PSF provided by Mike Fitzgerald and Pauline Arriaga (UCLA)
#
#   Required packages: numpy, matplotlib, astropy and pandas.
#   Required files: kp_dig.fits, info/filter_info.dat
#
#   Tested on python 3.6.10 while working on the Astroconda environment.
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import os
from astropy.io import ascii
import matplotlib.colors as colors
from astropy.nddata import block_reduce
import math
from etc_analytic_psf.wavefront import *
import streamlit as st

def round_up_to_even(f):
    """
    :param f:
    :return: Rounds number to the next highest even number.
    """
    return math.ceil(f / 2.) * 2


def get_overfill(wl, plate_scale, npix=1800, diameter=12.6, rebin=2.0):
    """
    :param wl: Accepted in Angstroms
    :param plate_scale: Accepted in arcseconds
    :param npix: Number of pixels in the image (Defaults to 1800)
    :return:    Returns the overfill factor required for the given plate scale ensuring the FFT image has
                an even number of pixels.
    """

    ovf = wl * 1e-10 / diameter * 206265 / (plate_scale / rebin)
    ovf = (float(round_up_to_even(float(ovf * npix))) + 0.001) / float(npix)
    if (ovf < 1.0): # Make sure the overfill factor is large enough, lest it break code
        rebin *= 2.0
        ovf = get_overfill(wl,plate_scale,rebin=rebin)
    if isinstance(ovf, tuple):
        ovf = ovf[0]

    return ovf,rebin

def crop_center(img, cropx, cropy):
    """
    :param img: Input image
    :param cropx: X side length (in pixels) for the final image
    :param cropy: Y side length (in pixels) for the final image
    :return: Image cropped while still centered on the original central pixel by trimming the outer edges.
    """
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

@st.cache(hash_funcs={complex: hash}, show_spinner=False)
def analytic_psf(strehl, lvals, plate_scale, fried_parameter=20, verbose=False, stack=False, simdir='~/'):
    """

    :param strehl:          Required.
    :param lvals:           List of wavelengths (in Angstroms) with lowest wavelength the first element
    :param plate_scale:     Required plate scale for the final image.
    :param fried_parameter: User input (in cm). Defaults to 20 cm.
    :param verbose:         Prints additional diagnostics. Defaults to False.
    :param stack:           If set to True, returns a stack of PSFs for all lambda values instead of averaged version.
                            Defaults to False.
    :return:        If stack is set to True, returns a stack of PSFs for the filter using Lambda min, max and central.
                    Default behavior is to return a single PSF image for the band averaging the three PSFs.
                    PSF is normalised to a flux of unity.

    """

    diameter = 12.6  # Diameter of Telescope (in m.)
    if verbose:
        print("Verbose set to True.")

    pupil_file = simdir + 'info/kp_dig.fits'  # Retrieve the pupil image.

    #filename = get_pkg_data_filename(pupil_file)  # Obtain the filename from the file
    hdu = fits.open(pupil_file)[0]  # Extract Header information
    keck_pupil = hdu.data  # Extract the data

    pwf = plane_wave(npix=np.size(keck_pupil, axis=0))  # Generate a plane wavefront

    if verbose:
        print("Wavelength values: " + str(lvals))

    final_im = []

    #   Iterate over the wavelength values
    if not isinstance(lvals,tuple) and not isinstance(lvals, list): lvals = [lvals]
    for wl in lvals:
        if verbose:
            print("At wavelength " + str(wl) + " Angstroms")

        overfill_factor, rebin = get_overfill(wl, plate_scale, diameter=diameter)  # Calculate overfill factor

        if verbose:
            print("Overfill Factor: " + str(overfill_factor))
            print("Padded side length for FFT: " + str(int(overfill_factor * 1800)))

        diffrac = psf(keck_pupil, pwf, overfill=overfill_factor)  # Generate Diffraction Limited PSF
        diffrac = diffrac / np.sum(diffrac)  # Normalise flux to unity

        wl = wl * 1e-10  # Wavelength is now in meters.

        pixel_scale = plate_scale / rebin  # Raw image generated at twice the resolution before binning

        n_pix = 900  # Half Size of image (in pixels)

        #   Generate a gaussian seeing disk corresponding to the fried parameter.
        x, y = np.meshgrid(np.linspace(- n_pix + 1, n_pix, 2 * n_pix), np.linspace(- n_pix + 1, n_pix, 2 * n_pix))
        d = np.sqrt(x * x + y * y) * pixel_scale

        seeing_fwhm = wl / fried_parameter * 206265 / 1e-2  # Determine FWHM given the Fried Parameter and Wavelength.

        g_seeing = np.exp(
            -(d ** 2 / (2.0 * seeing_fwhm ** 2))) / np.pi / 2.0 / seeing_fwhm  # g_seeing is the seeing halo disk
        g_flux = np.sum(g_seeing)
        g_seeing = g_seeing / g_flux  # Normalise flux to unity

        if verbose:
            print("Flux for Seeing Disk: " + str(np.sum(g_seeing)))

        op_g = (1.0 - strehl) * g_seeing + strehl * diffrac  # Generate the PSF given the Strehl value
        op_g = op_g / np.sum(op_g)  # Normalise flux to unity

        if verbose:
            print("Original Sum: " + str(np.sum(op_g)))
        resized_im = block_reduce(op_g, int(rebin))  # Bin the image to the required plate scale.
        final_im.append(resized_im)
        if verbose:
            print("Resized image Sum: " + str(np.sum(resized_im)))

    #   Crop the images to the size of the smallest image, which will be the lowest wavelength.
    #   Cropping is done centered on the central pixel so effectively excess outer edges are trimmed in the procedure.
    if verbose:
        lcounter = 0
        for ii in lvals:
            lcounter += 1
            print("Image "+ str(lcounter) +" Sum Uncropped: " + str(np.sum(final_im[lcounter-1])))

    final_im = np.asarray(final_im)
    imlen = np.size(final_im[0], axis=0)
    imlen = int(imlen)

    cropped_im = np.ndarray(shape=[len(lvals),imlen,imlen])

    lcounter = 0
    for ii in lvals:
        cropped_im[lcounter] = np.asarray(crop_center(final_im[lcounter], imlen, imlen))
        lcounter += 1

    lcounter = 0
    if verbose:
        print("Image "+ str(lcounter+1) +" Sum: " + str(np.sum(cropped_im[lcounter])))
        lcounter += 1

    lcounter = 0
    imfinal =  np.mean(cropped_im,axis=0) # Generate final image by averaging the waveband images.

    if verbose:
        print("Final image Sum: " + str(np.sum(imfinal)))

    if stack:
        return cropped_im  # Return stack
    else:
        return imfinal  # Return single image


# #   Sample code to demonstrate calling the function
#
# fig1 = plt.figure()  # Generate a blank figure
# ax1 = fig1.add_subplot(111)  # Create axis object
# im1 = analytic_psf(0.8, [10875.0], 0.05, 20, verbose=True, stack=True, simdir='~/Documents/OIRLab/OSIRIS/osiris_etc/')  # Generate PSF and store in variable im1
#
# ax1.imshow(im1[0], norm=colors.LogNorm(vmin=im1[0].max() / 1e5, vmax=im1[0].max()))  # Plot the image
#
# plt.show()
