import streamlit as st
import numpy as np

def helppage(page_container=None):
    if page_container is None:
        pc = st
    else:
        pc = page_container

    pc.markdown('''
    This page is meant to help the user navigate and use the OSIRIS Exposure Time Calculator (ETC).  It will provide
    background and information regarding OSIRIS and the methods used to calculate the OSIRIS Signal-to-Noise Ratio (SNR)
    and exposure time results, explain the different parts and functions of the main ETC Graphical User Interface (GUI).
    If you want to submit a bug, please click the "Submit a Bug" button in the Page Naviation section at the top of 
    the page. If you would like to contact someone regarding the ETC, please email oirlab@physics.ucsd.edu with your 
    inquiries.
    ''')
    #  and provide a few example ETC cases

    pc.header('Introduction')
    pc.markdown('''
    OH-Suppressing Infra-Red Imaging Spectrograph (OSIRIS) is an Integral Field Spectrograph (IFS) and Imager used in 
    in conjunction with the Keck Adaptive Optics system by Keck Observatory astronomers. The function of this online 
    calculator is to help astronomers assess the quality of OSIRIS data achievable for their individual distinct science 
    cases.
    More information about the Keck Observatory and the OSIRIS instrument can be found on the [Keck Observatory 
    website](https://www2.keck.hawaii.edu/inst/osiris/) and the 
    [UCLA IRLab website](https://irlab.astro.ucla.edu/instruments/osiris/). 
    ''')

    pc.header('SNR Calculation')
    pc.markdown('''
    OSIRIS is simulated by assessing the per-pixel signal and noise using a [background model from the Gemini 
    Observatory](https://www.gemini.edu/observing/telescopes-and-sites/sites#Transmission) (Lord 1992) and estimates of 
    OSIRIS noise performance to calculate the SNR per pixel element according to the equation \n
    $SNR = \\frac{S * \sqrt{T}}{\sqrt{(S + B_{N} + R^{2}/t + D_{N})}}$ \t \t (1) \n 
    where S is the Flux per pixel, T is the total integration time, $B_N$ is the background contribution per pixel, 
    R is the read noise contribution, t is the integration time per frame, and D is the dark current contribution per 
    pixel. The equation for integration time required to achieve an input SNR can be found by inverting the equation.
    The Gemini Background model assumes a water vapor column of 1.6mm and an air mass of 1.5 on Mauna Kea.
    Background from moonlight is not included.
    ''')

    backg_dropdown = pc.expander('Plot Gemini Background Spectrum')
    if backg_dropdown:
        import configparser
        import os
        from astropy.io import fits
        import plotly.graph_objects as go
        config = configparser.ConfigParser()
        config.read('config.ini')
        simdir = config.get('CONFIG', 'simdir')
        geminifile = os.path.expanduser(simdir + 'skyspectra/mk_skybg_zm_16_15_ph.fits')
        ext = 0
        pf = fits.open(geminifile)
        # load gemini file and convert from  ph/sec/arcsec^2/nm/m^2 to  ph/sec/arcsec^2/micron/m^2
        geminiSpec = pf[ext].data * 1e3
        head = pf[ext].header
        cdelt1 = head["cdelt1"]
        crval1 = head["crval1"]
        nelem = geminiSpec.shape[0]
        backwave = (np.arange(nelem)) * cdelt1 + crval1  # compute wavelength
        backwave /= 1e7  # nm
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=backwave, y=geminiSpec))
        fig.update_layout(xaxis_title="Wavelength (microns)", yaxis_title="photons/second/arcsec^2/micron/m^2",
                          xaxis_range=[0.9, 2.4], yaxis_range=[
                np.min(geminiSpec[int(np.argmin(np.abs(backwave-0.9))):int(np.argmin(np.abs(backwave-2.4)))]),
                np.max(geminiSpec[int(np.argmin(np.abs(backwave-0.9))):int(np.argmin(np.abs(backwave-2.4)))])])
        backg_dropdown.plotly_chart(fig)

    pc.markdown('''
    The background spectrum is combined with a blackbody spectrum generated from the estimated temperature and 
    emissivity of the telescope and adaptive optics system. 
    Dark current is estimated as 0.05 $e^{-}$/second and read noise is estimated at 7 $e^{-}$. Using these estimates for
    the noise values, the background within the selected bandpass and signal values are then scaled according to the 
    Keck telescope collecting area (78.5 $m^{2}$) and the estimated throughput values for the mode and bandpass 
    selected. Instrument throughputs are estimated to be relatively consistent at ~15% towards the blue end for the IFS, 
    with an increase to 20% in K.  For the imager we use values of 22% and 29% for these wavelengths as well, 
    respectively. These instrument throughputs are then adjusted with the telescope and adaptive optics throughputs 
    (assumed to be 65% and 50% respectively) to yield the total throughputs estimated in the plot below.
    ''')

    eff_dropdown = pc.expander('Plot Throughput Values over Wavelength')
    if eff_dropdown:
        wi = 9000  # Ang
        wf = 24000  # Ang
        from math import log10, ceil
        from scipy import interpolate
        resolution=3800
        dxspectrum = int(ceil(log10(wf / wi) / log10(1.0 + 1.0 / (resolution * 2.0))))
        ## resolution times 2 to get nyquist sampled
        crval1 = wi / 10.  # nm
        cdelt1 = ((wf - wi) / dxspectrum) / 10.  # nm/channel
        # Throughput calculation
        teltot = 0.65
        aotot = 0.80
        wav = [830, 900, 2000, 2200, 2300, 2412]  # nm
        imagtput = [0.22, 0.22, 0.29, 0.29, 0.29, 0.29]  # imager
        ifstput = [0.15, 0.15, 0.2, 0.2, 0.2, 0.2]  # IFS lenslet
        w = (np.arange(dxspectrum) + 1) * cdelt1 + crval1  # compute wavelength
        #####################################################################
        # Interpolating the OSIRIS throughputs from the PDR-1 Design Description
        # Document (Table 7, page 54)
        #####################################################################
        imagR = interpolate.interp1d(wav, imagtput, fill_value='extrapolate')
        eff_imag_lambda = np.array([imagR(w0) for w0 in w]) * teltot * aotot
        ifsR = interpolate.interp1d(wav, ifstput, fill_value='extrapolate')
        eff_ifs_lambda = np.array([ifsR(w0) for w0 in w]) * teltot * aotot
        ###############################################################
        # MEAN OF THE INSTRUMENT THROUGHPUT BETWEEN THE FILTER BANDPASS
        ###############################################################
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w/1e3, y=eff_ifs_lambda*100., name='IFS Throughput Value'))
        fig.add_trace(go.Scatter(x=w/1e3, y=eff_imag_lambda*100., name='Imager Throughput Value'))
        fig.update_layout(xaxis_title="Wavelength (microns)", yaxis_title="Throughput (%)")
        eff_dropdown.plotly_chart(fig)

    pc.markdown('''
    The SNR is then found by convolving a given input flux (after convolving the spectrum to the correct resolution for 
    IFS mode) with the Point Spread Function, adjusting throughputs, and assessing the SNR according to equation 1. 
    ''')

    pc.header('Using the ETC')
    pc.markdown('''
    The ETC operates in either IFS or Imager mode, and calculates either the SNR for an input integration time or the 
    required integration time for an input SNR.  Because both SNR and integration time are flux-dependent, both of these
    results change over wavelength for the input spectrum when the ETC is used in the IFS mode. 
    If the ETC is used in Imager mode, these values are simply returned for the integrated bandpass flux values as they 
    are distributed over the imager spatial pixels. The work flow for using the OSIRIS ETC should follow this structure:
    
    1. Select OSIRIS Mode (IFS or Imager)
    
    2. Select Desired Calculation (SNR or Exposure Time)
    
    3. Edit Source Properties (Spectral Shape if using IFS mode, and Flux: Magnitude, Spectral Density, Integrated Flux).
    
    4. Select desired bandpass for calculation.  The ETC has the capability to show the available filters for a selected
    field of view or plate scale, and vice versa. 
    
    5. Select or configure desired Point-Spread Function (PSF) for simulation.  The ETC has the capability to either use
    pre-simulated SCAO PSFs for the on-axis location of the Telescope assuming K-band, or for the user to analytically
    generate the PSF given a certain Strehl Ratio and Fried Parameter.
    
    6. Click the 'CLICK HERE to Calculate OSIRIS Results' button to run the OSIRIS Exposure Time Calculator code and 
    view the results! For IFS mode you can optionally download a CSV file of the resulting spectra.
    
    
    ''')

    pc.subheader('Definitions')
    pc.markdown('''
    This section will provide short definitions for the various keywords used in the ETC.
    
    **Mode**: IFS or Imager.
    
    **Filter**: This field selects the filter bandpass through which the signal-to-noise ratio or exposure time will be 
    calculated. 
    
    **Plate Scale**: The OSIRIS Imager operates at a 20mas/pixel plate scale, with plate scales of 20mas, 35mas, 50mas, 
    and 100mas available to the IFS depending on the field of view requirements of the user. 
    
    **Field** of View: The OSIRIS Imager provides a 20” x 20” field of view, and the IFS ranges in field of view from 
    0.32" x 1.28" to 4.8" x 6.4". 
    
    **Signal-To-Noise Ratio (SNR)**: This is as defined by equation 1. Enter the desired peak signal to noise ratio of 
    the observation if calculating the required exposure time.
    
    **Exposure Time**: The exposure time in seconds for each individual frame to be used in the simulation. Enter the 
    exposure time of the observation if calculating the resulting signal-to-noise ratio.
    
    **Number of Frames**: Enter the number of frames taken in the observation of the provided exposure time if 
    calculating the resulting signal-to-noise ratio.
    
    **Total Integration Time**: This returns the total integration time needed for an observation with the given 
    signal-to-noise ratio or the provided exposure time and number of frames.
    
    **Flux Magnitude**: The magnitude (either AB or Vega standards) of the source, resumed to be per square arcsecond 
    if "Extended" source type is selected. 
    
    **Flux Density**: Flux in ergs* cm$^{-2}$ s$^{-1}$ $\r{A}^{-1}$
    
    **Integrated Flux**: Flux in ergs* cm$^{-2}$ s$^{-1}$ (integrated over the filter bandpass).
    
    **Point Source**: This option indicates that the simulated source should be point-source like in nature, with flux 
    distributed according to the Point Spread Function associated with that spatial position and wavelength range.
    
    **Extended** Source: This option indicates that the flux should be distributed over the observed spatial field, 
    according to some profile and not point-source like. The current assumed profile for the OSIRIS ETC is top-hat.
    We also have optional Sersic profiles, which can be generated for a given Sersic index and effective radius in
    arcseconds.
    WARNING: The Sersic Profile option significantly increases processing time as it requires convolution of the OSIRIS
    PSF with the generated profile.
    
    **Spectrum**: Enter the desired spectrum shape for the simulated observation. We have options from a Vega, Flat, 
    emission line, Phoenix Stellar Library, Black Body, Simple Stellar Population Models of Maraston & Stromback (2011), 
    Custom (meaning combinations of any of these), and the option to upload your own spectral files. If you choose 
    upload, you must use a text file with the second to last column in microns and the last column with the corresponding
    flux.  The minimum resolution ($$\\frac{\lambda}{d \lambda}$$) of the input spectra must be greater than 8000. The 
    units of the input flux is assumed $$photons / second / \mu m/ m^{2}$$ but will be normalized to the input flux 
    provided in the ETC GUI. 
    
    **Redshift**: Redshift is available as an optional input for the Simle Stellar Population Models and for uploaded 
    spectra. This redshift is given by the equation $$z = \\frac{\lambda_{obs} - \lambda_{rest}}{\lambda_{rest}}$$. 
    The input redshift parameter will therefore shift the input wavelength by $$\lambda = \lambda_{input} * (1 + z)$$.
    
    **Point Spread Function (PSF)**: The Point Spread Function defines the flux distribution of a point source through 
    the optical system according to atmosphere, telescope, adaptive optics, and instrument effects. We recommend using
    the PSF generator to generate a PSF given the likely achieved Strehl ratio and atmospheric conditions (and therefore 
    Fried Parameter) for your science case.
    
    **Strehl Ratio**: This is the Strehl ratio for a generated PSF.  The Strehl ratio is defined as the ratio of the 
    Peak PSF value with that of a perfect PSF limited only by diffraction.  In this calculator it effectively defines
    the encircled energy within the core of PSF as compared to the energy in the outer PSF. 
    
    **Fried Parameter**: r0 is used in the PSF generation code to define a halo seeing-disk which is added to the PSF. 
    This does not affect the Strehl Ratio in the PSF generation code. 
    
    **Aperture Radius**: The signal-to-noise (and therefore integration time as well) is a function of the flux, and 
    therefore is assessed over a given number of detector pixels.  The number of pixels used for assessing the SNR or 
    integration time is defined by the aperture radius.  The aperture radius is selected in arcseconds, but can be 
    converted to pixels as pixel radius = arcsecond radius / plate scale. This radius defines the circular aperture
    over which the flux of the source is estimated. 
    
    ''')

    pc.subheader('Configuring Your Source - Custom Spectra')
    pc.markdown('''
    
    The OSIRIS ETC has a variety of spectral templates built in to help the user.  These are Vega, Flat, Emission Lines,
    Black Body, [Gottingen Phoenix Stellar Library Spectra](https://phoenix.astro.physik.uni-goettingen.de/?page_id=15), 
    and the [Simple Stellar Population Models of Maraston & Stromback (2011)](http://www.icg.port.ac.uk/~maraston/M11/)
     [(Paper Here)](https://arxiv.org/pdf/1109.0543.pdf). Because of the high-resolution requirement of OSIRIS some of 
     these spectra were not included. 
    
    The OSIRIS ETC has a built-in spectrum generator to help you create your own science source and simulate its 
    viability using OSIRIS.  This can be done by going to the "Spectrum Shape" dropdown list in the "Source Properties"
    section and selecting "Custom".
    If your desired simulation is only a single flat, Vega, emission line, black body, Phoenix Stellar Library,
    or a Simple Stellar Population spectrum, then this is not necessary and you should select one of those from the list. 
    The OSIRIS ETC spectrum generator allows you to combine any or all of the listed spectra types to be passed to the 
    ETC, each normalized to its own flux value. 
    
    When you select "Custom" from the "Source Properties" section, a new section will appear in the main body of the ETC
    GUI labeled "Input Spectrum Generator". This section will have another dropdown list labeled "Add Spectrum" from 
    which you can select any of this list.  When you select one from the list, another optional list will appear from 
    which you can select, which will be added to the previous one you selected.  Each spectrum must be normalized to an
    input flux, the input boxes for which will appear to the right of the spectra selection drop list.  
    
    As with the "Source Properties" section, you will have the option of choosing what type of input flux you would like
    the custom spectrum normalized to.  The input flux boxes in the "Source Properties" section in the sidebar have no
    functionality if you choose to create a custom spectrum instead.
    
    ''')

    pc.image('CustomSpecETC.png',
             caption='Spectral shape in the ETC may be altered in two ways, either by selecting a single spectral shape '
                     'in the side bar (highlighted in the lower left red box) or by configuring a custom spectral shape '
                     'in the custom spectrum generator. If you select "Custom" in the sidebar panel, a new section in '
                     'the main window will open to allow for spectra customization (highlighted in larger red box). '
                     'This is where different spectra may be added together.',
             width=1000)
    pc.image('CustomSpecETC2.png', caption='After selecting a spectral shape, the flux for that spectrum must be set in'
                                           'the same row (highlighted in upper red box).  Subsequent rows are added on '
                                           'as they are needed in order to customize the particular spectrum selected. '
                                           'The two larger red boxes highlight the inputs concerning their respective '
                                           'selected sectral shape.', width=1000)

    pc.header('PSF Generator')
    pc.markdown('''
    The PSF generator is a relatively simple piece of the ETC. We first generate a diffraction limited Point Spread 
    Function for the Keck Pupil based on the pupil images provided by Mike Fitzgerald and Pauline Arriaga at UCLA. 
    These pupil images are zero padded appropriately for a given wavelength and then we take the Fast Fourier Transform 
    to generate the initial diffraction limited PSF at a spatial resolution which is twice the requested final plate 
    scale of the image. Then, we use the Fried parameter to determine the Full width half maximum of the Gaussian seeing 
    disk generated. Both these images are flux normalised to unity. Then we take the input strehl value and adjust the 
    flux of the diffraction limited PSF with respect to the seeing disk and add the two together. If a monochromatic PSF 
    is requested, this summed image is binned forward to the required plate scale and returned. If a waveband averaged 
    PSF is requested, the PSFs generated at the minimum, central and maximum wavelengths are generated by the same 
    procedure and averaged to return the final PSF.
    ''')

    pc.subheader('Inputs:')
    pc.markdown('''
    **Strehl**: Determines the flux ratio of the diffraction limited PSF and the generated Gaussian seeing disk. 
    If testing the SNR capabilities of OSIRIS with the upcoming Keck All-sky Precision Adaptive-optics (KAPA) project, 
    we suggest calculating an appropriate strehl for the user's particular AO setup using the 
    [KAPA Strehl Calculator](http://bhs.astro.berkeley.edu/cgi-bin/kapa_strehl).
    
    **Fried parameter**: Determines the full width half maximum of the gaussian seeing disk as $$\lambda / r_0$$. 
    Reasonable values typically range from 10cm for average atmospheric conditions to 20cm for excellent seeing 
    conditions. 
    Entering unreasonable values for $$r_0$$ may result in un-physical PSFs for our approximation of the Strehl Ratio. 
    ''')

    pc.header('Understanding Results')
    pc.markdown('The results section of the OSIRIS ETC is shown below.')

    pc.image('ETC_res.png', width=1000, caption='The results section of the ETC that appears after each run is '
                                                'highlighted in the red box.')

    pc.markdown('''
    The Signal-to-noise values shown on the left and right sides of the screen are for the integrated flux values across
    the spectrum, and the per-spectral-element values respectively.  The integrated values are integrated across 
    wavelength using the selected aperture.
    
    If exposure time is being calculated, this same method is applied.  Because the exposure time is calculated over 
    spectral elements, one should adopt the peak integration time for the given input SNR across the spectrum as the 
    required time to achieve an observation with the input SNR as the maximum SNR achieved. 
    
    For imager results, a simulated image is shown and flux values are summed spatially. 
    
    The assumed saturation limit of the detector is 90000 electrons, with an assumed gain of 1. 
    ''')

    # pc.header('Example Cases')
    # pc.markdown('''
    # Case of an Imager point source SNR calculation:
    # ''')
    #
    # pc.markdown('''
    #     Case of an IFS extended source SNR calculation with emission line:
    #     ''')