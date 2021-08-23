import pysynphot
import matplotlib.pyplot as plt
import urllib.request
import wget
import bs4
import requests
import os
import shutil
from math import log10,ceil,sqrt,log
from get_filterdat import get_filterdat
from scipy import interpolate
from misc_funcs import extrap1d
import streamlit as st

# constants
c_km = 2.9979E5      # km/s
c = 2.9979E10       # cm/s
h = 6.626068E-27    # cm^2*g/s
k = 1.3806503E-16   # cm^2*g/(s^2*K)
Ang = 1E-8          # cm
mu = 1E-4           # cm
page_cont = st.container()


col1,col2 = page_cont.columns(2)
page_container = st.container()
with page_container.container():
    num1 = col1.number_input('First Number:',min_value=0.1, value=1.5, max_value=5.)
    num2 = col2.number_input('Second Number:', min_value = 1000., value = 2000., max_value=5000.)

    k=0
    if num1 < 4:
        num3 = col1.number_input('First Number:', min_value=0.1, value=1.5, max_value=5., key='num3'+str(k))
        num4 = col2.number_input('Second Number:', min_value=1000., value=2000., max_value=5000., key='num4'+str(k))
        col1.write('Third Number is:' + str(num3))

        st.markdown('Fourth Number is :' + str(num4))

    col1.write('First Number is:'+str(num1))

    st.markdown('Second Number is :'+str(num2))

import streamlit as st
import numpy as np
import pandas as pd
from liger_snr_sim import OSIRIS_ETC
from etc_analytic_psf.etc_analytic_psf import analytic_psf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.graph_objects as go
import base64
import configparser
import glob as glob
import os as os
from astropy.io import fits
from math import log10, ceil, sqrt, log
from get_filterdat import get_filterdat
from scipy import interpolate
from misc_funcs import extrap1d, ab2vega, gen_spec

# constants
c_km = 2.9979E5  # km/s
c = 2.9979E10  # cm/s
h = 6.626068E-27  # cm^2*g/s
k = 1.3806503E-16  # cm^2*g/(s^2*K)
Ang = 1E-8  # cm
mu = 1E-4  # cm


def exec_gui(page_container=None, side_container=None):
    if page_container is None:
        page_container = st
    if side_container is None:
        side_container = st.sidebar

    def plot_fov(fov):
        """
        :param fov: field of view as a string in "X x Y" where X and Y are numbers
        :return:    matplotlib figure for use by streamlit
        """
        fig, ax = plt.subplots()
        ffov = [float(x) for x in fov.split('x')]
        # ax.scatter([-ffov[0]/2., 0, ffov[0]/2.], [-ffov[1]/2., 0, ffov[1]/2.], alpha=1.)
        ax.set_ylim(-np.max(ffov) / 2., np.max(ffov) / 2.)
        ax.set_xlim(-np.max(ffov) / 2., np.max(ffov) / 2.)
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        ax.set_xticks(xticks[::2])
        ax.set_yticks(yticks[::2])
        ax.set_ylabel('Arcseconds')
        ax.set_xlabel('Arcseconds')
        ax.set_title('Field of View Dimensions')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(b=True, which='major', alpha=1.0, color='gray', linestyle='-', linewidth=1)
        rect = Rectangle((-ffov[0] / 2., -ffov[1] / 2.), ffov[0], ffov[1], linestyle='dashdot', linewidth=1.5, fill=0)
        ax.add_patch(rect)
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas = FigureCanvas(fig)
        canvas.draw()  # draw the canvas, cache the renderer
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return (img)

    def plot_filter(filterfile):
        """
        :param filterfile: directory of filter file to plot
        :return:    matplotlib figure for use by streamlit
        """
        dat = np.genfromtxt(filterfile, skip_header=2, usecols=(0, 1))
        wvl = dat[:, 0]
        transm = dat[:, 1]
        filterfig = go.Figure()
        filterfig.add_trace(go.Scatter(x=wvl, y=transm))
        filterfig.update_layout(yaxis_title='Transmission (%)', xaxis_title='Wavelength (nm)', title='Filter Curve')
        return (filterfig)

    def download_link(object_to_download, download_filename, download_link_text):
        """
        Generates a link to download the given object_to_download.

        object_to_download (str, pd.DataFrame):  The object to be downloaded.
        download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
        download_link_text (str): Text to display for download link.

        Examples:
        download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
        download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

        """
        if isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

        return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    #  define PSF and simulations directories for filter information, Pupil FITS for PSF
    #  and PSF directory if feeding in PSF instead of generating one
    config = configparser.ConfigParser()
    config.read('config.ini')
    simdir = config.get('CONFIG', 'simdir')
    # simdir = config['CONFIG']['simdir']
    psfdir = config.get('CONFIG', 'psfdir')

    #   Page title, set 2 columns for page format. Setup the configuration sidebar.
    container = page_container.container()
    container.title('OSIRIS Exposure Time Calculator')
    col1, col2 = container.columns(2)
    col1.subheader('OSIRIS Configuration')

    #   select the mode for the ETC -- Imager or IFS
    side_container.title('Exposure / Signal-To-Noise Setup')
    mode = side_container.radio("Please Select OSIRIS Mode", ['IFS', 'Imager'])
    modes = pd.read_csv(simdir + 'info/osiris_modes.csv', header=0, dtype=str)
    if mode == 'Imager':
        scale = '20mas'
        col1.markdown('Plate Scale ( Miliarcseconds per Spatial Pixel ): \n 20mas')
        col1.markdown('Field of View (Arcseconds x Arcseconds): \n 20.4 x 20.4')
        fov = '20.4 x 20.4'
        filt = col1.selectbox("Filter: ",
                              ['Zbb', 'Jbb', 'Hbb', 'Kbb', 'Kcb', 'Zn3', 'Zn4', 'Jn1', 'Jn2', 'Jn3', 'Jn4', 'Hn1',
                               'Hn2',
                               'Hn3', 'Hn4', 'Hn5', 'Kn1', 'Kn2', 'Kn3', 'Kc3', 'Kn4', 'Kc4', 'Kn5', 'Kc5',
                               'FeII', 'Hcont', 'Y', 'J', 'Kp', 'BrGamma', 'Kcont', 'HeI_B', 'Kcb'])
    elif mode == 'IFS':
        #   User can select whether to configure with the Filter and Plate scale or the Field of View
        subconf = side_container.radio("Configure With: ", ['Filter / Plate Scale', 'Field of View'])
        if subconf == 'Filter / Plate Scale':
            filts = modes['Filter'].values
            scales = modes.columns[-4:].values
            filt = col1.selectbox("Filter: ", [f for f in filts.flatten()])
            # col1.write([s for s in scales.flatten()])
            if filt in ['Kcb', 'Kc3', 'Kc4', 'Kc5']:
                scale = '100mas'
                col1.markdown('Plate Scale ( Arcseconds per Spatial Pixel ): 100mas')
            else:
                scale = col1.select_slider('Plate Scale ( Arcseconds per Spatial Pixel )', [s for s in scales])
            fovs = modes[scale].iloc[(np.where(modes['Filter'] == filt)[0])].values
            if len(fovs) > 1:
                fov = col1.select_slider('Field of View:', [f for f in fovs.flatten()])
            else:
                fov = fovs[0]
        else:
            fov = col1.select_slider('Field of View:',
                                     ['0.32 x 1.28', '0.64 x 1.28', '0.72 x 1.28', '0.84 x 1.28', '0.90 x 1.28',
                                      '0.96 x 1.28',
                                      '0.56 x 2.24', '1.12 x 2.24', '1.26 x 2.24', '1.47 x 2.24', '1.58 x 2.24',
                                      '1.68 x 2.24',
                                      '0.8 x 3.2', '1.6 x 3.2', '1.8 x 3.2', '2.1 x 3.2', '2.25 x 3.2', '2.4 x 3.2',
                                      '1.6 x 6.4', '3.2 x 6.4', '3.6 x 6.4', '4.2 x 6.4', '4.5 x 6.4', '4.8 x 6.4'])
            if fov == '0.32 x 1.28':
                filt = col1.selectbox('Filter: ', ['Zbb', 'Jbb', 'Hbb', 'Kbb'])
                scale = '20mas'
            if fov == '0.64 x 1.28':
                filt = col1.selectbox('Filter: ', ['Zn4', 'Jn1', 'Hn5', 'Kn5'])
                scale = '20mas'
            if fov == '0.72 x 1.28':
                filt = col1.selectbox('Filter: ', ['Hn1', 'Kn1'])
                scale = '20mas'
            if fov == '0.84 x 1.28':
                filt = col1.selectbox('Filter: ', ['Jn2', 'Jn4', 'Hn4', 'Kn4'])
                scale = '20mas'
            if fov == '0.90 x 1.28':
                filt = col1.selectbox('Filter: ', ['Hn2', 'Kn2'])
                scale = '20mas'
            if fov == '0.96 x 1.28':
                filt = col1.selectbox('Filter: ', ['Jn3', 'Hn3', 'Kn3'])
                scale = '20mas'
            if fov == '0.56 x 2.24':
                filt = col1.selectbox('Filter: ', ['Zbb', 'Jbb', 'Hbb', 'Kbb'])
                scale = '35mas'
            if fov == '1.12 x 2.24':
                filt = col1.selectbox('Filter: ', ['Zn4', 'Jn1', 'Hn5', 'Kn5'])
                scale = '35mas'
            if fov == '1.26 x 2.24':
                filt = col1.selectbox('Filter: ', ['Hn1', 'Kn1'])
                scale = '35mas'
            if fov == '1.47 x 2.24':
                filt = col1.selectbox('Filter: ', ['Jn2', 'Jn4', 'Hn4', 'Kn4'])
                scale = '35mas'
            if fov == '1.58 x 2.24':
                filt = col1.selectbox('Filter: ', ['Hn2', 'Kn2'])
                scale = '35mas'
            if fov == '1.68 x 2.24':
                filt = col1.selectbox('Filter: ', ['Jn3', 'Hn3', 'Kn3'])
                scale = '35mas'
            if fov == '0.8 x 3.2':
                filt = col1.selectbox('Filter: ', ['Zbb', 'Jbb', 'Hbb', 'Kbb'])
                scale = '50mas'
            if fov == '1.6 x 3.2':
                filt = col1.selectbox('Filter: ', ['Zn4', 'Jn1', 'Hn5', 'Kn5'])
                scale = '50mas'
            if fov == '1.8 x 3.2':
                filt = col1.selectbox('Filter: ', ['Hn1', 'Kn1'])
                scale = '50mas'
            if fov == '2.1 x 3.2':
                filt = col1.selectbox('Filter: ', ['Jn2', 'Jn4', 'Hn4', 'Kn4'])
                scale = '50mas'
            if fov == '2.25 x 3.2':
                filt = col1.selectbox('Filter: ', ['Hn2', 'Kn2'])
                scale = '50mas'
            if fov == '2.4 x 3.2':
                filt = col1.selectbox('Filter: ', ['Jn3', 'Hn3', 'Kn3'])
                scale = '50mas'
            if fov == '1.6 x 6.4':
                filt = col1.selectbox('Filter: ', ['Zbb', 'Jbb', 'Hbb', 'Kbb', 'Kcb'])
                scale = '100mas'
            if fov == '3.2 x 6.4':
                filt = col1.selectbox('Filter: ', ['Zn4', 'Jn1', 'Hn5', 'Kn5', 'Kc5'])
                scale = '100mas'
            if fov == '3.6 x 6.4':
                filt = col1.selectbox('Filter: ', ['Hn1', 'Kn1'])
                scale = '100mas'
            if fov == '4.2 x 6.4':
                filt = col1.selectbox('Filter: ', ['Jn2', 'Jn4', 'Hn4', 'Kn4', 'Kc4'])
                scale = '100mas'
            if fov == '4.5 x 6.4':
                filt = col1.selectbox('Filter: ', ['Hn2', 'Kn2'])
                scale = '100mas'
            if fov == '4.8 x 6.4':
                filt = col1.selectbox('Filter: ', ['Jn3', 'Hn3', 'Kn3', 'Kc3'])
                scale = '100mas'

    #   Find the relevant wavelengths for the calculation
    filts = np.array([fil.lower() for fil in modes['Filter']])
    wavs = np.where(filts == filt.lower())[0]
    lmin = modes['λ (nm) min'][wavs]
    lmax = modes['λ (nm) max'][wavs]
    lmin = lmin.values
    lmax = lmax.values
    if not isinstance(lmin, str): lmin = lmin[0]
    if not isinstance(lmax, str): lmax = lmax[0]
    if '*' in lmin: lmin = lmin.replace('*', '')
    if '*' in lmax: lmax = lmax.replace('*', '')
    lmin = float(lmin)
    lmax = float(lmax)
    lmean = np.mean([lmin, lmax])

    #   Convert selected values to inputs for ETC code
    etc_scale = float(scale.split('mas')[0]) * 1e-3
    etc_fov = [float(fov.split('x')[0]), float(fov.split('x')[1])]

    #   Setup for more configuration of ETC code
    calc = side_container.radio('Calculate: ', ['Signal-to-Noise Ratio (SNR)', 'Exposure Time'])
    col1.subheader('Selected Configuration:')
    col1.write('Selected Mode: ' + mode)
    if calc == 'Signal-to-Noise Ratio (SNR)':
        itime = side_container.number_input('Frame Integration Time (Seconds): ', min_value=0., value=30.)
        nframes = side_container.number_input('Number of Frames: ', min_value=0, value=1)
        col1.write('Calculating SNR for Input Integration Time')
        col1.markdown('Total Integration Time: ' + str(itime * nframes))
        etc_calc = 'snr'
        plot_subhead = 'SNR per Spectral Flux Element:'
        snr = 10.
    elif calc == 'Exposure Time':
        snr = side_container.number_input('SNR: ', min_value=0., value=5.)
        etc_calc = 'exptime'
        plot_subhead = 'Exposure Time Required for Input SNR per Spectral Flux Element:'
        itime = 10.
        nframes = 1.
        col1.write('Calculating Integration Time for Input SNR')

    #   select flux type and input the source flux
    side_cont = side_container.container()
    side_cont.subheader('Source Properties')
    fl = side_cont.selectbox('Input Flux Method:', ['Magnitude', 'Flux Density', 'Integrated Flux over Bandpass'])
    if fl == 'Magnitude':
        side_col1, side_col2 = side_cont.columns([3, 1])
        mag = side_col1.number_input('Magnitude: ', value=20.)
        veg = side_col2.radio('Magnitude Standard:', ('Vega', 'AB'))
        if veg == 'AB':
            # ETC only takes vega input so this converts it from AB to vega, kinda silly.
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
        fint = None
        flambda = None
    elif fl == 'Flux Density':
        flambda = side_cont.number_input('Ergs/s/cm^2/Angstrom * 10^-19: ', value=1.62)
        flambda = flambda * 1e-19
        fint = None
        mag = None
    elif fl == 'Integrated Flux over Bandpass':
        fint = side_cont.number_input('Ergs/s/cm^2 * 10^-17: ', value=4.)
        fint = fint * 1e-17
        flambda = None
        mag = None
    #   select the type of source
    source = side_container.radio('Source Type:', ('Point Source', 'Extended (Mag/Sq. Arcsec.)'))
    if source == 'Point Source':
        etc_source = 'point_source'
    elif source == 'Extended (Mag/Sq. Arcsec.)':
        etc_source = 'extended'

    #   Print the configured options for user display
    col1.write('Filter: ' + filt)
    col1.write('Plate Scale: ' + scale)
    col1.write('Field of View: ' + fov)
    if mode == 'IFS':
        col2.subheader('Available OSIRIS Modes:')
        col2.dataframe(modes, height=200)
        spec = side_container.selectbox('Spectrum Shape:', ['Vega', 'Flat', 'Emission',
                                                            'Phoenix Stellar Library Spectrum', 'Custom'])
    else:
        spec = None

    #   plot the field of view in the second column, filter curve if it exists in the info directory
    fov_img = plot_fov(fov)
    if mode == 'Imager':
        col2.image(fov_img, width=500)
        filt_files = glob.glob(simdir + '/info/*_imag_*.dat')
        filt_index = 0
        for filt_file in filt_files:
            if filt.lower() in filt_file.lower():
                plot_filt = col2.button('Plot Filter Curve')
                if plot_filt:
                    filt_img = plot_filter(filt_files[filt_index])
                    col2.plotly_chart(filt_img)
            filt_index += 1
    elif mode == 'IFS':
        col2.image(fov_img, width=400)
        filt_files = glob.glob(simdir + '/info/*_spec_*.dat')
        filt_index = 0
        for filt_file in filt_files:
            if filt.lower() in filt_file.lower():
                plot_filt = col1.button('Plot Filter Curve')
                if plot_filt:
                    filt_img = plot_filter(filt_files[filt_index])
                    col1.plotly_chart(filt_img, )
            filt_index += 1

    #   Deal with input spectrum
    if spec == 'Emission':
        line_width = side_container.number_input('Line Width (nm):', min_value=1., value=200.)
        lam_obs = side_container.number_input('Line Wavelength (nm)', min_value=lmin, value=lmean, max_value=lmax)
        lam_obs = lam_obs * 1e-3
    else:
        line_width = 200.
        lam_obs = lmean * 1e-3
    if spec == 'Phoenix Stellar Library Spectrum':
        pcol1, pcol2 = side_container.columns(2)
        teff = pcol1.selectbox('Teff (K):',
                               [str(int(i)) for i in np.append(np.arange(23, 70) * 100, np.arange(35, 61) * 200)])
        logg = pcol1.selectbox('log(g):', [str(i)[0:3] for i in np.arange(0, 13) / 2.])
        feh = pcol2.selectbox('Fe/H:', ['-4.0', '-3.0', '-2.0', '-1.5', '-1.0', '-0.5', '-0.0', '+0.5', '+1.0'],
                              index=6)
        aom = pcol2.selectbox('alpha/M:', ['-0.2', '0.0', '+0.2', '+0.4', '+0.6', '+0.8', '+1.0', '+1.2'], index=1)
        if aom == '0.0':
            specfile = 'lte0' + teff + '-' + logg + '0' + feh + '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        else:
            specfile = 'lte0' + teff + '-' + logg + '0' + feh + '.Alpha=' + aom + \
                       '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        specdir = simdir + 'model_spectra/phoenix/' + specfile
        if os.path.isfile(specdir):
            side_container.write('Using file: ' + specfile)
            wav = fits.open(simdir + 'model_spectra/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data / 10.
            inspec = fits.open(specdir)[0].data
            specfig = go.Figure()
            wav = wav[np.argmin(np.abs(wav - lmin)):np.argmin(np.abs(wav - lmax))]
            inspec = inspec[np.argmin(np.abs(wav - lmin)):np.argmin(np.abs(wav - lmax))]
            specfig.add_trace(go.Scatter(x=wav, y=inspec))
            specfig.update_layout(yaxis_title='Flux erg/s/cm^2/cm', xaxis_title='Wavelength (nm)',
                                  title='Phoenix Spectrum Curve')
            col2.plotly_chart(specfig)
            from misc_funcs import get_filterdat
            filterdat = get_filterdat(filt, simdir=simdir, mode=mode)
            zp = filterdat["zp"]
            dxspectrum = int(ceil(log10(lmax / lmin) / log10(1.0 + 1.0 / (4000. * 2.0))))
            wave = np.linspace(lmin / 1e3, lmax / 1e3, dxspectrum)
            if mag is not None:
                # convert to flux density (flambda)
                flux_phot = zp * 10 ** (-0.4 * mag)  # photons/s/m^2
                if source == 'extended':
                    flux_phot = flux_phot * (scale ** 2)
            elif flambda is not None:
                flux_phot = zp * 10 ** (-0.4 * mag)
                if source == 'extended':
                    flux_phot = flux_phot * (scale ** 2)
            elif fint is not None:
                # Compute flux_phot from flux
                E_phot = (h * c) / (lmean * Ang)
                flux_phot = 1e4 * fint / E_phot
                if source == 'extended':
                    flux_phot = flux_phot * (scale ** 2)
            spec_func = interpolate.interp1d(wav / 1e4, inspec)
            spec_temp = spec_func(wave)
            specinput = [np.array(wave), np.array(spec_temp)]
        else:
            side_container.write('Unable to find file ' + specfile + ', please make another selection.')
    elif spec == 'Custom':
        side_cont.empty()
        dxspectrum = int(ceil(log10(lmax / lmin) / log10(1.0 + 1.0 / (4000. * 2.0))))
        wave = np.linspace(lmin / 1e3, lmax / 1e3, dxspectrum)
        specinput = [np.array(wave), np.zeros((dxspectrum))]
        page_container.subheader('Input Spectrum Generator')
        specol1, specol2, specol3 = page_container.columns(3)
        l = 1
        k = 1
        spec1 = specol1.selectbox('Add Spectrum:', ['None', 'Vega', 'Flat', 'Emission',
                                                    'Phoenix Stellar Library Spectrum'],
                                  key='specgen' + str(k))
        if spec1 != 'None':
            sfl1 = specol2.selectbox('Input Flux Method:', ['Magnitude', 'Flux Density',
                                                            'Integrated Flux over ''Bandpass'],
                                     key='sfl' + str(k))
            if spec1 == 'Emission':
                line_width = specol3.number_input('Line Width (nm):', min_value=1., value=200.,
                                                  key='lw' + str(k))
                lam_obs = specol3.number_input('Line Wavelength (nm)', min_value=lmin, value=lmean,
                                               max_value=lmax, key='lamobs' + str(k))
            if spec1 == 'Phoenix Stellar Library Spectrum':
                teff = specol1.selectbox('Teff (K):', [str(int(i)) for i in
                                                       np.append(np.arange(23, 70) * 100, np.arange(35, 61) * 200)],
                                         key='teff' + str(k))
                logg = specol1.selectbox('log(g):', [str(i)[0:3] for i in np.arange(0, 13) / 2.], key='logg' + str(k))
                feh = specol2.selectbox('Fe/H:',
                                        ['-4.0', '-3.0', '-2.0', '-1.5', '-1.0', '-0.5', '-0.0', '+0.5', '+1.0'],
                                        index=6, key='feh' + str(k))
                aom = specol2.selectbox('alpha/M:', ['-0.2', '0.0', '+0.2', '+0.4', '+0.6', '+0.8', '+1.0', '+1.2'],
                                        index=1, key='aom' + str(k))
            else:
                aom = None
                teff = None
                logg = None
                feh = None
            if sfl1 == 'Magnitude':
                mag = specol3.number_input('Magnitude: ', value=20., key='mag' + str(k))
                veg = specol3.radio('Magnitude Standard:', ('Vega', 'AB'), key='veg' + str(k))
                if veg == 'AB':
                    # ETC only takes vega input so this converts it from AB to vega, kinda silly.
                    mag = ab2vega(mag, lmean=lmean)
                fint = None
                flambda = None
            elif sfl1 == 'Flux Density':
                flambda = specol3.number_input('Ergs/s/cm^2/Angstrom * 10^-19: ', value=1.62, key='flux' + str(k))
                flambda = flambda * 1e-19
                fint = None
                mag = None
            elif sfl1 == 'Integrated Flux over Bandpass':
                fint = specol3.number_input('Ergs/s/cm^2 * 10^-17: ', value=4., key='intflux' + str(k))
                fint = fint * 1e-17
                flambda = None
                mag = None
            spec_temp = gen_spec(spec1, wave=wave, scale=etc_scale, fint=fint, flambda=flambda, mag=mag,
                                 lam_obs=lam_obs, line_width=line_width, source=source, simdir=simdir,
                                 filt=filt, mode=mode, teff=teff, logg=logg, feh=feh, aom=aom)
            specinput[1] = specinput[1] + spec_temp
            k = 2
            spec2 = specol1.selectbox('Add Spectrum:', ['None', 'Vega', 'Flat', 'Emission',
                                                        'Phoenix Stellar Library Spectrum'],
                                      key='specgen' + str(k))
            if spec2 != 'None':
                sfl2 = specol2.selectbox('Input Flux Method:', ['Magnitude', 'Flux Density',
                                                                'Integrated Flux over ''Bandpass'],
                                         key='sfl' + str(k))
                if spec2 == 'Emission':
                    line_width = specol3.number_input('Line Width (nm):', min_value=1., value=200.,
                                                      key='lw' + str(k))
                    lam_obs = specol3.number_input('Line Wavelength (nm)', min_value=lmin, value=lmean,
                                                   max_value=lmax, key='lamobs' + str(k))
                if spec2 == 'Phoenix Stellar Library Spectrum':
                    teff = specol1.selectbox('Teff (K):', [str(int(i)) for i in
                                                           np.append(np.arange(23, 70) * 100,
                                                                     np.arange(35, 61) * 200)],
                                             key='teff' + str(k))
                    logg = specol1.selectbox('log(g):', [str(i)[0:3] for i in np.arange(0, 13) / 2.],
                                             key='logg' + str(k))
                    feh = specol2.selectbox('Fe/H:',
                                            ['-4.0', '-3.0', '-2.0', '-1.5', '-1.0', '-0.5', '-0.0', '+0.5',
                                             '+1.0'],
                                            index=6, key='feh' + str(k))
                    aom = specol2.selectbox('alpha/M:',
                                            ['-0.2', '0.0', '+0.2', '+0.4', '+0.6', '+0.8', '+1.0', '+1.2'],
                                            index=1, key='aom' + str(k))
                if sfl2 == 'Magnitude':
                    mag = specol3.number_input('Magnitude: ', value=20., key='mag' + str(k))
                    veg = specol3.radio('Magnitude Standard:', ('Vega', 'AB'), key='veg' + str(k))
                    if veg == 'AB':
                        # ETC only takes vega input so this converts it from AB to vega, kinda silly.
                        mag = ab2vega(mag, lmean=lmean)
                    flambda = None
                elif sfl2 == 'Flux Density':
                    flambda = specol3.number_input('Ergs/s/cm^2/Angstrom * 10^-19: ', value=1.62, key='flux' + str(k))
                    flambda = flambda * 1e-19
                    fint = None
                    mag = None
                elif sfl2 == 'Integrated Flux over Bandpass':
                    fint = specol3.number_input('Ergs/s/cm^2 * 10^-17: ', value=4., key='intflux' + str(k))
                    fint = fint * 1e-17
                    flambda = None
                    mag = None
                spec_temp = gen_spec(spec2, wave=wave, scale=etc_scale, fint=fint, flambda=flambda,
                                     lam_obs=lam_obs, line_width=line_width, mag=mag, source=source, simdir=simdir,
                                     filt=filt, mode=mode, teff=teff, logg=logg, feh=feh, aom=aom)
                specinput[1] = specinput[1] + spec_temp
                k = 3
                spec3 = specol1.selectbox('Add Spectrum:', ['None', 'Vega', 'Flat', 'Emission',
                                                            'Phoenix Stellar Library Spectrum'],
                                          key='specgen' + str(k))
                if spec3 != 'None':
                    sfl3 = specol2.selectbox('Input Flux Method:', ['Magnitude', 'Flux Density',
                                                                    'Integrated Flux over ''Bandpass'],
                                             key='sfl' + str(k))
                    if spec3 == 'Emission':
                        line_width = specol3.number_input('Line Width (nm):', min_value=1., value=200.,
                                                          key='lw' + str(k))
                        lam_obs = specol3.number_input('Line Wavelength (nm)', min_value=lmin,
                                                       value=lmean,
                                                       max_value=lmax, key='lamobs' + str(k))
                    if spec3 == 'Phoenix Stellar Library Spectrum':
                        teff = specol1.selectbox('Teff (K):', [str(int(i)) for i in
                                                               np.append(np.arange(23, 70) * 100,
                                                                         np.arange(35, 61) * 200)],
                                                 key='teff' + str(k))
                        logg = specol1.selectbox('log(g):', [str(i)[0:3] for i in np.arange(0, 13) / 2.],
                                                 key='logg' + str(k))
                        feh = specol2.selectbox('Fe/H:',
                                                ['-4.0', '-3.0', '-2.0', '-1.5', '-1.0', '-0.5', '-0.0', '+0.5',
                                                 '+1.0'],
                                                index=6, key='feh' + str(k))
                        aom = specol2.selectbox('alpha/M:',
                                                ['-0.2', '0.0', '+0.2', '+0.4', '+0.6', '+0.8', '+1.0', '+1.2'],
                                                index=1, key='aom' + str(k))
                    if sfl3 == 'Magnitude':
                        mag = specol3.number_input('Magnitude: ', value=20., key='mag' + str(k))
                        veg = specol3.radio('Magnitude Standard:', ('Vega', 'AB'), key='veg' + str(k))
                        if veg == 'AB':
                            # ETC only takes vega input so this converts it from AB to vega, kinda silly.
                            mag = ab2vega(mag, lmean=lmean)
                        fint = None
                        flambda = None
                    elif sfl3 == 'Flux Density':
                        flambda = specol3.number_input('Ergs/s/cm^2/Angstrom * 10^-19: ', value=1.62,
                                                       key='flux' + str(k))
                        flambda = flambda * 1e-19
                        fint = None
                        mag = None
                    elif sfl3 == 'Integrated Flux over Bandpass':
                        fint = specol3.number_input('Ergs/s/cm^2 * 10^-17: ', value=4., key='intflux' + str(k))
                        fint = fint * 1e-17
                        flambda = None
                        mag = None
                    spec_temp = gen_spec(spec3, wave=wave, scale=etc_scale, fint=fint, flambda=flambda,
                                         lam_obs=lam_obs, line_width=line_width, mag=mag, source=source,
                                         simdir=simdir, filt=filt, mode=mode, teff=teff,
                                         logg=logg, feh=feh, aom=aom)
                    specinput[1] = specinput[1] + spec_temp
                    k = 4
                    spec4 = specol1.selectbox('Add Spectrum:', ['None', 'Vega', 'Flat', 'Emission',
                                                                'Phoenix Stellar Library Spectrum'],
                                              key='specgen' + str(k))

                    if spec4 != 'None':
                        sfl4 = specol2.selectbox('Input Flux Method:', ['Magnitude', 'Flux Density',
                                                                        'Integrated Flux over ''Bandpass'],
                                                 key='sfl' + str(k))
                        if spec4 == 'Emission':
                            line_width = specol3.number_input('Line Width (nm):', min_value=1., value=200.,
                                                              key='lw' + str(k))
                            lam_obs = specol3.number_input('Line Wavelength (nm)', min_value=lmin,
                                                           value=lmean,
                                                           max_value=lmax, key='lamobs' + str(k))
                        if spec4 == 'Phoenix Stellar Library Spectrum':
                            teff = specol1.selectbox('Teff (K):', [str(int(i)) for i in
                                                                   np.append(np.arange(23, 70) * 100,
                                                                             np.arange(35, 61) * 200)],
                                                     key='teff' + str(k))
                            logg = specol1.selectbox('log(g):', [str(i)[0:3] for i in np.arange(0, 13) / 2.],
                                                     key='logg' + str(k))
                            feh = specol2.selectbox('Fe/H:',
                                                    ['-4.0', '-3.0', '-2.0', '-1.5', '-1.0', '-0.5', '-0.0', '+0.5',
                                                     '+1.0'],
                                                    index=6, key='feh' + str(k))
                            aom = specol2.selectbox('alpha/M:',
                                                    ['-0.2', '0.0', '+0.2', '+0.4', '+0.6', '+0.8', '+1.0', '+1.2'],
                                                    index=1, key='aom' + str(k))
                        if sfl4 == 'Magnitude':
                            mag = specol3.number_input('Magnitude: ', value=20., key='mag' + str(k))
                            veg = specol3.radio('Magnitude Standard:', ('Vega', 'AB'), key='veg' + str(k))
                            if veg == 'AB':
                                # ETC only takes vega input so this converts it from AB to vega, kinda silly.
                                mag = ab2vega(mag, lmean=lmean)
                            fint = None
                            flambda = None
                        elif sfl4 == 'Flux Density':
                            flambda = specol3.number_input('Ergs/s/cm^2/Angstrom * 10^-19: ', value=1.62,
                                                           key='flux' + str(k))
                            flambda = flambda * 1e-19
                            fint = None
                            mag = None
                        elif sfl4 == 'Integrated Flux over Bandpass':
                            fint = specol3.number_input('Ergs/s/cm^2 * 10^-17: ', value=4., key='intflux' + str(k))
                            fint = fint * 1e-17
                            flambda = None
                            mag = None
                        spec_temp = gen_spec(spec4, wave=wave, scale=etc_scale, fint=fint,
                                             lam_obs=lam_obs, line_width=line_width, flambda=flambda, mag=mag,
                                             source=source, simdir=simdir, filt=filt, mode=mode, teff=teff,
                                             logg=logg, feh=feh, aom=aom)
                        specinput[1] = specinput[1] + spec_temp
                        k = 5
                        spec5 = specol1.selectbox('Add Spectrum:', ['None', 'Vega', 'Flat', 'Emission',
                                                                    'Phoenix Stellar Library Spectrum'],
                                                  key='specgen' + str(k))
                        if spec5 != 'None':
                            sfl5 = specol2.selectbox('Input Flux Method:', ['Magnitude', 'Flux Density',
                                                                            'Integrated Flux over ''Bandpass'],
                                                     key='sfl' + str(k))
                            if spec5 == 'Emission':
                                line_width = specol3.number_input('Line Width (nm):', min_value=1., value=200.,
                                                                  key='lw' + str(k))
                                lam_obs = specol3.number_input('Line Wavelength (nm)', min_value=lmin,
                                                               value=lmean,
                                                               max_value=lmax, key='lamobs' + str(k))
                            if spec5 == 'Phoenix Stellar Library Spectrum':
                                teff = specol1.selectbox('Teff (K):', [str(int(i)) for i in
                                                                       np.append(np.arange(23, 70) * 100,
                                                                                 np.arange(35, 61) * 200)],
                                                         key='teff' + str(k))
                                logg = specol1.selectbox('log(g):', [str(i)[0:3] for i in np.arange(0, 13) / 2.],
                                                         key='logg' + str(k))
                                feh = specol2.selectbox('Fe/H:',
                                                        ['-4.0', '-3.0', '-2.0', '-1.5', '-1.0', '-0.5', '-0.0',
                                                         '+0.5', '+1.0'],
                                                        index=6, key='feh' + str(k))
                                aom = specol2.selectbox('alpha/M:',
                                                        ['-0.2', '0.0', '+0.2', '+0.4', '+0.6', '+0.8', '+1.0',
                                                         '+1.2'],
                                                        index=1, key='aom' + str(k))
                            if sfl5 == 'Magnitude':
                                mag = specol3.number_input('Magnitude: ', value=20, key='mag' + str(k))
                                veg = specol3.radio('Magnitude Standard:', ('Vega', 'AB'), key='veg' + str(k))
                                if veg == 'AB':
                                    # ETC only takes vega input so this converts it from AB to vega, kinda silly.
                                    mag = ab2vega(mag, lmean=lmean)
                                fint = None
                                flambda = None
                            elif sfl5 == 'Flux Density':
                                flambda = specol3.number_input('Ergs/s/cm^2/Angstrom * 10^-19: ', value=1.62,
                                                               key='flux' + str(k))
                                flambda = flambda * 1e-19
                                fint = None
                                mag = None
                            elif sfl5 == 'Integrated Flux over Bandpass':
                                fint = specol3.number_input('Ergs/s/cm^2 * 10^-17: ', value=4., key='intflux' + str(k))
                                fint = fint * 1e-17
                                flambda = None
                                mag = None
                            spec_temp = gen_spec(spec5, wave=wave, scale=etc_scale, fint=fint,
                                                 lam_obs=lam_obs, line_width=line_width, flambda=flambda, mag=mag,
                                                 source=source, simdir=simdir, filt=filt, mode=mode, teff=teff,
                                                 logg=logg, feh=feh, aom=aom)
                            specinput[1] = specinput[1] + spec_temp
                            k = 6
        specfig = go.Figure()
        specfig.add_trace(go.Scatter(x=specinput[0], y=specinput[1]))
        specfig.update_layout(yaxis_title='Photons/s/um/m2', xaxis_title='Wavelength (nm)',
                              title='Custom Spectrum Curve')
        specol1.plotly_chart(specfig)
    else:
        specinput = None

    ##input PSF configuration
    with page_container.container():
        psf_col1, psf_col2 = page_container.columns(2)
        psf_col1.subheader('Point Spread Function (PSF) Configuration:')
        psfmode = psf_col1.selectbox('Select PSF Option:', ['Generated Analytic PSF', 'Pre-generated SCAO PSF'])
        if psfmode == 'Pre-generated SCAO PSF':
            psf_input = None
            if filt[0] != 'K': psf_col1.markdown('WARNING: selected filter not in K, PSF will mis-match simulated '
                                                 'wavelengths.')
        elif psfmode == 'Generated Analytic PSF':
            with st.spinner('Generating PSF....'):
                strehl = psf_col1.number_input('Strehl Ratio:', value=0.5)
                fried = psf_col1.number_input('Fried Parameter r0 (cm):', value=20.)
                lam_obs = [lmin * 10., lmean * 10., lmax * 10.]
                wav = psf_col1.radio('PSF Wavelength:',
                                     ['Monochromatic (Central Wavelength: ' + str(lmean / 1e3) + 'um)',
                                      'Averaged over Bandpass (' + str(lmin / 1e3) + 'um, ' + str(
                                          lmean / 1e3) + 'um, ' + str(lmax / 1e3) + 'um)'])
                if wav == 'Monochromatic (Central Wavelength: ' + str(lmean / 1e3) + 'um)':
                    lam_obs = lam_obs[1]
                elif wav == 'Averaged over Bandpass (' + str(lmin / 1e3) + 'um, ' + str(lmean / 1e3) + 'um, ' + str(
                        lmax / 1e3) + 'um)':
                    lam_obs = lam_obs
                # wfe = col1.number_input('Additional Input Wavefront Error (nm):',value=0.)
                # blurring = col1.checkbox('Include Gaussian Blurring')
                # if blurring is not None:
                #     blur = True
                st.cache
                psf_input = analytic_psf(strehl, lam_obs, etc_scale, fried_parameter=fried, verbose=False, stack=True,
                                         simdir=simdir)
                psf_input = psf_input[-1] / np.sum(psf_input[-1])
                psf_col2.subheader('PSF Preview:')
                s = np.shape(psf_input)
                logscale = lambda a, im: np.log10(a * (im / np.max(im)) + 1.0) / (np.log10(a))
                psf_col2.image(logscale(1000., np.abs(
                    psf_input[int(s[0] / 2 - 20):int(s[0] / 2 + 20), int(s[1] / 2 - 20):int(s[1] / 2 + 20)])),
                               clamp=True, caption='Generated PSF (Central ' + str(40. * etc_scale)[
                                                                               0:4] + ' arcseconds) - Log Scale',
                               width=250)

    page_container.write('--------------------------------------------------------------------------')

    #   Define default radius value based on scale and lambda.
    if etc_scale == 0.02:
        radiusl = 1. * lmean * 1e2 * 206265 / (1.26e11)
    elif etc_scale == 0.035:
        radiusl = 1.5 * lmean * 1e2 * 206265 / (1.26e11)
    elif etc_scale == 0.05:
        radiusl = 20 * lmean * 1e2 * 206265 / (1.26e11)
    else:
        radiusl = 40 * lmean * 1e2 * 206265 / (1.26e11)

    #   Container for Results and result plots.
    with page_container.container():
        col3, col4 = page_container.columns(2)
        col3.subheader('OSIRIS Calculation Results:')
        #   Calculation
        aperture = col3.slider(label='Aperture Radius (arcseconds):', min_value=0.01,
                               max_value=float(np.min(etc_fov) / 2.), value=float(radiusl))
        if aperture / etc_scale < 1: col2.markdown('WARNING: You have selected an aperture radius less than 1 pixel.')
        res, fig, csvarr = OSIRIS_ETC(filter=filt, mag=mag, flambda=flambda,
                                      fint=fint, itime=itime, nframes=nframes, snr=snr,
                                      aperture=aperture, gain=3.04, readnoise=7., darkcurrent=0.05,
                                      scale=etc_scale, resolution=4000, collarea=78.5, positions=[0, 0],
                                      bgmag=None, efftot=None, mode=mode.lower(), calc=etc_calc,
                                      spectrum=spec, specinput=specinput, lam_obs=lam_obs, line_width=line_width,
                                      png_output=None, source=etc_source, source_size=0.2,
                                      csv_output=True, fov=etc_fov, psf_loc=[8.8, 8.8],
                                      psf_time=1.4, verb=1, psf_input=psf_input, simdir=simdir, psfdir=psfdir, test=0)
        #   parse the results into a nice readable format and write them into the two columns
        val_lens = np.array([len(i) for i in res.values()])
        ind = np.where(val_lens > 0)[0]
        vals = list(res.values())
        keys = list(res.keys())
        for i in ind:
            string = str(keys[i]) + ': ' + str(vals[i])
            col3.write(string)
        if (mode == 'Imager' and etc_source == 'extended'):
            col4.write('The extended-source simulation assumes a perfectly even brightness ' +
                       'distribution, so there is no meaningful simulated image.')
        elif mode == 'Imager':
            col4.subheader('Simulated Image:')
            col4.plotly_chart(fig)
        else:
            col4.subheader('Simulated ' + plot_subhead)
            col4.plotly_chart(fig)
        if mode == 'IFS':
            csv_output = col4.button('Download ' + calc + ' Spectrum as CSV')
            if csv_output:
                if etc_calc == 'exptime':
                    header = "Wavelength(microns),Int_Time_PeakFlux(s),Int_Time_MedianFlux(s)," \
                             "Int_Time_MeanFlux(s),Int_Time_Total_Aperture_Flux(s)"
                    csvlabel = 'inputSNR' + str(int(snr))
                elif etc_calc == 'snr':
                    header = "Wavelength(microns),SNR_Peak,SNR_Median,SNR_Mean,SNR_Aperture_Total"
                    csvlabel = 'inputITime' + str(int(itime * nframes))
                df = pd.DataFrame(csvarr, columns=header.split(','))
                if fl == 'Magnitude':
                    csv_mag = veg + 'Mag' + str(int(mag))
                elif fl == 'Flux Density':
                    csv_mag = 'Flambda' + str(flambda * 1e19)[0:4] + 'e19'
                elif fl == 'Integrated Flux over Bandpass':
                    csv_mag = 'Fint' + str(fint * 1e17)[0:4] + 'e17'
                csv_title = 'OSIRIS_' + mode + '_ETC_' + etc_calc + '_' + filt + '_' + scale + \
                            '_' + csvlabel + '_' + etc_source + '_' + csv_mag
                tmp_download_link = download_link(df, csv_title + '.csv',
                                                  'Click Here to Download CSV: ' +
                                                  csv_title + '.csv')
                col4.markdown(tmp_download_link, unsafe_allow_html=True)


def vega_spec():
    dat = fits.open('/Volumes/Backup Plus/osiris/sim/model_spectra/vega_all.fits')
    spec = dat[0].data
    head = dat[0].header
    cdelt1 = head["cdelt1"]
    crval1 = head["crval1"]

    nelem = spec.shape[0]
    specwave = (np.arange(nelem)) * cdelt1 + crval1
    specwave = specwave/1e4
    import csv
    with open("/Users/nils-erikrundquist/Documents/vegaspec.dat", 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(specwave, spec))

#
# backCube = backgroundCube * itime * nframes + darkcurrent * itime * nframes + readnoise ** 2.0 * nframes
# backCube_tot = np.random.poisson(lam=backCube, size=totalObservedCube.shape).astype("float64")
# simbackCube = backCube_tot / (itime * nframes)
# fits.writeto(simdir + 'Sim_Cube_Tot.fits', simCube)
# fits.writeto(simdir + 'Sim_Back_Cube_Tot.fits', simbackCube)
# fits.writeto(simdir + 'Background_SUBT_sim_Cube.fits', simCube - simbackCube)


''':exception
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
def galaxy_spec(specname, simdir='~/osiris/sim/'):

    if specname =='CGCG 049-057 (Irr; LIRG)':
        dat = fits.open(simdir + 'model_spectra/brown/cgcg_049-057_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname =='IC 4553 (ULIRG; Arp 220)':
        dat = fits.open(simdir + 'model_spectra/brown/ic_4553_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'II Zw 096 (Pec; Merger/Starburst)':
        dat = fits.open(simdir + 'model_spectra/brown/ii_zw_096_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'Mrk 33 (Wolf-Rayet Galaxy)':
        dat = fits.open(simdir + 'model_spectra/brown/mrk_33_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 0337 (SBd; SF)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_0337_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 0695 (S0; Interacting)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_0695_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 3079 (SB(s)c; SF/Seyfert 2)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_3079_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 3521 (SABbc; SF)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_3521_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 3690 (Merger/Wolf-Rayet Galaxy)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_3690_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 4125 (E6 pec; -)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_4125_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 4138 (SA(r)0; SF/AGN)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_4138_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 4552 (E w/ UV upturn)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_4552_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 4725 (SABab; Seyfert 2)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_4725_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 5256 (Pec; Merger)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_5256_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 5953 (Sa; SF/AGN)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_5953_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 6090 (Pec; Merger/LIRG)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_6090_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'NGC 6240 (Pec; AGN)':
        dat = fits.open(simdir + 'model_spectra/brown/ngc_6240_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])
    if specname == 'UGCA 219 (Blue Compact Dwarf)':
        dat = fits.open(simdir + 'model_spectra/brown/ugca_219_spec.fits')
        data = dat[len(dat)-1].data
        wvl = np.array([i[0] for i in data]) / 10.
        spec = np.array([i[1] for i in data])

    return wvl, spec

def readthing():
    from astropy.io import fits
    simdir = '/Volumes/Backup Plus/osiris/sim/'
    modes = pd.read_csv(simdir + 'info/osiris_modes.csv', header=0)
    lmin = 900
    lmax = 2400
    lmean = np.mean([lmin, lmax])
    dxspectrum = int(ceil(log10(lmax / lmin) / log10(1.0 + 1.0 / (4000. * 2.0))))
    wave = np.linspace(lmin / 1e3, lmax / 1e3, dxspectrum)
    import glob
    files = glob.glob(simdir+'model_spectra/SSP_M11/*/*', recursive=True)

    for filename in files:
        print(filename)
        dat = np.genfromtxt(filename)
        wvl = dat[:, -2] / 1e4
        spec = dat[:, -1]
        print(np.mean((wvl[1:]/(wvl[1:]-wvl[0:-1]))))
        if np.max(wvl) < np.max(wave):
            print('Input wavelength must be redshifted to encompass bandpass.')
            z_i = (np.max(wave) - np.max(wvl)) / np.max(wvl)
            print('Automatically redshifting to redshift %f, please enter your own redshift.' % z_i)

            wvl = wvl*(1+z_i)
        minwvl = np.argmin(np.abs(wvl-wave[0]))
        maxwvl = np.argmin(np.abs(wvl-wave[-1])) + 1
        cutwvl = wvl[minwvl:maxwvl]
        print('in wavelength bandpass:')
        print(np.mean((cutwvl[1:] / (cutwvl[1:] - cutwvl[0:-1]))))
        if np.mean((cutwvl[1:] / (cutwvl[1:] - cutwvl[0:-1]))) < 8000:
            print('REMOVING FILE: ' + filename)
            os.remove(filename)

'''