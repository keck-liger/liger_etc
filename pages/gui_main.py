import streamlit as st
import numpy as np
import pandas as pd
from liger_snr_sim import LIGER_ETC
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
from scipy import interpolate
from misc_funcs import get_filterdat, extrap1d, ab2vega, gen_spec

# constants
c_km = 2.9979E5      # km/s
c = 2.9979E10       # cm/s
h = 6.626068E-27    # cm^2*g/s
k = 1.3806503E-16   # cm^2*g/(s^2*K)
Ang = 1E-8          # cm
mu = 1E-4           # cm

def exec_gui(page_container=None, side_container=None):
    if page_container is None:
        page_container = st
    if side_container is None:
        side_container = st.sidebar

    #@st.cache
    def plot_fov(fov):
        """
        :param fov: field of view as a string in "X x Y" where X and Y are numbers
        :return:    matplotlib figure for use by streamlit
        """
        fig = plt.figure(facecolor='none')
        ax = fig.add_subplot()
        ax.patch.set_alpha(0.)
        ffov = [float(x) for x in fov.split('x')]
        #ax.scatter([-ffov[0]/2., 0, ffov[0]/2.], [-ffov[1]/2., 0, ffov[1]/2.], alpha=1.)
        ax.set_ylim(-np.max(ffov)/2.,np.max(ffov)/2.)
        ax.set_xlim(-np.max(ffov)/2.,np.max(ffov)/2.)
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
        rect = Rectangle((-ffov[0]/2.,-ffov[1]/2.), ffov[0], ffov[1], linestyle='dashdot', linewidth=1.5, fill=0)
        ax.add_patch(rect)
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        # ... convert to a NumPy array ...
        X = np.asarray(buf)
        return(X)

    @st.cache
    def plot_filter(filterfile):
        """
        :param filterfile: directory of filter file to plot
        :return:    matplotlib figure for use by streamlit
        """
        dat = np.genfromtxt(filterfile, skip_header=2, usecols=(0,1))
        wvl = dat[:,0]
        transm = dat[:,1]
        filterfig = go.Figure()
        filterfig.add_trace(go.Scatter(x=wvl, y=transm))
        filterfig.update_layout(yaxis_title='Transmission (%)', xaxis_title='Wavelength (nm)', title='Filter Curve')
        return(filterfig)

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
        if isinstance(object_to_download,pd.DataFrame):
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
    container = page_container.beta_container()
    col1, col2 = container.beta_columns(2)
    col1.subheader('Liger Configuration')

    #   select the mode for the ETC -- Imager or IFS
    side_container.title('Exposure / Signal-To-Noise Setup')
    mode = side_container.radio("Please Select Liger Mode", ['IFS', 'Imager'])
    modes = pd.read_csv(simdir+'info/Liger_modes.csv', header=0, dtype=str)
    if mode == 'Imager':
        #   clarification of calculation
        calc_title = ''
        scale = '14mas'
        col1.markdown('Plate Scale ( Miliarcseconds per Spatial Pixel ): \n 14mas')
        col1.markdown('Field of View (Arcseconds x Arcseconds): \n 20.4 x 20.4')
        fov = '20.4 x 20.4'
        filt = col1.selectbox("Filter: ",
                            ['Zbb', 'Ybb', 'Jbb', 'Hbb', 'Kbb', 'Z', 'Y', 'J', 'H', 'K', 'Kcb','Zn3', 'Zn4', 'Jn1',
                            'Jn2', 'Jn3', 'Jn4', 'Hn1', 'Hn2', 'Hn3', 'Hn4', 'Hn5', 'Kn1', 'Kn2', 'Kn3', 'Kc3', 'Kn4',
                             'Kc4', 'Kn5', 'Kc5', 'FeII','Hcont','Y','J','Kp','BrGamma','Kcont','HeI_B','Kcb'])
    elif mode == 'IFS':
        #   clarification of calculation
        calc_title = ' ( per Spectral Element )'
        #   User can select whether to configure with the Filter and Plate scale or the Field of View
        subconf = side_container.radio("Configure With: ", ['Filter / Plate Scale', 'Field of View'])
        if subconf=='Filter / Plate Scale':
            filts = modes['Filter'].values
            scales = modes.columns[-4:].values
            filt = col1.selectbox("Filter: ", [f for f in filts.flatten()])
            #col1.write([s for s in scales.flatten()])
            if filt in ['Kcb','Kc3','Kc4','Kc5']:
                scale = '100mas'
                col1.markdown('Plate Scale ( Arcseconds per Spatial Pixel ): 100mas')
            else:
                scale = col1.select_slider('Plate Scale ( Arcseconds per Spatial Pixel )', [s for s in scales])
            fovs = modes[scale].iloc[(np.where(modes['Filter'] == filt)[0])].values
            if len(fovs)>1:
                fov = col1.select_slider('Field of View:',[f for f in fovs.flatten()])
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
    etc_scale = float(scale.split('mas')[0])*1e-3
    etc_fov = [float(fov.split('x')[0]), float(fov.split('x')[1])]

    #   Setup for more configuration of ETC code
    calc = side_container.radio('Calculate: ', ['Signal-to-Noise Ratio (SNR)', 'Exposure Time'])
    col1.subheader('Selected Configuration:')
    col1.write('Selected Mode: ' + mode)
    if calc == 'Signal-to-Noise Ratio (SNR)':
        if mode == 'Imager':
            input_title = 'Frame Integration Time (Seconds) per Bandpass: '
        elif mode == 'IFS':
            input_title = 'Frame Integration Time (Seconds) per Wavelength Element: '
        itime = side_container.number_input(input_title, min_value=0., value=30.)
        nframes = side_container.number_input('Number of Frames: ', min_value=0, value = 1)
        col1.write('Calculating SNR for Input Integration Time' + calc_title)
        col1.markdown('Total Integration Time: ' + str(itime*nframes))
        etc_calc = 'snr'
        plot_subhead = 'SNR per Spectral Flux Element:'
        snr = 10.
    elif calc == 'Exposure Time':
        if mode == 'Imager':
            input_title = 'SNR per Bandpass: '
        elif mode == 'IFS':
            input_title = 'SNR per Wavelength Element: '
        snr = side_container.number_input(input_title, min_value=0., value=5.)
        etc_calc = 'exptime'
        plot_subhead = 'Exposure Time Required for Input SNR per Spectral Flux Element:'
        itime = 10.
        nframes = 1.
        col1.write('Calculating Integration Time for Input SNR' + calc_title)

    #   select flux type and input the source flux
    side_cont = side_container.beta_container()
    side_cont.subheader('Source Properties')
    fl = side_cont.selectbox('Input Flux Method:', ['Magnitude', 'Flux Density','Integrated Flux over Bandpass'])
    side_col1, side_col2 = side_cont.beta_columns([3, 1])
    if fl == 'Magnitude':
        mag = side_col1.number_input('Magnitude: ', value = 20.)
        veg = side_col2.radio('Magnitude Standard:', ('Vega', 'AB'))
        if veg == 'AB':
            #ETC only takes vega input so this converts it from AB to vega, kinda silly.
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
    elif fl =='Integrated Flux over Bandpass':
        fint = side_cont.number_input('Ergs/s/cm^2 * 10^-17: ', value=4.)
        fint = fint * 1e-17
        flambda = None
        mag = None
    filterdat = get_filterdat(filt, simdir=simdir, mode=mode)
    zp = filterdat["zp"]
    #   select the type of source
    source = side_container.radio('Source Type:', ('Point Source', 'Extended (Mag/Sq. Arcsec.)'))
    if source == 'Point Source':
        etc_source = 'point_source'
        etc_prof = None
    elif source == 'Extended (Mag/Sq. Arcsec.)':
        etc_source = 'extended'
        prof_type = side_container.radio('Extended Brightness Profile:', ('Top-Hat', 'Sersic Profile'))
    #   calculate the total flux for the input values and source type
    #   generate the wavelengths for the input spectra if applicable
    dxspectrum = int(ceil(log10(lmax / lmin) / log10(1.0 + 1.0 / (4000. * 3.0))))
    wave = np.linspace(lmin / 1e3, lmax / 1e3, dxspectrum)
    if mag is not None:
        # convert to flux density (flambda)
        flux_phot = zp * 10 ** (-0.4 * mag)  # photons/s/m^2
        if etc_source == 'extended':
            flux_phot = flux_phot * (etc_scale* 2)
    elif flambda is not None:
        fnu = flambda / (Ang / ((lmean * 1e-9) ** 2 / c))
        ABmag = -2.5 * log10(fnu) - 48.60
        mag = ab2vega(ABmag, lmean=lmean)
        flux_phot = zp * 10 ** (-0.4 * mag)  # photons/s/m^2
        if etc_source == 'extended':
            flux_phot = flux_phot * (etc_scale* 2)
    elif fint is not None:
        # Compute flux_phot from flux
        E_phot = (h * c) / (lmean * 10. * Ang)
        flux_phot = 1e4 * fint / E_phot
        if etc_source == 'extended':
            flux_phot = flux_phot * (etc_scale* 2)

    if source == 'Extended (Mag/Sq. Arcsec.)':
        if prof_type == 'Sersic Profile':
            prof_side_col1, prof_side_col2 = side_container.beta_columns([2,3])
            from misc_funcs import sersic_profile
            s_ind = prof_side_col1.selectbox('Sersic Index:', [1,2,3,4])
            r_eff = prof_side_col2.number_input('Effective Radius ("):', min_value=etc_scale, value=0.5)
            etc_prof = sersic_profile(r_eff / etc_scale, s_ind, 500, 500, zp=zp, flux=flux_phot)
        else:
            etc_prof = None

    #   Print the configured options for user display
    col1.write('Filter: ' + filt)
    col1.write('Plate Scale: ' + scale)
    col1.write('Field of View: ' + fov)
    if mode == 'IFS':
        col2.subheader('Available Liger Modes:')
        col2.dataframe(modes, height=200)
        spec = side_container.selectbox('Spectrum Shape:',
                                        ['Vega', 'Flat', 'Emission', 'Phoenix Stellar Library Spectrum', 'Black Body',
                                         'Stellar Population Spectra - Maraston & Stromback (2011)','Custom', 'Upload'])
    else:
        spec = None
    #   plot the field of view in the second column, filter curve if it exists in the info directory
    fov_img = plot_fov(fov)
    if mode == 'Imager':
        col2.image(fov_img, width=500, clamp=True)
        filt_files = glob.glob(simdir+'/info/*_imag_*.dat')
        filt_index = 0
        for filt_file in filt_files:
            if filt.lower() in filt_file.lower():
                plot_filt = col2.button('Plot Filter Curve')
                if plot_filt:
                    filt_img = plot_filter(filt_files[filt_index])
                    col2.plotly_chart(filt_img)
            filt_index += 1
    elif mode == 'IFS':
        col2.image(fov_img, width=400, clamp=True)
        filt_files = glob.glob(simdir + '/info/*_spec_*.dat')
        filt_index = 0
        for filt_file in filt_files:
            if filt.lower() in filt_file.lower():
                plot_filt = col1.button('Plot Filter Curve')
                if plot_filt:
                    filt_img = plot_filter(filt_files[filt_index])
                    col1.plotly_chart(filt_img,)
            filt_index += 1

#   Deal with input spectrum
    if spec == 'Emission':
        lam_obs = side_container.number_input('Line Wavelength (nm)', min_value=lmin, value=lmean, max_value=lmax)
        eside_col1, eside_col2 = side_container.beta_columns([3, 1])
        emis_unit = eside_col2.radio('Width Unit:',['km/s', 'nm'])
        if emis_unit == 'km/s':
            line_width = eside_col1.number_input('Line Width (km/s):', min_value=1., value=200.)
        elif emis_unit == 'nm':
            line_width = eside_col1.number_input('Line Width (nm):', min_value=1e-6, value=1.)
            line_width = line_width * c_km / lam_obs
        lam_obs = lam_obs * 1e-3
    else:
        line_width = 200.
        lam_obs = lmean*1e-3
    if spec == 'Phoenix Stellar Library Spectrum':
        pcol1, pcol2 = side_container.beta_columns(2)
        teff = pcol1.selectbox('Teff (K):', [str(int(i)) for i in np.append(np.arange(23,70)*100, np.arange(35,61)*200)])
        logg = pcol1.selectbox('log(g):', [str(i)[0:3] for i in np.arange(0,13)/2.])
        feh = pcol2.selectbox('Fe/H:', ['-4.0', '-3.0', '-2.0', '-1.5', '-1.0', '-0.5', '-0.0', '+0.5', '+1.0'], index=6)
        aom = pcol2.selectbox('alpha/M:', ['-0.2', '0.0', '+0.2', '+0.4', '+0.6', '+0.8', '+1.0', '+1.2'], index=1)
        if aom == '0.0':
            specfile = 'lte0' + teff + '-' + logg +'0' + feh + '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        else:
            specfile = 'lte0' + teff + '-' + logg +'0' + feh + '.Alpha=' + aom + \
                       '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        specdir = simdir + 'model_spectra/phoenix/' + specfile
        if os.path.isfile(specdir):
            side_container.write('Using file: ' + specfile)
            wav = fits.open(simdir + 'model_spectra/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data / 10.
            e_phot = (h * c) / (wav * 10. * Ang)
            inspec = fits.open(specdir)[0].data / e_phot
            inspec = inspec[np.argmin(np.abs(wav - lmin)):np.argmin(np.abs(wav - lmax)) + 1]
            wav = wav[np.argmin(np.abs(wav - lmin)):np.argmin(np.abs(wav - lmax)) + 1]
            specfig = go.Figure()
            specfig.add_trace(go.Scatter(x=wav/1e3, y=inspec))
            specfig.update_layout(yaxis_title='Flux erg/s/cm^2/cm', xaxis_title='Wavelength (um)',
                                  title='Phoenix Spectrum Curve')
            col2.plotly_chart(specfig)
            spec_func = interpolate.interp1d(wav / 1e3, inspec)
            spec_temp = flux_phot * spec_func(wave) / np.trapz(spec_func(wave), x=wave)
            specinput = [np.array(wave), np.array(spec_temp)]
        else:
            side_container.write('Unable to find file '+ specfile + ', please make another selection.')
    elif spec == 'Black Body':
        temp = side_container.number_input('Source Temperature (K):', min_value=1., value=6000., max_value=40000.)
        spec_temp = (2. * (6.626e-34) * ((3e8) ** 2.)) / (((wave * 1e-6) ** 5.) * (
                np.exp((((6.626e-34) * (3e8)) / ((wave * 1e-6) * temp * 1.380 * 1e-23))) - 1.))
        spec_temp = flux_phot * spec_temp / np.trapz(spec_temp, x=wave)
        specinput = [np.array(wave), np.array(spec_temp)]
    elif spec == 'Stellar Population Spectra - Maraston & Stromback (2011)':
        pcol1, pcol2 = side_container.beta_columns(2)
        speclib = pcol1.selectbox('Stellar Population Library:', ['MILES', 'STELIB', 'ELODIE',
                                                                    'MARCS', 'LRG'])
        if speclib == 'LRG':
            specfile = simdir + 'model_spectra/SSP_M11/CSP_LRG_highres/csp_LRG_M09_STELIB_UVextended.ss'
        else:
            specfiles = glob.glob(simdir + 'model_spectra/SSP_M11/*' + speclib + '*/*', recursive = True)
            specimfs_n = []
            specz_n = []
            for specfile in specfiles:
                if '.ss' in specfile:
                    specimfs_n.append('Sapeter IMF')
                if '.kr' in specfile:
                    specimfs_n.append('Kroupa IMF')
                if '.cha' in specfile:
                    specimfs_n.append('Chabrier IMF')
            specimf_n = pcol2.selectbox('Initital Mass Function:', specimfs_n)
            if specimf_n == 'Sapeter IMF':
                specimf = '.ss'
            elif specimf_n == 'Kroupa IMF':
                specimf = '.kr'
            elif specimf_n == 'Chabrier IMF':
                specimf = '.cha'
            specfiles = glob.glob(simdir + 'model_spectra/SSP_M11/*' + speclib + '*/*' + specimf + '*', recursive=True)
            for specfile in specfiles:
                if 'z10m4' in specfile:
                    specz_n.append('0.0001')
                if 'z0001' in specfile:
                    specz_n.append('0.001')
                if 'z001' in specfile:
                    specz_n.append('0.01')
                if 'z002' in specfile:
                    specz_n.append('0.02 (solar metallicity)')
                if 'z004' in specfile:
                    specz_n.append('0.04')
            specz_n = side_container.selectbox('Chemical Composition (metallicity in z):', specz_n)
            if specz_n == '0.0001':
                specz = 'z10m4'
            elif specz_n == '0.001':
                specz = 'z0001'
            elif specz_n == '0.01':
                specz = 'z001'
            elif specz_n == '0.02 (solar metallicity)':
                specz = 'z002'
            elif specz_n == '0.04':
                specz = 'z004'
            specfiles = glob.glob(
                simdir + 'model_spectra/SSP_M11/*' + speclib + '*/*' + specimf + '*' + specz + '*')
            if len(specfiles)==0:
                side_container.markdown('Selection not available, defaulting to MARCS Sapeter IMF, z=0.02')
                specfile = simdir + 'model_spectra/SSP_M11/SSP_M11_MARCS/ssp_M11_MARCS.ssz002'
            elif len(specfiles) > 1:
                maxwvl = []
                for specfile in specfiles:
                    dat = np.genfromtxt(specfile)
                    wvl = dat[:, -2] / 1e4
                    maxwvl.append(np.max(wvl))
                ind = np.argmax(np.array(maxwvl))
                specfile = specfiles[ind]
            else:
                specfile = specfiles[0]
        dat = np.genfromtxt(specfile)
        wvl = dat[:, -2]
        e_phot = (h * c) / (wvl * Ang)
        inspec = dat[:, -1] / e_phot #  convert to photons/second from ergs/sec, only conversion that matters
        wvl = wvl / 1e4
        if np.max(wvl) < np.max(wave):
            z_i = (np.max(wave) - np.max(wvl)) / np.max(wvl)
            side_container.markdown('Input wavelength must be redshifted to at least %f in order to encompass bandpass.'
                                    % z_i)
        else:
            z_i = 0.
        z = side_container.number_input('Redshift z:', min_value=z_i, max_value=(np.min(wave) - np.min(wvl)) / np.min(wvl),
                               value=z_i)
        wvl = wvl * (1 + z)
        spec_func = interpolate.interp1d(wvl, inspec)
        spec_temp = flux_phot * spec_func(wave) / np.trapz(spec_func(wave), x=wave)
        specinput = [np.array(wave), np.array(spec_temp)]
    elif spec == 'Upload':
        file = side_container.file_uploader('Upload your own spectrum in compatible text file with columns '
                                            '[microns, flux]:')
        try:
            filespec = np.genfromtxt(file)
            wvl = filespec[:, -2]
            inspec = filespec[:, -1]
            if np.max(wvl) < np.max(wave):
                z_i = (np.max(wave) - np.max(wvl)) / np.max(wvl)
                side_container.markdown('Input wavelength must be redshifted to at least %f in order to encompass bandpass.'
                                         % z_i)
            else:
                z_i = 0.
            z = side_container.number_input('Redshift z:', min_value=z_i, max_value=(np.min(wave) - np.min(wvl)) / np.min(wvl),
                                             value=z_i)
            wvl = wvl * (1 + z)
            spec_func = interpolate.interp1d(wvl, inspec)
            spec_temp = flux_phot * spec_func(wave) / np.trapz(spec_func(wave), x=wave)
            specinput = [np.array(wave), np.array(spec_temp)]
        except:
            side_container.markdown('Please upload your spectrum file.')
            specinput = specinput = [np.array(wave), np.ones((len(wave)))]
    elif spec == 'Custom':
        specinput = [np.array(wave), np.zeros((dxspectrum))]
        page_container.subheader('Input Spectrum Generator')
        specol1, specol2, specol3, specol4 = page_container.beta_columns(4)

        def add_spec(spec1, k, filt='K', wave=None, scale=None, fint=None, flambda=None, mag=None, source=None,
             simdir='~/Liger/sim', lam_obs=None, line_width=None, mode='ifs'):
            sfl1 = specol2.selectbox('Input Flux Method:', ['Magnitude', 'Flux Density',
                                                            'Integrated Flux over ''Bandpass'],
                                     key='sfl' + str(k))
            if sfl1 == 'Magnitude':
                mag = specol3.number_input('Magnitude: ', value=20., key='mag' + str(k))
                veg = specol4.radio('Magnitude Standard:', ('Vega', 'AB'), key='veg' + str(k))
                if veg == 'AB':
                    # ETC only takes vega input so this converts it from AB to vega, kinda silly.
                    mag = ab2vega(mag, lmean=lmean)
                fint = None
                flambda = None
            elif sfl1 == 'Flux Density':
                flambda = specol3.number_input('Ergs/s/cm^2/Angstrom * 10^-19: ', value=1.62, key='flux' + str(k))
                flambda = flambda * 1e-19
                specol4.markdown('#')
                specol4.markdown('##')
                specol4.markdown(' ')
                fint = None
                mag = None
            elif sfl1 == 'Integrated Flux over Bandpass':
                fint = specol3.number_input('Ergs/s/cm^2 * 10^-17: ', value=4., key='intflux' + str(k))
                specol4.markdown('#')
                specol4.markdown('##')
                specol4.markdown(' ')
                fint = fint * 1e-17
                flambda = None
                mag = None

            if mag is not None:
                # convert to flux density (flambda)
                flux_phot = zp * 10 ** (-0.4 * mag)  # photons/s/m^2
                if etc_source == 'extended':
                    flux_phot = flux_phot * (etc_scale * 2)
            elif flambda is not None:
                fnu = flambda / (Ang / ((lmean * 1e-9) ** 2 / c))
                ABmag = -2.5 * log10(fnu) - 48.60
                mag = ab2vega(ABmag, lmean=lmean)
                flux_phot = zp * 10 ** (-0.4 * mag)  # photons/s/m^2
                if etc_source == 'extended':
                    flux_phot = flux_phot * (etc_scale * 2)
            elif fint is not None:
                # Compute flux_phot from flux
                E_phot = (h * c) / (lmean * 10. * Ang)
                flux_phot = 1e4 * fint / E_phot
                if etc_source == 'extended':
                    flux_phot = flux_phot * (etc_scale * 2)

            if spec1 == 'Emission':
                lam_obs = specol1.number_input('Line Wavelength (nm)', min_value=lmin, value=lmean,
                                               max_value=lmax, key='lamobs' + str(k))
                emis_unit = specol3.radio('Width Unit:', ['km/s', 'nm'])
                if emis_unit == 'km/s':
                    line_width = specol2.number_input('Line Width (km/s):', min_value=1., value=200.,
                                                  key='lw' + str(k))
                elif emis_unit == 'nm':
                    line_width = specol2.number_input('Line Width (nm):', min_value=1e-6, value=1.,
                                                  key='lw' + str(k))
                    line_width = line_width * c_km / lam_obs

                specol3.markdown('#')
                specol3.markdown('##')
                specol3.markdown(' ')
                specol4.markdown('#')
                specol4.markdown('##')
                specol4.markdown(' ')
            if spec1 == 'Phoenix Stellar Library Spectrum':
                teff = specol1.selectbox('Teff (K):', [str(int(i)) for i in
                                                       np.append(np.arange(23, 70) * 100, np.arange(35, 61) * 200)],
                                         key='teff' + str(k))
                logg = specol2.selectbox('log(g):', [str(i)[0:3] for i in np.arange(0, 13) / 2.], key='logg' + str(k))
                feh = specol3.selectbox('Fe/H:',
                                        ['-4.0', '-3.0', '-2.0', '-1.5', '-1.0', '-0.5', '-0.0', '+0.5', '+1.0'],
                                        index=6, key='feh' + str(k))
                aom = specol4.selectbox('alpha/M:', ['-0.2', '0.0', '+0.2', '+0.4', '+0.6', '+0.8', '+1.0', '+1.2'],
                                        index=1, key='aom' + str(k))
            else:
                aom = None
                teff = None
                logg = None
                feh = None

            if spec1 == 'Black Body':
                temp = specol1.number_input('Source Temperature (K):', min_value=1., value=6000.,
                                            key='blackbody' + str(k))
                specol2.markdown('#')
                specol2.markdown('##')
                specol2.markdown(' ')
                specol3.markdown('#')
                specol3.markdown('##')
                specol3.markdown(' ')
                specol4.markdown('#')
                specol4.markdown('##')
                specol4.markdown(' ')
            else:
                temp=None

            if spec1 == 'Stellar Population Spectra - Maraston & Stromback (2011)':
                speclib = specol1.selectbox('Stellar Population Library:', ['MILES', 'STELIB', 'ELODIE',
                                                                            'MARCS', 'LRG'],
                                      key='ssp_'+str(k))
                if speclib == 'LRG':
                    specfile = simdir + 'model_spectra/SSP_M11/CSP_LRG_highres/csp_LRG_M09_STELIB_UVextended.ss'
                else:
                    specfiles = glob.glob(simdir + 'model_spectra/SSP_M11/*' + speclib + '*/*', recursive=True)
                    specimfs_n = []
                    specz_n = []
                    for specfile in specfiles:
                        if '.ss' in specfile:
                            specimfs_n.append('Sapeter IMF')
                        if '.kr' in specfile:
                            specimfs_n.append('Kroupa IMF')
                        if '.cha' in specfile:
                            specimfs_n.append('Chabrier IMF')
                    specimf_n = specol2.selectbox('Initital Mass Function:', specimfs_n, key='specimfs'+str(k))
                    if specimf_n == 'Sapeter IMF':
                        specimf = '.ss'
                    elif specimf_n == 'Kroupa IMF':
                        specimf = '.kr'
                    elif specimf_n == 'Chabrier IMF':
                        specimf = '.cha'
                    specfiles = glob.glob(simdir + 'model_spectra/SSP_M11/*' + speclib + '*/*' + specimf + '*',
                                          recursive=True)
                    for specfile in specfiles:
                        if 'z10m4' in specfile:
                            specz_n.append('0.0001')
                        if 'z0001' in specfile:
                            specz_n.append('0.001')
                        if 'z001' in specfile:
                            specz_n.append('0.01')
                        if 'z002' in specfile:
                            specz_n.append('0.02 (solar metallicity)')
                        if 'z004' in specfile:
                            specz_n.append('0.04')
                    specz_n = specol3.selectbox('Chemical Composition (metallicity in z):', specz_n, key='specz'+str(k))
                    if specz_n == '0.0001':
                        specz = 'z10m4'
                    elif specz_n == '0.001':
                        specz = 'z0001'
                    elif specz_n == '0.01':
                        specz = 'z001'
                    elif specz_n == '0.02 (solar metallicity)':
                        specz = 'z002'
                    elif specz_n == '0.04':
                        specz = 'z004'
                    specfiles = glob.glob(
                        simdir + 'model_spectra/SSP_M11/*' + speclib + '*/*' + specimf + '*' + specz + '*')
                    if len(specfiles)==0:
                        specol1.markdown('Selection not available, defaulting to MARCS Sapeter IMF, z=0.02')
                        specfile = simdir + 'model_spectra/SSP_M11/SSP_M11_MARCS/ssp_M11_MARCS.ssz002'
                    elif len(specfiles) > 1:
                        maxwvl = []
                        for specfile in specfiles:
                            dat = np.genfromtxt(specfile)
                            wvl = dat[:, -2] / 1e4
                            maxwvl.append(np.max(wvl))
                        ind = np.argmax(np.array(maxwvl))
                        specfile = specfiles[ind]
                    else:
                        specfile = specfiles[0]
                dat = np.genfromtxt(specfile)
                wvl = dat[:, -2]
                e_phot = (h * c) / (wvl * Ang)
                inspec = dat[:, -1] / e_phot  # convert to photons/second from ergs/sec, only conversion that matters
                wvl = wvl / 1e4
                if np.max(wvl) < np.max(wave):
                    z_i = (np.max(wave) - np.max(wvl)) / np.max(wvl)
                else:
                    z_i = 0.
                z = specol4.number_input('Redshift z:', min_value=z_i,
                                         max_value=(np.min(wave) - np.min(wvl)) / np.min(wvl), value=z_i, key='zi'+str(k))
                wvl = wvl * (1 + z)
                spec_func = interpolate.interp1d(wvl, inspec)
                spec_temp = flux_phot * spec_func(wave) / np.trapz(spec_func(wave), x=wave)
                specname = np.array(spec_temp)
            else:
                specname = None

            if spec1 == 'Upload':
                file = specol1.file_uploader('Upload your own spectrum in compatible text file with columns '
                                             '[microns, flux]:', key='file_up'+str(k))
                try:
                    filespec = np.genfromtxt(file)
                    wvl = filespec[:, -2]
                    inspec = filespec[:, -1]
                    if np.max(wvl) < np.max(wave):
                        z_i = (np.max(wave) - np.max(wvl)) / np.max(wvl)
                    else:
                        z_i = 0.
                    z = specol2.number_input('Redshift z:', min_value=z_i,
                                                    max_value=(np.min(wave) - np.min(wvl)) / np.min(wvl),
                                                    value=z_i, key='upz'+str(k))
                    wvl = wvl * (1 + z)
                    spec_func = interpolate.interp1d(wvl, inspec)
                    spec_temp = flux_phot * spec_func(wave) / np.trapz(spec_func(wave), x=wave)
                    specname = np.array(spec_temp)
                except:
                    specol1.markdown('Please upload your spectrum file.')
                    specname = np.ones((len(wave)))
                specol2.markdown('#')
                specol2.markdown(' ')
                specol2.markdown('##')
                specol2.markdown(' ')
                specol3.markdown('#')
                specol3.markdown(' ')
                specol3.markdown('##')
                specol3.markdown(' ')
                specol4.markdown('#')
                specol4.markdown(' ')
                specol4.markdown('##')
                specol4.markdown(' ')
                specol3.markdown('#')
                specol3.markdown('##')
                specol3.markdown(' ')
                specol4.markdown('#')
                specol4.markdown('##')
                specol4.markdown(' ')
            spec_temp = gen_spec(spec1, wave=wave, scale=scale, fint=fint, flambda=flambda, mag=mag,
                                 lam_obs=lam_obs, line_width=line_width, source=source, simdir=simdir,
                                 filt=filt, mode=mode, teff=teff, logg=logg, feh=feh, aom=aom, temp=temp,
                                 specname=specname)
            return spec_temp

        def custom_spec():
            k = 1
            spec1 = specol1.selectbox('Add Spectrum:', ['Vega', 'Flat', 'Emission',
                                                        'Phoenix Stellar Library Spectrum', 'Black Body',
                                                        'Stellar Population Spectra - Maraston & Stromback '
                                                        '(2011)', 'Upload'],
                                      key='specgen'+str(k)) #
            if spec1 != 'None':
                spec_temp = add_spec(spec1, k, wave=wave, scale=etc_scale, fint=fint, flambda=flambda, mag=mag,
                                     lam_obs=lam_obs, line_width=line_width, source=source, simdir=simdir,
                                     filt=filt, mode=mode)
                if spec_temp == '':
                    specol1.write('ERROR: Unable to find file, please make another selection.')
                    return
                specinput[1] = specinput[1] + spec_temp
                k = 2
                spec2 = specol1.selectbox('Add Spectrum:', ['None', 'Vega', 'Flat', 'Emission',
                                                           'Phoenix Stellar Library Spectrum', 'Black Body',
                                                            'Stellar Population Spectra - Maraston & '
                                                            'Stromback (2011)', 'Upload'],
                                         key='specgen' + str(k)) #
                if spec2 != 'None':
                    spec_temp = add_spec(spec2, k, wave=wave, scale=etc_scale, fint=fint, flambda=flambda, mag=mag,
                                         lam_obs=lam_obs, line_width=line_width, source=source, simdir=simdir,
                                         filt=filt, mode=mode)
                    if spec_temp == '':
                        specol1.write('ERROR: Unable to find file, please make another selection.')
                        return specinput
                    specinput[1] = specinput[1] + spec_temp
                    k = 3
                    spec3 = specol1.selectbox('Add Spectrum:', ['None', 'Vega', 'Flat', 'Emission',
                                                                'Phoenix Stellar Library Spectrum', 'Black Body',
                                                                'Stellar Population Spectra - Maraston & '
                                                                'Stromback (2011)', 'Upload'],
                                             key='specgen' + str(k)) #
                    if spec3 != 'None':
                        spec_temp = add_spec(spec3, k, wave=wave, scale=etc_scale, fint=fint, flambda=flambda, mag=mag,
                                             lam_obs=lam_obs, line_width=line_width, source=source, simdir=simdir,
                                             filt=filt, mode=mode)
                        if spec_temp == '':
                            specol1.write('ERROR: Unable to find file, please make another selection.')
                            return specinput
                        specinput[1] = specinput[1] + spec_temp
                        k = 4
                        spec4 = specol1.selectbox('Add Spectrum:', ['None', 'Vega', 'Flat', 'Emission',
                                                                    'Phoenix Stellar Library Spectrum', 'Black Body',
                                                                    'Stellar Population Spectra - Maraston & '
                                                                    'Stromback (2011)', 'Upload'],
                                                 key='specgen' + str(k)) #

                        if spec4 != 'None':
                            spec_temp = add_spec(spec4, k, wave=wave, scale=etc_scale, fint=fint, flambda=flambda, mag=mag,
                                                 lam_obs=lam_obs, line_width=line_width, source=source, simdir=simdir,
                                                 filt=filt, mode=mode)
                            if spec_temp == '':
                                specol1.write('ERROR: Unable to find file, please make another selection.')
                                return specinput
                            specinput[1] = specinput[1] + spec_temp
                            k = 5
                            spec5 = specol1.selectbox('Add Spectrum:', ['None', 'Vega', 'Flat', 'Emission',
                                                                        'Phoenix Stellar Library Spectrum',
                                                                        'Black Body',
                                                                        'Simple Stellar Population Spectra - '
                                                                        'Maraston & Stromback (2011)', 'Upload'],
                                                     key='specgen' + str(k)) #
                            if spec5 != 'None':
                                spec_temp = add_spec(spec5, k, wave=wave, scale=etc_scale, fint=fint, flambda=flambda,
                                                     mag=mag, lam_obs=lam_obs, line_width=line_width, source=source,
                                                     simdir=simdir, filt=filt, mode=mode)
                                if spec_temp == '':
                                    specol1.write('ERROR: Unable to find file, please make another selection.')
                                    return specinput
                                specinput[1] = specinput[1] + spec_temp
                                k = 6
                                spec6 = specol1.selectbox('Add Spectrum:', ['None', 'Vega', 'Flat', 'Emission',
                                                                            'Phoenix Stellar Library Spectrum',
                                                                            'Black Body',
                                                                            'Simple Stellar Population Spectra - '
                                                                            'Maraston & Stromback (2011)', 'Upload'],
                                                          key='specgen' + str(k))
                                if spec6 != 'None':
                                    spec_temp = add_spec(spec6, k, wave=wave, scale=etc_scale, fint=fint, flambda=flambda,
                                                         mag=mag,
                                                         lam_obs=lam_obs, line_width=line_width, source=source,
                                                         simdir=simdir,
                                                         filt=filt, mode=mode)
                                    if spec_temp == '':
                                        specol1.write('ERROR: Unable to find file, please make another selection.')
                                        return specinput
                                    specinput[1] = specinput[1] + spec_temp
                                    k=7
                                    spec7 = specol1.selectbox('Add Spectrum:', ['None', 'Vega', 'Flat', 'Emission',
                                                                                'Phoenix Stellar Library Spectrum',
                                                                                'Black Body',
                                                                                'Simple Stellar Population Spectra -'
                                                                                ' Maraston & Stromback (2011)',
                                                                                'Upload'],
                                                              key='specgen' + str(k))
                                    if spec7 != 'None':
                                        spec_temp = add_spec(spec7, k, wave=wave, scale=etc_scale, fint=fint,
                                                             flambda=flambda,
                                                             mag=mag,
                                                             lam_obs=lam_obs, line_width=line_width, source=source,
                                                             simdir=simdir,
                                                             filt=filt, mode=mode)
                                        if spec_temp == '':
                                            specol1.write('ERROR: Unable to find file, please make another selection.')
                                            return specinput
                                        specinput[1] = specinput[1] + spec_temp
            return specinput
        specinput = custom_spec()
        if specinput is None: specinput = [np.array(wave), np.ones((len(wave)))]
        specfig = go.Figure()
        specfig.add_trace(go.Scatter(x=specinput[0], y=specinput[1]))
        specfig.update_layout(yaxis_title='Photons/s/um/m2', xaxis_title='Wavelength (um)',
                              title='Custom Spectrum Curve')
        specol1.plotly_chart(specfig)
        csv_outspec = specol1.button('Download My Input Spectrum as CSV')
        if csv_outspec:
            spec_csv = np.array([wave, specinput[1]]).T
            specdf = pd.DataFrame(spec_csv, columns=['Wavelength(microns)','Flux (Photons/s/um/m2)'])
            tmp_download_link = download_link(specdf, 'my_input_spec.csv', 'Click Here to Download CSV: ' +
                                          'my_input_spec.csv')
            specol1.markdown(tmp_download_link, unsafe_allow_html=True)

    else:
        specinput = None
    ##input PSF configuration
    with page_container.beta_container():
        psf_col1, psf_col2 = page_container.beta_columns(2)
        psf_col1.subheader('Point Spread Function (PSF) Configuration:')
        psfmode = psf_col1.selectbox('Select PSF Option:', ['Generated Analytic PSF','Pre-generated LTAO PSF'])
        if psfmode == 'Pre-generated LTAO PSF':
            if filt[0] == 'Z': psf_col1.markdown('WARNING: selected filter not available, PSF will mis-match simulated '
                                                 'wavelengths. PSF wavelength will default to 1.02 microns.')
            x_arr = y_arr = [0, -5, 5, -10, 10, -15, 15]
            loclist = list()
            xarrs = list()
            yarrs = list()
            for i in range(0, len(x_arr)):
                for j in range(0, len(y_arr)):
                    loclist.append([str(x_arr[i]) + '"', str(y_arr[j]) + '"'])
                    xarrs.append(x_arr[i])
                    yarrs.append(y_arr[j])
            psflocs = psf_col1.selectbox('PSF Focal Plane Location: ', loclist)
            xloc = psflocs[0][0:-1]
            yloc = psflocs[1][0:-1]
            psf_loc_grid = go.Figure(data=go.Scatter(x=xarrs, y=yarrs, mode='markers',
                                                     name='PSF Focal Plane Locations'))
            psf_loc_grid.add_trace(go.Scatter(x=[xloc], y=[yloc], mode='markers',
                                              marker=dict(size=20), name='Selected Location'))
            psf_loc_grid.update_layout(xaxis_title="Arcseconds", yaxis_title="Arcseconds",
                                       title="Location on Focal Plane")
            psf_col2.plotly_chart(psf_loc_grid)
            psf_input = None
            psf_loc = [float(xloc), float(yloc)]
        elif psfmode == 'Generated Analytic PSF':
            with st.spinner('Generating PSF....'):
                strehl = psf_col1.number_input('Strehl Ratio:', value = 0.5)
                fried = psf_col1.number_input('Fried Parameter r0 (cm):', value = 20.)
                lam_obs_psf = [lmin*10., lmean*10., lmax*10.]
                wav = psf_col1.radio('PSF Wavelength:',['Monochromatic (Central Wavelength: '+str(lmean/1e3)+'um)',
                    'Averaged over Bandpass ('+str(lmin/1e3)+'um, '+str(lmean/1e3)+'um, '+str(lmax/1e3)+'um)'])
                if wav == 'Monochromatic (Central Wavelength: '+str(lmean/1e3)+'um)':
                    lam_obs_psf = lam_obs_psf[1]
                elif wav == 'Averaged over Bandpass ('+str(lmin/1e3)+'um, '+str(lmean/1e3)+'um, '+str(lmax/1e3)+'um)':
                    lam_obs_psf = lam_obs_psf
                # wfe = col1.number_input('Additional Input Wavefront Error (nm):',value=0.)
                # blurring = col1.checkbox('Include Gaussian Blurring')
                # if blurring is not None:
                #     blur = True

                psf_input = analytic_psf(strehl, lam_obs_psf, etc_scale, fried_parameter=fried, verbose=False, stack=True,
                                         simdir=simdir)
                psf_input = psf_input[-1]/np.sum(psf_input[-1])
                psf_loc = None
                psf_col2.subheader('PSF Preview:')
                s = np.shape(psf_input)
                logscale = lambda a, im: np.log10(a * (im / np.max(im)) + 1.0) / (np.log10(a))
                psf_col2.image(logscale(1000., np.abs(
                    psf_input[int(s[0] / 2 - 20):int(s[0] / 2 + 20), int(s[1] / 2 - 20):int(s[1] / 2 + 20)])),
                               clamp=True, caption='Generated PSF (Central ' + str(40. * etc_scale)[
                                                                               0:4] + ' arcseconds) - Log Scale',
                               width=250)
                psf_col2.markdown('[KAPA Strehl Calculator](http://bhs.astro.berkeley.edu/cgi-bin/kapa_strehl)')
    page_container.write('--------------------------------------------------------------------------')

    #   Define default radius value based on scale and lambda.
    if etc_scale == 0.014:
        radiusl = 1. * lmean*1e2 * 206265 / (1.26e11)
    elif etc_scale == 0.031:
        radiusl = 1.5 * lmean*1e2 * 206265 / (1.26e11)
    elif etc_scale == 0.075:
        radiusl = 20 * lmean*1e2 * 206265 / (1.26e11)
    else:
        radiusl = 40 * lmean*1e2 * 206265 / (1.26e11)

    #   Container for Results and result plots.
    with page_container.beta_container():
        col3, col4 = page_container.beta_columns(2)
        col3.subheader('Liger Calculation Results:')
        #   Calculation
        aperture = col3.slider(label='Aperture Radius (arcseconds):', min_value=0.01,
                               max_value=float(np.min(etc_fov)/2.),
                               value=float(np.min([float(radiusl), np.min(etc_fov)/2.])))
        if aperture/etc_scale < 1: col2.markdown('WARNING: You have selected an aperture radius less than 1 pixel.')
        if col3.button('CLICK HERE to Calculate Liger Results'):
            res, fig, csvarr = LIGER_ETC(filter=filt, mag=mag, flambda=flambda,
                                 fint=fint, itime=itime,nframes=nframes, snr=snr,
                                 aperture=aperture, gain=3.04, readnoise=7., darkcurrent=0.05,
                                 scale=etc_scale, resolution=4000, collarea=78.5, positions=[0, 0],
                                 bgmag=None, efftot=None, mode=mode.lower(), calc=etc_calc,
                                 spectrum=spec, specinput=specinput, lam_obs=lam_obs, line_width=line_width,
                                 png_output=None, source=etc_source, profile=etc_prof, source_size=0.2,
                                 csv_output=True, fov = etc_fov, psf_loc=psf_loc,
                                 psf_time=1.4, verb=1, psf_input=psf_input, simdir=simdir, psfdir=psfdir, test=0)
            #   parse the results into a nice readable format and write them into the two columns
            val_lens = np.array([len(i) for i in res.values()])
            ind = np.where(val_lens > 0)[0]
            vals = list(res.values())
            keys = list(res.keys())
            for i in ind:
                string = str(keys[i]) + ': ' + str(vals[i])
                col3.write(string)
            if (mode == 'Imager' and etc_source == 'extended' and prof_type != 'Sersic Profile'):
                col4.write('The extended-source simulation assumes a perfectly even brightness ' +
                           'distribution, so there is no meaningful simulated image.')
            elif mode == 'Imager':
                col4.subheader('Simulated Image:')
                col4.plotly_chart(fig)
            else:
                col4.subheader('Simulated ' + plot_subhead)
                col4.plotly_chart(fig)
                col4.markdown('**The plot above is interactive!**')
            if mode == 'IFS':
                if etc_calc == 'exptime':
                    header = "Wavelength(microns),Int_Time_PeakFlux(s),Int_Time_MedianFlux(s)," \
                             "Int_Time_MeanFlux(s),Int_Time_Total_Aperture_Flux(s)"
                    csvlabel = 'inputSNR' + str(int(snr))
                elif etc_calc == 'snr':
                    header = "Wavelength(microns),SNR_Peak,SNR_Median,SNR_Mean,SNR_Aperture_Total"
                    csvlabel = 'inputITime' + str(int(itime*nframes))
                df = pd.DataFrame(csvarr, columns=header.split(','))
                if fl == 'Magnitude':
                    csv_mag = veg + 'Mag' + str(int(mag))
                elif fl == 'Flux Density':
                    csv_mag = 'Flambda' + str(flambda*1e19)[0:4] + 'e19'
                elif fl == 'Integrated Flux over Bandpass':
                    csv_mag = 'Fint' + str(fint * 1e17)[0:4] + 'e17'
                csv_title = 'Liger_' + mode + '_ETC_' + etc_calc + '_' + filt + '_' + scale + \
                            '_' + csvlabel + '_' + etc_source + '_' + csv_mag
                tmp_download_link = download_link(df, csv_title + '.csv',
                                                  'Click Here to Download CSV: ' +
                                                  csv_title + '.csv')
                col4.markdown(tmp_download_link, unsafe_allow_html=True)
