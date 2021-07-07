from __future__ import print_function
import streamlit as st
import pandas as pd
from bug_reporter import export_bug_data
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
simdir = config.get('CONFIG', 'simdir')
psfdir = config.get('CONFIG', 'psfdir')
#   Collects the bug information to be exported to the google sheet with the google bug rep

def bugpage(page_container=None):
    if page_container is None:
        pc = st.form("bug_form")
    else:
        pc = page_container.form("bug_form")
    pc.markdown('Thank you for helping develop the OSIRIS ETC by submitting your bug! '
            'Please use the following submisison processes to record your problem.')
    bugtype = pc.selectbox('What best describes your bug?', ['Crashed Code', 'Incorrect Results', 'General Usability',
                                                             'Other'])
    bugtext = pc.text_input('Please describe your bug as succinctly as possible and please copy/paste the error message'
                            ' if any:')
    expectext = pc.text_input('What is the expected behavior which you are not experiencing with this bug?')
    actualtext = pc.text_input('What is the actual behavior you are experiencing because of this bug?')
    if bugtype != 'General Usability':
        if bugtype == 'Incorrect Results':
            expanded = True
        else:
            expanded = False
        pc.markdown('If your bug is dependent on specific ETC input values, please enter them using the dropdown '
                    'below.')
        with pc.beta_expander('Fill out ETC values', expanded=expanded):
            st.markdown('Please make the ETC box selections with which your bug occurred.')
            bugmode = st.radio("Please Select OSIRIS Mode", ['IFS', 'Imager'], key='bugmode')
            bugcalc = st.radio('calculate: ', ['Signal-to-Noise Ratio (SNR)', 'Exposure Time'], key='bugcalc')
            if bugcalc == 'Signal-to-Noise Ratio (SNR)':
                itime = st.number_input('Frame Integration Time (Seconds): ', min_value=0., value=30.)
                nframes = st.number_input('Number of Frames: ', min_value=0, value = 1)
                st.markdown('Total Integration Time: ' + str(itime*nframes))
                bugsnr = str(itime*nframes)
            elif bugcalc == 'Exposure Time':
                bugsnr = st.number_input('SNR: ', min_value=0., value=5.)

            bugfl = st.selectbox('Input Flux Method:', ['Magnitude', 'Flux Density', 'Integrated Flux over Bandpass'],
                                 key='bugfl')
            if bugfl == 'Magnitude':
                side_col1, side_col2 = st.beta_columns([3, 1])
                bugmag = side_col1.number_input('Magnitude: ', value=20.)
                veg = side_col2.radio('Magnitude Standard:', ('Vega', 'AB'), key='bugveg')
            elif bugfl == 'Flux Density':
                flambda = st.number_input('Ergs/s/cm^2/Angstrom * 10^-19: ', value=1.62)
                flambda = flambda * 1e-19
                bugmag = flambda
            elif bugfl == 'Integrated Flux over Bandpass':
                fint = st.number_input('Ergs/s/cm^2 * 10^-17: ', value=4.)
                fint = fint * 1e-17
                bugmag = fint
            #   select the type of source
            bugsource = st.radio('Source Type:', ('Point Source', 'Extended (Mag/Sq. Arcsec.)'), key='bugsource')
            if bugsource == 'Extended (Mag/Sq. Arcsec.)':
                prof_type = st.radio('Extended Brightness Profile:', ('Top-Hat', 'Sersic Profile'))
                bugsource = bugsource + '_' + prof_type
            bugspec = st.selectbox('Spectrum Shape:', ['Vega', 'Flat', 'Emission''Phoenix Stellar Library Spectrum',
                                                       'Black Body', 'Stellar Population Spectra - Maraston & Stromback (2011)',
                                                       'Custom', 'Upload'],
                                   key='bugspec')
            bugscale = st.select_slider('Plate Scale ( Arcseconds per Spatial Pixel )', ['20','35','50','100'], key='bugscale')
            bugfilt = st.selectbox("Filter: ",
                                  ['Zbb', 'Jbb', 'Hbb', 'Kbb', 'Kcb', 'Zn3', 'Zn4', 'Jn1', 'Jn2', 'Jn3', 'Jn4', 'Hn1', 'Hn2',
                                   'Hn3', 'Hn4', 'Hn5', 'Kn1', 'Kn2', 'Kn3', 'Kc3', 'Kn4', 'Kc4', 'Kn5', 'Kc5',
                                   'FeII', 'Hcont', 'Y', 'J', 'Kp', 'BrGamma', 'Kcont', 'HeI_B', 'Kcb'], key='bugfilt')
            if bugfl == 'Magnitude':
                if veg == 'AB':
                    import numpy as np
                    from misc_funcs import ab2vega
                    modes = pd.read_csv(simdir + 'info/osiris_modes.csv', header=0)
                    filts = np.array([fil.lower() for fil in modes['Filter']])
                    wavs = np.where(filts == bugfilt.lower())[0]
                    lmin = modes['λ (nm) min'][wavs]
                    lmax = modes['λ (nm) max'][wavs]
                    lmin = lmin.values
                    lmax = lmax.values
                    if not isinstance(lmin, str): lmin = lmin[0]
                    if not isinstance(lmax, str): lmax = lmax[0]
                    if '*' in lmin: lmin = lmin.replace('*', '')
                    if '*' in lmax: lmax = lmax.replace('*', '')
                    lambdamin = float(lmin) * 10.
                    lambdamax = float(lmax) * 10.
                    lmean = np.mean([lambdamin, lambdamax])
                    bugmag = ab2vega(bugmag, lmean / 10.)
            bugpsfmode = st.selectbox('Select PSF Option:', ['Generated Analytic PSF', 'Pre-generated SCAO PSF'],
                                      key='bugpsfmode')
            if bugpsfmode == 'Pre-generated SCAO PSF':
                bugstrehl = 'N/A'
                bugfried = 'N/A'
            elif bugpsfmode == 'Generated Analytic PSF':
                    bugstrehl = st.number_input('Strehl Ratio:', value=0.5, key='bugstrehl')
                    bugfried = st.number_input('Fried Parameter r0 (cm):', value=20., key='bugfried')
            bugaperture = st.slider(label='Aperture Radius (arcseconds):', min_value=0.01, max_value=10.1,
                                    key='bugaperture')
    else:
        bugmode = ''
        bugcalc = ''
        bugsnr = ''
        bugsource = ''
        bugfl = ''
        bugmag = ''
        bugspec = ''
        bugfilt = ''
        bugscale = ''
        bugpsfmode = ''
        bugstrehl = ''
        bugfried = ''
        bugaperture = ''
    columns = ['Bug Type', 'Bug Description', 'Expected Behavior', 'Actual Behavior', 'Mode', 'Calculation',
                        'Integration Time/SNR', 'Source Type','Input Flux Type', 'Input Flux', 'Spectrum', 'Filter',
                        'Scale', 'PSF type', 'PSF Strehl', 'Fried Parameter', 'Aperture']

    bug_df = pd.DataFrame(columns=columns, dtype=str)
    bug_df.loc[0] = [bugtype, bugtext, expectext, actualtext, bugmode, bugcalc, bugsnr, bugsource, bugfl, str(bugmag),
                     bugspec, bugfilt,bugscale, bugpsfmode, bugstrehl, bugfried, bugaperture]
    pc.markdown('#')
    bug_submitted = pc.form_submit_button('Submit Bug')
    if bug_submitted:
        export_bug_data(bug_df)
        pc.markdown('Your bug has been submitted! Thank you for your help.')
        st.balloons()
