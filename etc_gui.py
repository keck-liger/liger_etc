########################################################################################################################
#
#   etc_gui.py written by Nils Rundquist for the Liger ETC under development by OIRLab (UCSD)
#   Date - 06/29/2021
#
#   Required packages: numpy, matplotlib, astropy and pandas.
#   Required files: kp_dig.fits, info/filter_info.dat
#
#   This is the primary code to be ran by streamlit for interfacing with the Liger
#   Exposure Time Calculator Code.  Refer to pages.gui_main.py for where the actual
#   ETC code is called. Refer to the readme or help page on the gui for information.
#
########################################################################################################################

import streamlit as st
#   Import the different pages we use.
import pages.gui_main
import pages.gui_help
from pages.gui_bug import bugpage
#   Streamlit page configuration
st.set_page_config(
    page_title='Keck Liger Exposure Time Calculator',
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="Logo_name.png"
)

title_container = st.beta_container()

#   Bottom of page setup.  Here we will include the buttons and about information.
with st.beta_container():
    col5, col6, col7, col8, col9, col10, col11 = st.beta_columns(7)
    col10.image('liger_logo.png', use_column_width='auto')
    col11.image('keck_obs_logo.png', use_column_width='auto')
    col5.subheader('Page Navigation:')
    page_ref = col5.radio('Current Page:', ['Main ETC GUI', 'Help Page', 'Submit a Bug'])

#   Because of the way Streamlit works, if we want to allow navigation to different pages with the click of a button
#   at the bottom of the page, we must first designate the different parts of the page where things are allocated.
#   These variables will be passed to the different page functions so that they can insert elements above the 'About'
#   section where the bug submission page and help page buttons will be.
page_main = st.beta_container()
side_main = st.sidebar.beta_container()
if page_ref == 'Help Page':
    title_container.title('Help With Liger ETC')
    pages.gui_help.helppage(page_container=page_main)
elif page_ref == 'Submit a Bug':
    title_container.title('Submit Bug for Liger ETC')
    bugpage(page_container=page_main)
else:
    title_container.title('W.M. Keck Observatory - Liger Exposure Time Calculator')
    pages.gui_main.exec_gui(page_container=page_main, side_container=side_main)

with st.beta_container():
    col5, col6, col7, col8, col9, col10, col11 = st.beta_columns(7)
    col11.image('Long-Name.png', use_column_width='auto')
    col10.image('mulab_logo_short.png', use_column_width='auto')
    col9.image('nasa_logo.png', use_column_width='auto')
    col8.image('caltech_logo.png', use_column_width='auto')
    col7.image('uc_logo.png', use_column_width='auto')


page_main.subheader('About')
page_main.info('''
The Liger ETC was developed by the UCSD Optical InfraRed Laboratory in collaboration with the Keck All Sky Precision 
Adaptive Optics (KAPA) Science Tools Team at UC Berkeley and others: Nils Rundquist (UCSD), Arun Surya (UCSD), Sanchit 
Sabhlok (UCSD), Gregory Walth (Carnegie), Shelley Wright (UCSD), Jessica Lu (UC Berkeley), et al. Graphical User 
Interface was built using Streamlit. This code development and webpage is funded by the Gordon and Betty Moore 
Foundation through Grant GBMF8549 to Shelley Wright.
''')
