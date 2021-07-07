# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np

def df():
    modes = pd.read_csv('osiris_modes.csv', header=0)
    filts = modes['Filter']
    scales = modes.columns[-4:]
    scale = scales[0:2].values
    filt = filts[0:2].values
    fovs = modes[[s for s in scale]].iloc[([np.where(modes['Filter'] == f)[0][0] for f in filt])].values
    print(scales.values)
    print(scales.values.flatten())



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

from etc_analytic_psf.etc_analytic_psf import analytic_psf
import matplotlib.pyplot as plt
import time
from get_filterdat import get_filterdat
def test_psf():
    strehl = 0.6
    filt = 'Zbb'
    scale = 0.02
    fried = 20
    simdir = '/Volumes/Backup Plus/liger/sim/'
    print(simdir)
    filterdat = get_filterdat(filt, simdir)
    wav = filterdat["lambdac"]
    start = time.time()
    print('starting time')
    psf_input = analytic_psf(strehl, wav, scale, fried_parameter=fried, verbose=False, stack=True,
                             simdir=simdir)
    end = time.time()
    print('ending time:')
    print(end - start)
    psf_input = psf_input[-1]
    plt.imshow(psf_input)
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
