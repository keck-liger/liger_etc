import numpy as np

def get_liger_psf(mode, filter, psf_loc):
    x, y = float(psf_loc[0]), float(psf_loc[1])
    if mode.lower() == "ifs":
        scale_arr = [0.02, 0.035, 0.05, 0.1] # plate scale [mas]
        x_arr = y_arr = [0] # position on focal plane [arcsec]
        ins = "ifu"
    elif mode.lower() == "imager":
        scale_arr = [0.02] # plate scale [mas]
        x_arr = y_arr = [0, -5, 5, -10, 10, -15, 15] # position on focal plane [arcsec]
        ins = "im"
    # select the closest file, in case the user guesses wrong
    x_ind = np.argmin(np.abs(np.array(x_arr) - x))
    y_ind = np.argmin(np.abs(np.array(y_arr) - y))
    x_s = x_arr[x_ind]
    y_s = y_arr[y_ind]
    dir = 'PSFs_LTAO_11_09_19/ltao_7x7_YJHK/'
    if 'Z' in filter:
        dir = dir + 'ltao_7_7_hy/'
        exten_no = 1
    elif 'Y' in filter:
        dir = dir + 'ltao_7_7_hy/'
        exten_no = 1
    elif 'J' in filter:
        dir = dir + 'ltao_7_7_kj/'
        exten_no = 1
    elif 'H' in filter:
        dir = dir + 'ltao_7_7_hy/'
        exten_no = 0
    elif 'K' in filter:
        dir = dir + 'ltao_7_7_kj/'
        exten_no = 0
    return dir + "evlpsfcl_1_x%s_y%s.fits" % (x_s,y_s), exten_no

def test_liger_psf():
    from astropy.io import fits
    import matplotlib.pyplot as plt
    filters = ['Z', 'Y', 'J', 'H', 'K']
    mode = 'imager'
    simdir = '/Volumes/Backup Plus/liger/sim/'
    for filt in filters:
        psfname, exten_no = get_liger_psf(mode, filt, [7.5,0])
        print(psfname)
        print(exten_no)
        dat = fits.open(simdir + psfname)[exten_no]
        print(filt)
        print(dat.header['WVL'])
        print(np.shape(dat))
        plt.imshow(dat.data)
        plt.show()


