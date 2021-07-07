import numpy as np

def get_psf(mode,psf_loc):
    x,y = psf_loc
    if mode.lower() == "ifs":
        scale_arr = [0.02, 0.035, 0.05, 0.1] # plate scale [mas]
        x_arr = y_arr = [0] # position on focal plane [arcsec]
        ins = "ifu"
    elif mode.lower() == "imager":
        scale_arr = [0.02] # plate scale [mas]
        x_arr = y_arr = [0, -7.5, 7.5, -15, 15] # position on focal plane [arcsec]
        ins = "im"
    # select the closest file, in case the user guesses wrong
    x_ind = np.argmin(np.abs(np.array(x_arr) - x))
    y_ind = np.argmin(np.abs(np.array(y_arr) - y))
    x_s = x_arr[x_ind]
    y_s = y_arr[y_ind]
    return "PSFs_SCAO_05_21_19/evlpsfcl_1_x%s_y%s.fits" % (x_s,y_s)
