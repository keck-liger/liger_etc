
import os
import numpy as np

def get_filterdat(filt, simdir='~/liger/sensitivity/', mode=''):

    filterfile = os.path.expanduser(simdir + "info/filter_info.dat")
    filterall= np.genfromtxt(filterfile, dtype=str)
    names = ["filterread", "lambdamin", "lambdamax", "lambdac",
             "bw", "backmag", "imagmag", "zp"]
    filterdat = dict(zip(names, filterall.T))
    filts = [i.lower().strip() for i in filterdat['filterread']]
    if mode.lower() == 'ifs':
        index = np.where(np.array(filts) == filt.lower().strip())[0][0]
    else:
        index = np.where(np.array(filts) == filt.lower().strip())[0][0]
    if not isinstance(index, np.int64) and mode.lower() == 'ifs':
        index = np.where(np.array(filts) == filt.lower().strip())[0][0]
    filterdat = dict(zip(names,[float(val) if val[0].isdigit() else str(val) for val in filterall[index]]))
    return filterdat
