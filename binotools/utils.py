# =====================================================================
# utils.py
#
# utils for bino tools
#
#
# HISTORY:
#   Started:                 2019-03-20 C Mason (CfA)
#
# =====================================================================

import pandas as pd
import numpy as np
import glob

c=3.e5

def half_gauss(x, amp, cen, FWHM):
    """
    Half-gaussian line model
    """
    sig = FWHM/2.355
    gauss = amp * np.exp(-0.5*(x-cen)**2 / sig**2.)
    gauss[x<cen] = 0.
    return gauss

def wave_to_kms(wave, mu0):
    return c*(wave-mu0)/mu0

def insensitive_glob(pattern):
    """
    Case insensitive find file names
    """
    def either(c):
        return '[%s%s]'%(c.lower(),c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either,pattern)))