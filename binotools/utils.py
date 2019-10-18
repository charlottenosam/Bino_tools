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
import astropy.io.fits as fits

c=3.e5

def get_wave_A_from_spec2D(spec2D_file):
    """
    Get wavelength array [Angstrom] from 2d spectra
    """
    hdu_list  = fits.open(spec2D_file)

    spec2D = hdu_list[1].data

    # # Get wavelength vector
    header  = hdu_list[1].header
    nstep   = spec2D.shape[1]
    wave_nm = header['CRVAL1'] + np.linspace(0., nstep*header['CDELT1'], nstep)
    wave_A  = wave_nm * 10.
    return wave_A

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

def f_cont(m, wave):
    """
    Get continuum flux density at observed wavelength 
    in [erg/cm^2/s/A]
    
    Input:
        m       [AB] apparent mag 
        wave    [A] observed frame wavelength
    """
    f0  = 3.631E-20  # erg/s/Hz/cm2
    c   = 3.E5       # km/s
    c_A = c * 1.E13  # km/s into A/s
    
    fcont = f0 * 10**(-0.4*m) * c_A / wave**2.
        
    return fcont