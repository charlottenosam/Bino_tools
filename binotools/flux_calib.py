# =====================================================================
# flux_calib.py
#
# Apply flux calibration to spectra
#
# Flux calibration from Ben's calbino.py script (fitting F star spectra),
# fit with 2nd order polynomial to smooth
#
# INPUTS:
#
# HISTORY:
#   Started:                 2019-05-21 C Mason (CfA)
#
# =====================================================================

import numpy as np
from astropy.io import fits
import os

# =====================================================================

def flux_calib_specfile(spec_filename, fluxcalib_filename, 
                        verbose=True, overwrite=True):

    # Load flux calibration file
    flux_calib = np.genfromtxt(fluxcalib_filename, names=True)

    if verbose:
        print('Processing %s...' % spec_filename.split('/')[-1])

    hdulist = fits.open(spec_filename)
    
    for slit in hdulist:
        dat2D = slit.data

        if type(dat2D) == np.ndarray:
            # Do flux calib
            slit.data = dat2D * flux_calib['flux_calibration']
            
            slit.header['FLUXCAL'] = 'Using %s' % (fluxcalib_filename.split('/')[-1])
    
    spec_filename_fluxcalib = spec_filename.replace('.fits', '_fluxcalib.fits')
    hdulist.writeto(spec_filename_fluxcalib, overwrite=overwrite)

    if verbose:
        print('    Written flux calib file to %s' % spec_filename_fluxcalib)

    return