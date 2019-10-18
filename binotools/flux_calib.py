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
                        transmission_filename=None,
                        verbose=True, overwrite=True):

    if verbose:
        print('Processing %s...' % spec_filename.split('/')[-1])

    # Load flux calibration file
    flux_calib = np.genfromtxt(fluxcalib_filename, names=True)

    if transmission_filename is not None:
        transmission_tab      = np.genfromtxt(transmission_filename, names=True)
        transmission_function = transmission_tab['transmission']
    else:
        transmission_function = np.ones_like(flux_calib['flux_calibration'])

    hdulist = fits.open(spec_filename)
    
    for slit in hdulist:
        dat2D = slit.data

        if type(dat2D) == np.ndarray:
            
            # Do transmission correction
            dat2D /= transmission_function
            dat2D[dat2D == 0.] = np.nan
            if transmission_filename is not None:
                slit.header['TELLCORR'] = 'Using %s' % (transmission_filename.split('/')[-1])

            # Do flux calib
            slit.data = dat2D * flux_calib['flux_calibration']
            
            slit.header['FLUXCAL'] = 'Using %s' % (fluxcalib_filename.split('/')[-1])
    
    spec_filename_fluxcalib = spec_filename.replace('.fits', '_fluxcalib.fits')
    hdulist.writeto(spec_filename_fluxcalib, overwrite=overwrite)

    if verbose:
        print('    Written flux calib file to %s' % spec_filename_fluxcalib)

    return