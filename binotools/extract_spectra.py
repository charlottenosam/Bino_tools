# =====================================================================
# extract_spectra.py
#
# extraction tools for bino spectra
#
#
# HISTORY:
#   Started:                 2019-10-18 C Mason (CfA)
#
# =====================================================================

import pandas as pd
import numpy as np
import glob
import astropy.io.fits as fits
from astropy.modeling import models, fitting

# =====================================================================

def extract1D_psf(spec2D, psf_lim=0.01, plot_psf=False, ax=None, clip_edges=4):
    """
    Extract 1D spectrum from 2D using PSF
    """
    
    # Get psf for extraction
    psf = np.nansum(spec2D, axis=1)
    pix = np.arange(len(psf))
    
    # Tidy
    psf[:clip_edges]  = np.nan # Clip edges
    psf[-clip_edges:] = np.nan # Clip edges
    psf -= np.nanmin(psf)      # Zero    
    psf /= np.nansum(psf)      # Normalize
    
    # Range to extract over
    psf_range = np.where(psf > psf_lim*np.nanmax(psf))[0]
    
    # Fit ------------------
    
    # Mask array
    psf = np.ma.masked_array(psf, mask=np.zeros_like(psf))
    psf.mask[np.isnan(psf)] = True
    
    # Fit
    g_init = models.Gaussian1D(amplitude=0.2, mean=np.argmax(np.nan_to_num(psf)), stddev=4.)
    fit_g  = fitting.LevMarLSQFitter()
    g      = fit_g(g_init, pix, psf)
    
    FWHM_pix = g.stddev.value * 2.355
    # ----------------------
    
    if plot_psf:
        ax.plot(psf, lw=1, alpha=0.7, label='data')
        ax.axvline(psf_range[0], ls='dashed', lw=0.5, c='k')
        ax.axvline(psf_range[-1], ls='dashed', lw=0.5, c='k')    
        
        ax.plot(pix, g(pix), lw=1, label='Gaussian fit')
        
        ax.legend()
    
    spec1D = np.nansum(spec2D[psf_range], axis=0)
    spec1D[spec1D == 0.] = np.nan

    return spec1D, FWHM_pix