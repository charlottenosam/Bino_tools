# =====================================================================
# bino_obj_ypos.py
#
# Find the y position of slits on the mask (measured from bottom of slit)
# 
# N.B. Seems like the actual source offsets from this value
#      with a normal distribution ~N(0, 1 PSF FWHM)
#
# From Sean:
#   target_offset=((mask_entry.y-mask_entry.bbox[1])*5.98802)/0.24
#
# INPUTS:
#
# HISTORY:
#   Started:                 2019-03-21 C Mason (CfA)
#
# =====================================================================

import numpy as np
from astropy.io import fits

# =====================================================================

arcsec_per_mm = 5.98802
arcsec_per_px = 0.24

def bino_get_obj_ypos(fname, type='slits2D'):

    if type == 'dark':
        ypos = get_ypos_from_darks(fname)
    elif type == 'slits2D':
        ypos = get_ypos_from_2Dslits(fname)

    return ypos

def get_ypos_from_darks(dark_fname):
    """
    Get ypos in pixels using table in darks
    """

    # Load dark
    dark_hdu = fits.open(dark_fname)

    # Load tables with the slit dimensions for each side of the detector
    sideA = dark_hdu[3].data
    sideB = dark_hdu[4].data

    target_offset_px = []
    slit_height_px   = []
    
    for side in [sideA, sideB]:
        
        # Number of targets on mask (i.e. exclude guide stars)
        ntargets = side['NTARGETS'][0]

        # Object position on mask
        obj_y = side['SLITY'][0][:ntargets]
        
        # Bounding box of slit Y (position of top and bottom in mm)
        slit_poly_y = side['POLY_Y'][:,:,:ntargets]
        slit_y_bot  = slit_poly_y[0,0,:]
        slit_y_top  = slit_poly_y[0,1,:]
        
        # Height of slit
        slit_height_px.append((slit_y_top - slit_y_bot)*arcsec_per_mm/arcsec_per_px)

        # Pixel offset positions of objects in slits
        target_offset_px.append((obj_y - slit_y_bot)*arcsec_per_mm/arcsec_per_px)

    slit_height_px   = np.array([item for sublist in slit_height_px for item in sublist])
    target_offset_px = np.array([item for sublist in target_offset_px for item in sublist])

    return target_offset_px


def get_ypos_from_2Dslits(slits2D_fname):
    """
    Get ypos in pixels using headers of 2D slits
    """

    # Load file
    slits2D_hdu = fits.open(slits2D_fname)

    target_offset_px = []

    for slit in slits2D_hdu[1:]:
        target_offset_px.append(slit.header['SLITYPIX'])

    target_offset_px = np.array(target_offset_px)

    return target_offset_px