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

def bino_get_obj_ypos(dark_fname):

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
