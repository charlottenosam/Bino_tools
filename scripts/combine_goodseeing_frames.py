# =====================================================================
# combine_goodseeing_frames.py
#
# Combine frames with seeing < seeing_cut
#
# Takes about 30mins to run
#
# INPUTS:
#
# HISTORY:
#   Started:                 2019-11-06 C Mason (CfA)
#
# =====================================================================

import numpy as np
import matplotlib.pylab as plt
import os, sys
import pandas as pd
import datetime, time

from astropy.io import fits
import astropy.stats

from binotools import calbino
import binotools as bt

import argparse            # argument managing
now = datetime.datetime.now()

# ==============================================================================
# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ----
# ---- optional arguments ----
parser.add_argument("--seeing_cut", type=float, help="Use seeing only below this value [default = 1'']")
parser.add_argument("--sigma_clip", type=float, help="Sigma clip sigma [default = 3 -- THIS WORKS, DO NOT CHANGE]")
parser.add_argument("--max_iters", type=int, help="Sigma clip maxiters [default = 5 -- THIS WORKS, DO NOT CHANGE]")
# ---- flags ------
# parser.add_argument("--clobber_calc_calib", action="store_true", help="Don't calculate a new calibration")
parser.add_argument("--no_tell_corr", action="store_true", help="Don't use telluric correction")

args = parser.parse_args()

# =====================================================================
# Seeing cut
seeing_cut = 1.
if args.seeing_cut:
    seeing_cut = args.seeing_cut
print('Using only seeing < %.2f arcsec' % seeing_cut)

# Sigma clipping sigma [3 works best!!]
sigma_clip = 3.
if args.sigma_clip:
    sigma_clip = args.sigma_clip
print('Using sigma_clip = %.1f' % sigma_clip)

# Sigma clipping max iterations
max_iters = 5
if args.max_iters:
    max_iters = args.max_iters
print('Using sigma clip max_iters = %i' % max_iters)

# Telluric correction?
tell_corr = True
if args.no_tell_corr:
    tell_corr = False
    print('Will NOT do telluric correction when calibrating')

# =====================================================================
datadir  = os.environ['BINO_DIR']+'SAO-9/individual_frames/'

# =====================================================================
# Load seeing list
seeing_filename = datadir+'seeing_PSF_arcsec.txt'
if tell_corr:
    seeing_filename = seeing_filename.replace('.txt','_tellcorr.txt')
seeing_list = pd.read_csv(seeing_filename, sep='\t', names=['fname','seeing'])
seeing_good = seeing_list[seeing_list.seeing < seeing_cut]

print('Using %i/%i frames' % (len(seeing_good),len(seeing_list)))

# =====================================================================
# Combine directory
combine_dir = os.environ['BINO_DIR']+'SAO-9/combined_fluxcalib/combine_%i_frames_seeing<%.2farcsec_%s' % \
                                             (len(seeing_good), seeing_cut, now.strftime('%Y-%m-%d'))

if tell_corr:
    combine_dir = combine_dir.replace('arcsec','arcsec_tellcorr')

if not os.path.exists(combine_dir):
    os.makedirs(combine_dir)

frames = [frame_path.replace('.fits', '_fluxcalib.fits') for frame_path in seeing_good.fname]
if tell_corr:
    frames = [frame_path.replace('.fits', '_tellcorr.fits') for frame_path in frames]
n_ob   = len(frames)

# Get number of slits
hdu0     = fits.open(frames[0])
hdu_err0 = fits.open(frames[0].replace('abs_slits_lin', 'abs_err_slits_lin'))
nslits   = len(hdu0) - 1
exptime  = hdu0[1].header['EXPTIME']

flux_combined_file = combine_dir+'/combined_fluxcalib_abs_slits_lin.fits'
err_combined_file  = combine_dir+'/combined_fluxcalib_abs_err_slits_lin.fits'

print('Saving to %s' % flux_combined_file)

hdu_flux_combined = hdu0.copy()
hdu_err_combined  = hdu_err0.copy()

# =====================================================================
# Combine each slit
time_start = time.time()
for slit_i in range(nslits):
    
    if type(hdu_flux_combined[slit_i].data) == np.ndarray:

        for k, f in enumerate(frames):
            
            # Open frame
            hdu_flux = fits.open(f)
            hdu_err  = fits.open(f.replace('abs_slits','abs_err_slits'))
                
            # Get flux and error
            flux_i = hdu_flux[slit_i].data
            err_i  = hdu_err[slit_i].data
            
            # Stack frames
            if k == 0:
                nx1  = int(hdu_flux[slit_i].header['NAXIS1'])
                ny1  = int(hdu_flux[slit_i].header['NAXIS2'])

                print('Slit %i/%i  OB=%i/%i, ny    =%i' % (slit_i, nslits, k, n_ob, ny1))
               
                flux_cube = np.nan * np.ones((ny1, nx1, n_ob))
                err_cube  = np.nan * np.ones((ny1, nx1, n_ob))
                
                flux_cube[:,:,k] = flux_i
                err_cube[:,:,k]  = err_i

            else:
                ny_cur = hdu_flux[slit_i].header['NAXIS2']  
                dy     = int((ny1-ny_cur)/2.0)

                print('Slit %i/%i  OB=%i/%i, ny_cur=%i, dy=%i' % (slit_i, nslits, k, n_ob, ny_cur, dy))
                
                if dy > 0:
                    flux_cube[dy:dy+ny_cur,:,k] = flux_i
                    err_cube[dy:dy+ny_cur,:,k]  = err_i
                else:
                    flux_cube[:,:,k] = flux_i[-dy:-dy+ny1,:]
                    err_cube[:,:,k]  = err_i[-dy:-dy+ny1,:]

        # Sigma clip
        flux_cube_clipped = astropy.stats.sigma_clip(flux_cube, sigma=sigma_clip, axis=2, maxiters=max_iters)
        err_cube_clipped  = np.ma.array(err_cube, mask=flux_cube_clipped.mask)

        # Combine
        flux_combined_clipped = np.nanmean(flux_cube_clipped, axis=2).filled(fill_value=np.nan)
        err_combined_clipped  = np.sqrt(np.nanmean(err_cube_clipped**2., axis=2)).filled(fill_value=np.nan) 

        # Do flux calib
        hdu_flux_combined[slit_i].data = flux_combined_clipped 
        hdu_err_combined[slit_i].data  = err_combined_clipped

        hdu_flux_combined[slit_i].header['EXPTIME'] = exptime * n_ob
        hdu_err_combined[slit_i].header['EXPTIME']  = exptime * n_ob
    
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%dT%H:%M:%S'))

hdu_flux_combined.writeto(flux_combined_file, overwrite=True)
hdu_err_combined.writeto(err_combined_file, overwrite=True)

time_end = time.time()
print('Took %.0f minutes' % ((time_end-time_start)/60.))
