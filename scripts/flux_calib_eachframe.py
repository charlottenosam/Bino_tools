# =====================================================================
# flux_calib_eachframe.py
#
# Calculate flux calibration to spectra in each frame
#
# Flux calibration from Ben's calbino.py script (fitting F star spectra),
# fit with 2nd order polynomial to smooth
#
# INPUTS:
#
# HISTORY:
#   Started:                 2019-10-18 C Mason (CfA)
#
# =====================================================================

import numpy as np
import matplotlib.pylab as plt
import os, sys
import pandas as pd
import datetime
import configparser

from astropy.io import fits

from binotools import calbino
import binotools as bt

import argparse            # argument managing

now = datetime.datetime.now()

# ==============================================================================
config = configparser.SafeConfigParser(delimiters=':')

# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ----
# ---- optional arguments ----
parser.add_argument("--config_file", type=str, help="Path to config file")
parser.add_argument("--slit_csv", type=str, help="Path to list of slits")
parser.add_argument("--cut_badpix", type=float, help="Percentile of bad pixels to mask when calibrating [default = 95%]")
parser.add_argument("--smooth_calib_order", type=int, help="Calibration smoothing polynomial order [default = 2]")
# ---- flags ------
parser.add_argument("--no_tell_corr", action="store_true", help="Don't do telluric correction to standard star")
# parser.add_argument("--clobber_calc_calib", action="store_true", help="Don't calculate a new calibration")

args = parser.parse_args()

# =====================================================================

# Get config
config_file = 'BLAS_1_2019.config'
if args.config_file:
    config_file = args.config_file

config.read(config_file)
star_id = int(config.get('slits','star_id'))

# Get slit table
slit_csv_fname = config.get('slits','slit_csv_fname')
if args.slit_csv:
    slit_csv_fname = args.slit_csv

# Telluric correction?
tell_corr = True
if args.no_tell_corr:
    tell_corr = False
    print('Will NOT do telluric correction when calibrating')

# Bad pixel mask percentile
cut_badpix = 95.
if args.cut_badpix:
    cut_badpix = args.cut_badpix
print('Cutting to %.1f percent bad pixels' % cut_badpix)

smooth_calib_order = 2
if args.smooth_calib_order:
    smooth_calib_order = args.smooth_calib_order
print('Smoothing calibration with %i order polynomial' % smooth_calib_order)

# =====================================================================
# Init flux cal and telluric correction
fluxcal  = calbino.FluxCalBino()

# =====================================================================
# Find stars
slits = pd.read_csv(slit_csv_fname)

# fits file ext # = slits.index + 1
slits['extnum'] = slits.index + 1

stars = slits[slits.type == 'STANDARD']

print('Found stars on slits:',stars.extnum.values)

# Add star mags
mags_all = []
for i in range(len(stars)):
    fluxcal.get_star_mags(stars.iloc[i].ra, stars.iloc[i].dec, verbose=False)
    star_mags = fluxcal.star_mags
    mags_all.append(star_mags)
    
stars['mags'] = mags_all
print('Star magnitudes from SDSS: gri =',stars['mags'])

# =====================================================================
# Get data
# New
datadir    = config.get('raw_data','data_dir')
spec2D_all = sorted(bt.insensitive_glob(datadir+'/*/*/reduced_series/obj_clean*abs_slits_lin.fits'))
nights     = np.array([s.split('individual_frames/')[-1].split('/')[0] for s in spec2D_all])

spec2D_all = spec2D_all
nights = np.unique(nights)

print('Found %i frames over %i nights' % (len(spec2D_all), len(nights)))

#####################################################
# For each frame calculate flux calibration

seeing_filename = datadir+'seeing_PSF_arcsec.txt'
calib_filename = datadir+'calibration_frames.txt'
if tell_corr:
    seeing_filename = seeing_filename.replace('.txt','_tellcorr.txt')
    calib_filename = calib_filename.replace('.txt','_tellcorr.txt')

if os.path.exists(seeing_filename):
    os.remove(seeing_filename)

wave_A = bt.get_wave_A_from_spec2D(spec2D_all[0])

if os.path.exists(calib_filename) is False:
    
    polyfitted_calibration_frame = np.zeros((len(spec2D_all),len(wave_A)))
    
    for ss, spec2D_file in enumerate(spec2D_all):

        spec2D_hdu = fits.open(spec2D_file)

        spec2D_file_dir = spec2D_file.split(spec2D_file.split('/')[-1])[0]

        # Init telluric correction
        star_spec_filename    = spec2D_file.replace('.fits', '_star%i.txt' % star_id)
        transmission_filename = star_spec_filename.replace('.txt','_transmission.txt')
        if os.path.exists(transmission_filename) is False:
            transmission_filename = sorted(bt.insensitive_glob(spec2D_file_dir+'*_transmission.txt'))[-1]
        tellcorr = bt.TelluricCorrection(calib_dir='', transmission_filename=transmission_filename)

        #------------------------------------------------
        # Estimate PSF
        seeing_FWHM_arcsec = np.zeros(len(stars))
        for i in range(len(stars)):
            spec2D_star = spec2D_hdu[stars.index[i]+1].data
            spec1D_star_eachframe, FWHM_pix = bt.extract1D_psf(spec2D_star, psf_lim=0.02, plot_psf=False, ax=None)
            seeing_FWHM_arcsec[i] = FWHM_pix * bt.arcsec_per_px

        if np.diff(seeing_FWHM_arcsec) > 0.2:
            print('WARNING: %s, difference between PSF of each star is > 0.2 arcsec' % spec2D_file)

        seeing_PSF_med = np.median(seeing_FWHM_arcsec)

        with open(seeing_filename, 'a') as PSF_file:
            PSF_file.write('%s\t%.3f\n' % (spec2D_file, seeing_PSF_med))

        #------------------------------------------------
        # Calibration
        polyfitted_calibration_allstars = np.zeros((len(stars),len(wave_A)))
        for i in range(len(stars)):
            # Load star mags
            fluxcal.star_mags = stars.iloc[i]['mags']
            
            spec2D_star = spec2D_hdu[stars.index[i]+1].data
            
            # Get 1D spectrum
            spec1D_star_eachframe, FWHM_pix = bt.extract1D_psf(spec2D_star, psf_lim=0.02, plot_psf=False, ax=None)
            
            # Do telluric correction
            if tell_corr:
                spec1D_star_eachframe = tellcorr.optimized_tellcorr(wave_A, spec1D_star_eachframe)
        
            # Calibration
            calibration = fluxcal.flux_calibration(spec1D_star_eachframe, wave_A)
        
            # Filter bad pixels
            low, high = np.nanpercentile(calibration, 100-cut_badpix), np.nanpercentile(calibration, cut_badpix)
            calibration[np.where((calibration < low) | (calibration > high))] = np.nan
            
            # Smooth calibration
            polyfitted_calibration_allstars[i] = fluxcal.smooth_calibration(wave_A, calibration, degree=smooth_calib_order)
            
        polyfitted_calibration_frame[ss] = np.nanmedian(polyfitted_calibration_allstars, axis=0)

    df = pd.DataFrame(data=polyfitted_calibration_frame.T, columns=spec2D_all)
    df['wave_A'] = wave_A

    df.to_csv(calib_filename, sep='\t', index=False)

df = pd.read_csv(calib_filename, sep='\t')

#####################################################
# Do flux calibration to frames

for ss, spec2D_file in enumerate(spec2D_all):

    spec2D_file_dir = spec2D_file.split(spec2D_file.split('/')[-1])[0]
    err2D_file = spec2D_file.replace('abs_slits_lin','abs_err_slits_lin')

    spec_filename_unfluxed = [spec2D_file, err2D_file]

    # Init telluric correction
    star_spec_filename    = spec2D_file.replace('.fits', '_star%i.txt' % star_id)
    transmission_filename = star_spec_filename.replace('.txt','_transmission.txt')
    if os.path.exists(transmission_filename) is False:
        transmission_filename = sorted(bt.insensitive_glob(spec2D_file_dir+'*_transmission.txt'))[-1]
    tellcorr = bt.TelluricCorrection(calib_dir='', transmission_filename=transmission_filename)

    for sss, spec_filename in enumerate(spec_filename_unfluxed):
        
        hdulist = fits.open(spec_filename)

        # Flux calib each slit
        for slit in hdulist:
            dat2D = slit.data

            if type(dat2D) == np.ndarray:

                # Do transmission correction
                if tell_corr:
                    # dat2D = tellcorr.optimized_tellcorr(wave_A, dat2D)
                    dat2D = tellcorr.telluric_correction(wave_A, dat2D)
                    slit.header['TELLCORR'] = 'Using %s' % (transmission_filename)

                dat2D[dat2D == 0.] = np.nan

                # Do flux calib                
                slit.data = dat2D * df[spec2D_file][None,:]

                slit.header['FLUXCAL'] = 'flux calibration %s' % now.strftime('%Y-%m-%dT%H:%M:%S')

        spec_filename_fluxcalib = spec_filename.replace('.fits', '_fluxcalib.fits')
        if tellcorr:
            spec_filename_fluxcalib = spec_filename_fluxcalib.replace('.fits', '_tellcorr.fits')

        hdulist.writeto(spec_filename_fluxcalib, overwrite=True)

    print('%i/%i - flux calibration for %s' % (ss+1, len(spec2D_all), spec2D_file))