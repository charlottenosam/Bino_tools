# =====================================================================
# telluric_correction_setup.py
#
# Extract stellar spectra and make molecfit .par files (in raw_data directory)
#
#
# INPUTS:
#
# HISTORY:
#   Started:                 2019-12-12 C Mason (CfA)
#
# =====================================================================

import numpy as np
import matplotlib.pylab as plt
import os, sys
import pandas as pd
import datetime, time
import configparser

from astropy.io import fits

import binotools as bt

import argparse            # argument managing

now = datetime.datetime.now()

# ==============================================================================
config = configparser.ConfigParser(delimiters=':')

# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ----
# ---- optional arguments ----
parser.add_argument("--config_file", type=str, help="Path to config file")
parser.add_argument("--star_id", type=int, help="Star slit ID [default = 19]")
# ---- flags ------
# parser.add_argument("--clobber_calc_calib", action="store_true", help="Don't calculate a new calibration")

args = parser.parse_args()

# =====================================================================

def remove_end_from_string(s):
    return s[:s.rfind('end')]


def load_molecfit_par(par_file_name):
    """Load .par file and parse
    
    """
    
    with open(par_file_name) as f:
        file_content = remove_end_from_string('[section]\n' + f.read())

    config_parser = configparser.RawConfigParser(delimiters=':')
    config_parser.read_string(file_content)
    
    return config_parser


def save_molecfit_par(cp_molecfit, output_par_name=sys.stdout):
    """Save molecfit parameter file
    
    """
    
    if type(output_par_name) is str:
        
        # Save to file
        with open(output_par_name, 'w') as par_file:
            cp_molecfit.write(par_file, space_around_delimiters=True)

        # Delete first line
        with open(output_par_name, 'r') as par_file:
            head, tail = par_file.read().split('\n', 1)
        
        # Add end
        with open(output_par_name, 'w') as par_file:
            par_file.writelines(tail.replace(' :',':')+'end\n')
    
    else:
        cp_molecfit.write(output_par_name)

# =====================================================================
# Get config for Bino
config_file = 'BLAS_1_2019.config'
if args.config_file:
    config_file = args.config_file
config.read(config_file)

star_id = 19
if args.star_id:
    star_id = args.star_id

# =====================================================================
# Get config for molecfit
telluric_dir = config.get('calibration_data', 'telluric_dir')
cp_molecfit  = load_molecfit_par(telluric_dir+'molecfit_BLAS_2019A_star19_example.par')

# =====================================================================
# Get frames
datadir    = config.get('raw_data','data_dir')
spec2D_all = sorted(bt.insensitive_glob(datadir+'/*/*/reduced_series/obj_clean*abs_slits_lin.fits'))
nights     = np.array([s.split('individual_frames/')[-1].split('/')[0] for s in spec2D_all])

spec2D_all = spec2D_all
nights = np.unique(nights)

print('Found %i frames over %i nights' % (len(spec2D_all), len(nights)))

# =====================================================================

wave_A = bt.get_wave_A_from_spec2D(spec2D_all[0])

print('Saved stellar spectrum and molecfit parameter file for:')

for ss, spec2D_file in enumerate(spec2D_all):
    
    night = spec2D_file.split('individual_frames/')[-1].split('/')[0]

    star_spec_filename       = spec2D_file.replace('.fits', '_star%i.txt' % star_id)
    telluric_output_filename = star_spec_filename.split('/')[-1].replace('.txt', '')
    telluric_output_dir      = telluric_dir+'output_individualframes/'+night+'_'+telluric_output_filename
    molecfit_par_filename    = star_spec_filename.replace('.txt','_molecfit.par')
    
    spec2D_hdu = fits.open(spec2D_file)
    
    hdr = spec2D_hdu[0].header
    
    UTC = time.strptime(hdr['UT'],'%H:%M:%S')
    UTC_sec = datetime.timedelta(hours=UTC.tm_hour,minutes=UTC.tm_min,seconds=UTC.tm_sec).total_seconds()
    
    cp_molecfit.set('section', 'filename', star_spec_filename)  # Data file name
    cp_molecfit.set('section', 'output_dir', telluric_output_dir)  # Directory for output files
    cp_molecfit.set('section', 'output_name', telluric_output_filename)  # Name for output files

    # Observing dates
    cp_molecfit.set('section', 'obsdate', hdr['MJD'])  # Observing date in years or MJD in days
    cp_molecfit.set('section', 'utc', UTC_sec)  # UTC in s

    # Observing conditions
    cp_molecfit.set('section', 'telalt', hdr['EL'])  # Telescope altitude angle in deg
    cp_molecfit.set('section', 'pres', hdr['ADCPRES'])  # Pressure in hPa
    cp_molecfit.set('section', 'temp', hdr['ADCTEMP'])  # Ambient temperature in deg C
    cp_molecfit.set('section', 'm1temp', hdr['ADCTEMP'])  # Mirror temperature in deg C
    
    # Save molecfit parameter file
    save_molecfit_par(cp_molecfit, output_par_name=molecfit_par_filename)

    # Get stellar spectrum
    spec2D_star = spec2D_hdu[star_id+1].data
    spec1D_star_eachframe, FWHM_pix = bt.extract1D_psf(spec2D_star, psf_lim=0.02, plot_psf=False, ax=None)
    
    # Save 1D stellar spectrum
    spec_tab = np.array([wave_A/10., spec1D_star_eachframe])
    np.savetxt(star_spec_filename, spec_tab.T, delimiter=' ')

    print('    %i/%i - %s' % (ss, len(spec2D_all), star_spec_filename.split(datadir)[-1]))
