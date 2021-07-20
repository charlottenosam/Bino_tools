# =====================================================================
# telluric_correction_get_trans.py
#
# Run molecfit to get transmission spectra for each frame
# Took 70 mins for 63 frames
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
import subprocess

from astropy.io import fits

import binotools as bt

import argparse            # argument managing
from joblib import Parallel, delayed

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
# Get config for Bino
config_file = 'BLAS_1_2019.config'
if args.config_file:
    config_file = args.config_file
config.read(config_file)

star_id = 19
if args.star_id:
    star_id = args.star_id

# =====================================================================
molecfit_path = config.get('calibration_data', 'molecfit_path')
telluric_dir = config.get('calibration_data', 'telluric_dir')+'output_individualframes/'
telluric_log = telluric_dir+'telluric_%s.log' % now.strftime('%Y-%m-%d')

# =====================================================================
# Get frames
datadir    = config.get('raw_data','data_dir')
spec2D_all = sorted(bt.insensitive_glob(datadir+'/*/*/reduced_series/obj_clean*abs_slits_lin.fits'))
nights     = np.array([s.split('individual_frames/')[-1].split('/')[0] for s in spec2D_all])

spec2D_all = spec2D_all
nights = np.unique(nights)

print('Found %i frames over %i nights' % (len(spec2D_all), len(nights)))

# =====================================================================

print('Saved stellar spectrum and molecfit parameter file for:')

def run_molecfit(ss):

    spec2D_file = spec2D_all[ss]
    
    night = spec2D_file.split('individual_frames/')[-1].split('/')[0]

    star_spec_filename       = spec2D_file.replace('.fits', '_star%i.txt' % star_id)
    telluric_output_filename = star_spec_filename.split('/')[-1].replace('.txt', '')
    molecfit_par_filename    = star_spec_filename.replace('.txt','_molecfit.par')
    transmission_filename    = star_spec_filename.replace('.txt','_transmission.txt')
    
    telluric_output_dir      = telluric_dir+night+'_'+telluric_output_filename

    # if os.path.exists(transmission_filename) is False:
        
    try:
    # Run molecfit
        cmd_molecfit = molecfit_path+'molecfit %s' % molecfit_par_filename
        subprocess.check_call(cmd_molecfit, shell=True)

        # Get transmission spectrum
        cmd_calctrans = molecfit_path+'calctrans %s' % molecfit_par_filename
        subprocess.check_call(cmd_calctrans, shell=True)

        # Output transmission file
        trans = np.genfromtxt('%s/%s_tac.asc' % (telluric_output_dir, telluric_output_filename), skip_header=2)
        wave_um      = trans[:,2]
        transmission = trans[:,6]

        trans_tab = np.array([wave_um*1e3, transmission])
        np.savetxt(transmission_filename, trans_tab.T, delimiter='\t', header='wave_nm\ttransmission')

        # Delete all the other files
        subprocess.call('rm -rf %s' % (telluric_output_dir), shell=True)

        print('    %i/%i - %s/%s' % (ss, len(spec2D_all), night, telluric_output_filename))
        with open(telluric_log, 'a') as log:
            log.write('\nSUCCESS %s/%s' % (night, telluric_output_filename))

    except:
        print('    %i/%i - FAILED %s/%s' % (ss, len(spec2D_all), night, telluric_output_filename))
        with open(telluric_log, 'a') as log:
            log.write('\nFAILED  %s/%s' % (night, telluric_output_filename))

    # else:
    #     print('    %i/%i - already exists %s/%s' % (ss, len(spec2D_all), night, telluric_output_filename))

# Run for all frames
time_start = time.time()
Parallel(n_jobs=5)(delayed(run_molecfit)(i) for i in range(len(spec2D_all)))
time_end = time.time()

print('Took %.0f minutes' % ((time_end-time_start)/60.))
