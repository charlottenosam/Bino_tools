# =====================================================================
# select_goodseeing_frames.py
#
# Calculate flux calibration to spectra in each frame
#
# Flux calibration from Ben's calbino.py script (fitting F star spectra),
# fit with 2nd order polynomial to smooth
#
# INPUTS:
#
# HISTORY:
#   Started:                 2019-11-05 C Mason (CfA)
#
# =====================================================================

import numpy as np
import matplotlib.pylab as plt
import os, sys
import pandas as pd
import datetime

from astropy.io import fits

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
# ---- flags ------
# parser.add_argument("--clobber_calc_calib", action="store_true", help="Don't calculate a new calibration")

args = parser.parse_args()

# =====================================================================
# Seeing cut
seeing_cut = 1.
if args.seeing_cut:
    seeing_cut = args.seeing_cut
print('Using only seeing < %.2f arcsec' % seeing_cut)

# =====================================================================
datadir  = os.environ['BINO_DIR']+'SAO-9/individual_frames/'

# =====================================================================
# Load seeing list
seeing_filename = datadir+'seeing_PSF_arcsec.txt'
seeing_list     = pd.read_csv(seeing_filename, sep='\t', names=['fname','seeing'])

seeing_good = seeing_list[seeing_list.seeing < seeing_cut]

print('Using %i/%i frames' % (len(seeing_good),len(seeing_list)))

# =====================================================================
# Combine directory
combine_dir = os.environ['BINO_DIR']+'SAO-9/combined_fluxcalib/combine_%i_frames_seeing<%.2farcsec_%s' % \
											 (len(seeing_good), seeing_cut, now.strftime('%Y-%m-%d'))

if not os.path.exists(combine_dir):
    os.makedirs(combine_dir)

combine_script_file = combine_dir+'/run_idl_combine_seeing<%.2farcsec.txt' % seeing_cut

# List frames
inpdir = [frame_path.split('obj_')[0] for frame_path in seeing_good.fname]
fdata  = ['obj_'+frame_path.split('obj_')[-1].replace('.fits', '_fluxcalib.fits') for frame_path in seeing_good.fname]
efdata = [frame_path.replace('abs_slits','abs_err_slits') for frame_path in fdata]

combine_cmd = 'bino_ob_combine,[\'%s\'],\'%s/\',fdata=[\'%s\'],efdata=[\'%s\']' % \
				('\', \''.join(map(str, inpdir)), combine_dir, '\', \''.join(map(str, fdata)), '\', \''.join(map(str, efdata)))

with open(combine_script_file, "w") as text_file:
    text_file.write(combine_cmd)