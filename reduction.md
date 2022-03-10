# Reduction steps

Create a `.config` file which stores all the info needed to run reductions.

## Telluric correction

- Run `scripts/telluric_correction_setup.py` to extract stellar spectra from each frame and make `molecfit` parameter files
- Run `scripts/telluric_correction_get_trans.py` to run `molecfit` on each frame and generate a transmission spectrum

## Flux calibration

`flux_calib_eachframe.py` calculates the flux calibration vector for each frame (from F stars) and then flux calibrates the frames. This file also outputs the FWHM of the PSF for each frame to `datadir+'seeing_PSF_arcsec.txt'`

- Find directory structure with frames [change in `flux_calib_eachframe.py`]
- Run `scripts/flux_calib_eachframe.py`
	+ outputs PSF list `datadir+'seeing_PSF_arcsec.txt'`
	+ outputs calibration arrays for every frame `datadir+'calibration_frames.txt'`
	+ flux calibrates frames and saves as `fname`+`_fluxcalib.fits`

## Combining frames

Combine exposures using sigma clipping. Small y offset adjustments based on the headers.

- Run `scripts/combine_goodseeing_frames.py --seeing_cut 1.`





### IDL version
- Run `scripts/select_goodseeing_frames.py --seeing_cut 1.`
	+ outputs IDL command script for combining frames
- Run IDLDE
	+ open `Bino_tools/binotools/bino_ob_combine.pro`
	+ run with script in combine_dir
