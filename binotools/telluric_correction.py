# =====================================================================
# telluric_correction.py
#
# tools to correct for telluric absorption in bino spectra
#
#
# HISTORY:
#   Started:                 2019-10-18 C Mason (CfA)
#
# =====================================================================
import numpy as np
import os

from astropy import units as u
import astropy.io.fits as fits
import scipy
import matplotlib.pylab as plt
# =====================================================================

class TelluricCorrection(object):
    """
    Binospec telluric correction
    """

    def __init__(self, calib_dir=os.environ['BINO_DIR']+'2649_BLAS_1/calib/',
                transmission_filename='BLAS_2019A_transmission.txt', plot=False):
        """
        Input wavelength array must = transmission wavelength array
        """
        trans_tab = np.genfromtxt(calib_dir+'BLAS_2019A_transmission.txt', names=True)

        self.transmission = trans_tab['transmission']
        self.wave_nm      = trans_tab['wave_nm']

        self.telluric_regions = {'O2 A band':[757., 775.],
                                 'O2 B-band':[685., 694.]}

        self.mask_telluric = np.array([], dtype=int)
        for region in self.telluric_regions:
            self.mask_telluric = np.hstack((self.mask_telluric, 
                                            (np.where((self.wave_nm > self.telluric_regions[region][0]) & \
                                            (self.wave_nm < self.telluric_regions[region][1]))[0])))

        if plot:
            plt.figure(figsize=(8,4), dpi=150)

            plt.plot(self.wave_nm, self.transmission)

            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Transmission')

        return

    def std_telluric_region(self, args, opt_args):
        """
        RMS of corrected flux in region of O2 A-band
        """
        
        alpha, beta = args    
        wave, flux_in = opt_args
        
        flux_corr = self.telluric_correction(wave, flux_in, alpha, beta)
        
        RMS = np.nanstd(flux_corr[self.mask_telluric])

        return RMS

    def telluric_correction(self, wave, flux_in, alpha=0., beta=1.):
        """
        Telluric correction (assuming std at same airmass as target)
        
        F_corr = F_in / T(lambda - alpha)**beta
        """
        
        # Translate and scale transmission
        transmission_function = np.interp(wave-alpha, wave, self.transmission)**beta
        
        flux_out = flux_in/transmission_function
        flux_out[flux_out == 0.] = np.nan
        
        return flux_out

    def optimized_tellcorr(self, wave, flux_in, verbose=False):
        """
        Find alpha and beta parameters by minimising RMS in O2 A-band
        """
        initial_guess = [0, 1.]  # initial guess can be anything
        result = scipy.optimize.minimize(self.std_telluric_region, initial_guess, 
                                         args=[wave, flux_in])
        if verbose:
            print(result.x)

        flux_tellcorr = self.telluric_correction(wave, flux_in, alpha=result.x[0], beta=result.x[1])

        return flux_tellcorr