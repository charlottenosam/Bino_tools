import numpy as np
import h5py
from sedpy.observate import load_filters, getSED
import os

import astropy.stats
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy import units as u
import astropy.io.fits as fits
from astropy.modeling import models, fitting
from scipy.signal import medfilt
import scipy

#libname = "/Users/bjohnson/Codes/SPS/ckc/ckc/spectra/lores/c3k_v1.3_R5K.h5"
libname_dir = os.environ['WORK_DIR']+'/code/ReductionSoftware/fstars/'
libname = libname_dir+"c3k_v1.3_R5K.Fstars.h5"
lightspeed = 3e18


class FluxCalBino(object):
    """
    Binospec flux calibration
    """

    def __init__(self, filter_set=["sdss_g0", "sdss_r0", "sdss_i0"]):
        """
        
        """

        # Get a reasonable set of model spectra
        self.libwave, self.libflux, self.libparams = self.get_library()
        self.filters = load_filters(filter_set)

        return

    def get_library(self):
        """Here's the library, after removing stars with very non-solar metallicities

        :returns libwave: 
            wavelength vector for the model stars, ndarray shape (nwave,)

        :returns flux: 
            flux vectors for the model stars, ndarray of shape (nmodel, nwave)

        :returns params: 
            parameter values for the models, structured ndarray of shape (nmodel,)
        """
        # Load the library of stellar models
        with h5py.File(libname, "r") as lib:
            wave = lib["wavelengths"][:]
            params = lib["parameters"][:]
            # restrict to ~solar metallicity
            g = (params["afe"] == 0) & (params["feh"] > -0.1) & (params["feh"] < 0.1)
            spectra = lib["spectra"][g, :]

        # convert to flambda
        flam = spectra * lightspeed / wave**2
        libwave = wave.copy()
        libflux = flam

        return libwave, libflux, params[g]


    def get_star_mags(self, ra, dec, radius_arcsec=1.2, verbose=True):
        """
        Find star mags from SDSS

        returns magnitudes of star in g, r, i
        """        
        if ra is not None:
            # Get mags from SDSS directly?
            pos  = coords.SkyCoord(ra, dec, unit="deg", frame='icrs')
            flds = ["ra", "dec", "psfMag_g", "psfMag_r", "psfMag_i",
                    "probPSF", "run", "rerun", "camcol", "field"]
            phot = SDSS.query_region(pos, radius=radius_arcsec*u.arcsec, spectro=False,
                                     photoobj_fields=flds).to_pandas()
            phot = phot.sort_values('probPSF', ascending=False)
            
            if verbose:
                print(phot)
                
            assert phot.iloc[0]["probPSF"] > 0

            self.star_mags = np.array([phot.iloc[0]["psfMag_g"], phot.iloc[0]["psfMag_r"], phot.iloc[0]["psfMag_i"]])
        
        else:
            # Put them in by hand
            self.star_mags = np.array([23., 23.02, 23.1]) # or something??
            
        return

    def choose_model(self, star_mags, nmod=1):
        """Choose the library model with colors closest to the input colors

        :param mags: 
            calibrator star magnitudes, ndarray

        :param filters: 
            list of sedpy.Filter objects, same length as `mags`

        :param libwave:

        :param libflux:

        :param nmod: integer, optional, default: 1 
            return the `nmod` closest models in color space.  Getting multiple
            models can be useful to explore calibration uncertainty.
        """
        target = star_mags[1:] - star_mags[0]

        # Get SED of models
        seds = getSED(self.libwave, self.libflux, self.filters)

        # get colors of models
        colors = seds[:, 1:] - seds[:, 0][:, None]

        # Get distance of models from target colors
        dist = ((target - colors)**2).sum(axis=-1)

        # choose the N-closest models
        order = np.argsort(dist)
        best = order[:nmod]

        return self.libflux[best, :]


    def flux_calibration(self, data_flux, data_wave, z=0.):  
        """
        Get the flux calibration vector from the star and data
        """

        # choose the model(s) with colors closest to the calibrator
        best_model = self.choose_model(self.star_mags, nmod=1)

        # Now work out the normalization of the model from the average magnitude offset
        best_sed = getSED(self.libwave, best_model, self.filters)
        dm       = np.mean(self.star_mags - best_sed, axis=-1)
        conv     = np.atleast_1d(10**(-0.4 * dm))

        # Here, finally, is the fluxed model (erg/s/cm^2/AA)
        fluxed_model = best_model * conv[:, None]

        # Now get the bestest model on the same wavelength vector as the data   
        fluxed_model_interp = np.interp(data_wave, self.libwave * (1 + z), fluxed_model[0])
        
        calibration = fluxed_model_interp / data_flux
        
        return calibration


    def smooth_calibration(self, wave, calibration, polyfit=True, 
                           degree=2, medfilt_kernel_size=101):

        # You probably want to median filter the calibration vector.  # Perhaps
        # after some sigma clipping.  differences on small scales could be due to
        # model imperfections (wrong metallicity, wrong gravity for model, LSF
        # mismatch)
        # you could also fit the calibration vector with a polynomial, taking into
        # account errors
        if polyfit:
            # Fit the data using astropy.modeling
            p_init = models.Polynomial1D(degree=degree)
            calibration_masked = np.ma.masked_array(calibration, mask=np.isnan(calibration))

            fit_p = fitting.LinearLSQFitter()
            p = fit_p(p_init, wave, calibration_masked)

            # Plot the data with the best-fit model
            return p(wave)

        else:
            # Do median filter smoothing
            smoothed_calibration = medfilt(calibration, medfilt_kernel_size)
        
            return smoothed_calibration

    
if __name__ == "__main__":


    # And here you'd get the bino spectrum for the calibrator star.
    data_pix, data_wave, data_flux = np.genfromtxt("star.csv", delimiter=",", unpack=True, skip_header=1)
    # header units are 1e-19 erg/s/cm^2/AA
    data_flux *= 1e-19
    # Get from header slit_ra?
    ra, dec = None, None
    
    # --- Data ---
    # Here is where you put the SDSS mags and the bino spectrum for the star of interest
    # This is the filters and magnitudes for the calibrator star
    filters = load_filters(["sdss_g0", "sdss_r0", "sdss_i0"])
    star_mags = get_star_mags(ra, dec)
 
    # Get a reasonable set of model spectra
    libwave, libflux, libparams = get_library()

    # choose the model(s) with colors closest to the calibrator
    best_model = choose_model(star_mags, filters, libwave, libflux, nmod=1)
    
    # Now work out the normalization of the model from the average magnitude offset
    best_sed = getSED(libwave, best_model, filters)
    dm = np.mean(star_mags - best_sed, axis=-1)
    conv = np.atleast_1d(10**(-0.4 * dm))

    # Here, finally, is the fluxed model (erg/s/cm^2/AA)
    fluxed_model = best_model * conv[:, None]

    # Now get the bestest model on the same wavelength vector as the data
    z = 0.0 # redshift of the star, if known.
    a = (1 + z)
    fluxed_model_interp = np.interp(data_wave, libwave * a, fluxed_model[0])
    calibration = fluxed_model_interp / data_flux

    # You probably want to median filter the calibration vector.  # Perhaps
    # after some sigma clipping.  differences on small scales could be due to
    # model imperfections (wrong metallicity, wrong gravity for model, LSF
    # mismatch)
    # you could also fit the calibration vector with a polynomial, taking into
    # account errors
    from scipy.signal import medfilt
    smoothed_calibration = medfilt(calibration, 101)


    import matplotlib.pyplot as pl
    fig, axes = pl.subplots(3, 1, sharex=True, figsize=(13, 11))    
    ax= axes[0]
    ax.plot(data_wave, calibration, label="raw calibration")
    ax.plot(data_wave, smoothed_calibration, label="smoothed calibration")
    ax.legend()
    ax.set_ylabel("actual / input")
    
    ax = axes[1]
    ax.plot(data_wave, data_flux, label="Bino spectrum")
    ax.plot(libwave, fluxed_model[0], label="Fluxed model")
    ax.set_xlim(data_wave.min(), data_wave.max())
    ax.legend()
    ax.set_ylabel("$F_{\lambda} \, (erg/s/cm^2/\AA)$")
    
    ax = axes[2]
    [f.display(ax=ax) for f in filters]
    ax.set_ylabel("Transmission")
    ax.set_label("$\lambda (\AA)$")

    fig.savefig("example_calbino.pdf")
    
