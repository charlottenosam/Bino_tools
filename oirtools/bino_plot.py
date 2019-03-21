# =====================================================================
# bino_plot.py
#
# Helper functions to plot nice Binospec spectra 
# (maybe one day generic 2D...)
#
# INPUTS:
#
# HISTORY:
#   Started:                 2019-03-20 C Mason (CfA)
#
# =====================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
import matplotlib as mpl

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, convolve
import photutils

from lmfit.models import GaussianModel, SkewedGaussianModel, Model

from .utils import *

mpl.rcParams['font.size'] = 18

# =====================================================================

c=3.e5

class BinoPlotter(object):
    """
    Plot Binospec spectra
    """

    def __init__(self, spec2D_file_name, psf_ypix=4.):
        """
        Load Binospec 2D spectra (slits) and make nice plots
        """

        self.hdu_list = fits.open(spec2D_file_name)

        # Get basic header params (wavelength) for all targets
        self.header = self.hdu_list[1].header

        wmin_nm = self.header['CRVAL1']
        wmax_nm = self.header['CRVAL1']+(self.header['NAXIS1']-1)*self.header['CDELT1']
        self.waveA   = 10*np.arange(wmin_nm, wmax_nm, step=self.header['CDELT1'])

        self.psf_ypix = 4. # TODO fit the stars in here

        return

    # -----------------------
    def spec2D(self, target, wave_lineA=8300., wave_limA=50, 
                   smooth=None, obposval=None, posy_d=10, 
                plot=True, plottitle=None, cmap='Greys',
                vmin_sig=-0.5, vmax_sig=3., 
                med_subtract=False):

        """
        Load 2D spectrum and plot it

        INPUTS:
            wave_lineA: expected observed wavelength of emission line [A]
            wave_limA:  wavelength limits to plot around [A] (i.e. plot wave_line \pm wave_limA)
            vmin_sig:   image limit [vmin_sig*np.std(image)]
            vmax_sig:   image limit [vmax_sig*np.std(image)]

        """

        # Load 2D spectrum for your target
        hdu   = self.hdu_list[target]
        image = hdu.data

        extent = [self.waveA.min(), self.waveA.max(),
                  0.5, image.shape[0]-0.5]

        # Subtract median 1D spectrum (maybe useful if high background)
        if med_subtract:
            spec_med = np.nanmedian(image, axis=0)
            image -= spec_med
        
        # Smooth by Gaussian kernel (in pixels)        
        if smooth:
            kernel = Gaussian2DKernel(smooth)
            implot = convolve(image, kernel)
        else:
            implot = image       
        
        imsky      = image.copy()
        imsky[10:-10] = np.nan
        noise = np.nanstd(imsky, axis=0)
                
        # Crop the regions outside of the wavelength limits 
        #(this makes finding peaks the image nicer)
        index_centered = np.where(np.abs(self.waveA - wave_lineA) > wave_limA/2.)
        kernel_basic = Gaussian2DKernel(1.)
        imcentered = convolve(image, kernel_basic)
        imcentered[:,index_centered] = np.nan    
        imcentered[:,np.where(noise > 1.*np.nanmedian(noise))] = np.nan    

        # Simple peak finder #TODO make this better (a la Maseda+16?)
        tab = photutils.detection.find_peaks(imcentered, threshold=1.*np.nanstd(imcentered),
                                             box_size=20, npeaks=1, subpixel=False,
                                             border_width=10)  
        # Try to find y positions
        try:
            posy = tab['y_peak']
        except:
            if obposval:
                posy = obposval
            else:
                posy = image.shape[0]/2.

        if plot:

            fig = plt.figure(figsize=(14,5))
            ax = fig.add_subplot(111)
            ax.annotate(plottitle, xy=(0.,0.94), xycoords='axes fraction')
        
            im = ax.imshow(implot, origin='lower', cmap=cmap, aspect='equal', extent=extent,
                          vmin=vmin_sig*np.nanstd(image), vmax=vmax_sig*np.nanstd(image))
        
         
            ax.axvline(wave_lineA, ymin=0.6, ymax=1., lw=5, c='tab:orange')
        
            # Y positions ----------------------
            if obposval:
                ax.axhline(obposval, lw=2,  ls='dashed', c='tab:blue', label='Pipeline')
            # ax.axhline(image.shape[0]/2., lw=2,  ls='dashed', c='tab:blue', label='Slit center')
        
            try:
                ax.plot(self.waveA[tab['x_peak']], tab['y_peak'], 'o', c='r', ms=20, mfc='none', mew=2)
                ax.axhline(tab['y_peak'], lw=2,  ls='dashed', c='tab:red', label='Peak finder')
            except:
                pass  
        
            ax.legend()
            ax.axhline(posy+posy_d/2., lw=2,  ls='dotted', c='tab:red')
            ax.axhline(posy-posy_d/2., lw=2,  ls='dotted', c='tab:red')
            
            ax.set_xlim((wave_lineA - wave_limA), (wave_lineA + wave_limA))
            ax.set_xlabel('Wavelength [$\mathrm{\AA}$]')
            ax.set_ylabel('Position [pixels]')
        
        return image, posy
            

    # =========================

    def spec1D(self, image, wave_lineA=8300., wave_limA=50, StoN=False,
                posy_errmask_pix=[None,None],
                posy_med=None, posy_d=10,
                med_subtract=False, plot=True, smooth=False, plottitle=None):
        """
        Plot 1D spectra

        INPUTS
            posy_med: y position of source
            posy_d:   y length of source aperture
        """

        if med_subtract:
            spec_med = np.nanmedian(image, axis=0)
            image -= spec_med

        # --------------------
        # Center of the source
        if posy_med is None:
            posy_med = int(image.shape[0]/2.)
        else: 
            posy_med = int(posy_med)

        posy_objmask = [int(posy_med-posy_d/2.), int(posy_med-posy_d/2.+1)]
        Npix = posy_objmask[1] - posy_objmask[0]

        # --------------------
        # Make noise spectrum (noise ~ std(flux in 'empty' pixels))
        # Pick a mask in y direction to extract pixel from
        posy_errmask = np.ones(image.shape[0])
        if np.array(posy_errmask_pix).all() is not None:
            # exclude regions around user defined mask
            posy_errmask[:posy_errmask_pix[0],posy_errmask_pix[1]:] = np.nan
            
        # exclude source and 1psf region around it
        posy_errmask[int(posy_objmask[0]-1.*self.psf_ypix):int(posy_objmask[1]+1.*self.psf_ypix)] = np.nan

        # exclude edges
        posy_errmask[:int(self.psf_ypix)]  = np.nan
        posy_errmask[-int(self.psf_ypix):] = np.nan
            
        # --------------------
        # Spectrum = sum flux in posy_objmask
        spec1D = np.nansum(image[posy_objmask[0]:posy_objmask[1]], axis=0)
        
        # Error = std flux in posy_errmask
        err_image = posy_errmask[:,None] * image.copy()
        err1D     = np.sqrt(Npix)*np.nanstd(err_image, axis=0)    
        
        # Flux and noise in wavelength range of interest
        where_inrange = np.where(np.abs(self.waveA - wave_lineA) < wave_limA/2.)
        Sin = spec1D[where_inrange]
        Nin = err1D[where_inrange]

        if plot:
            plt.figure(figsize=(14,5))

            ax = plt.subplot(111)

            if StoN:

                StoN = spec1D/err1D
                ax.plot(self.waveA, StoN, drawstyle='steps-mid', lw=3, label='S/N')
                
                dwaveA= np.median(np.diff(self.waveA))
                # if smooth:
                #     smooth_pix = 3
                #     ax.plot(self.waveA, KT.smooth_boxcar_simple(StoN, smooth_pix), 
                #             lw=3, label='S/N smoothed by %.1f$\mathrm{\AA}$' % np.round(smooth_pix*dwaveA,1))
                
                norm_err1D = err1D/np.nanmax(Nin)     
                ax.fill_between(self.waveA, -3, norm_err1D-3,  
                                color='0.7', zorder=0, label='Scaled noise')
                
                # Gaussian
        #         mod = GaussianModel()
            
        #         x = self.waveA[where_inrange]
        #         y = np.nan_to_num(Sin/Nin)
                
        #         pars = mod.guess(y, x=x)
        #         out = mod.fit(y, pars, x=x)
        # #         print(out.fit_report(min_correl=0.25))
        #         ax.plot(x, out.best_fit)
        #         print(out.params['height'])
                
        #         # half gaussian        
        #         gmodel = Model(half_gauss)
        #         params = gmodel.make_params()
        #         params.add('totalSN', expr='amp * FWHM * 1.06 / 2.')

        #         result = gmodel.fit(y, x=x, params=params,
        #                             amp=out.params['height'].value, 
        #                             cen=out.params['center'].value, 
        #                             FWHM=out.params['fwhm'].value)
        #         print(result.fit_report(min_correl=0.25))
                
        #         ax.plot(x, result.best_fit)
                
                ax.set_ylim(-3, 1.1*np.nanmax(Sin/Nin))                
                ax.set_ylabel('S/N')
                
            else:
                ax.plot(self.waveA, spec1D, c='tab:orange', drawstyle='steps-mid', lw=3, label='Flux')
                ax.fill_between(self.waveA, -err1D, err1D, color='0.7', step='mid', zorder=0, label='Noise')
                ax.set_ylim(-3*np.nanstd(Sin), 3*np.std(Sin))
                
                ax.set_ylabel('Flux [uncalib]')
            
            ax.annotate(plottitle, xy=(0.,0.94), xycoords='axes fraction')
            ax.axvline(wave_lineA, ymin=0.6, ymax=1., lw=2, c='k', ls='dashed', label='Literature $z_\mathrm{spec}$')
            ax.annotate(r'$z_{\mathrm{Ly}\alpha} = %.2f$' % (wave_lineA/1216.-1.), xy=(0.01,0.9), xycoords='axes fraction')
            ax.axhline(0., lw=1, c='0.1', zorder=0)
            
            # km/s axis
            ax_kms = ax.twiny()
            
            ax_kms.plot(wave_to_kms(self.waveA, wave_lineA), spec1D, color='k', lw=0, ls='dashed')

            ax.set_xlim((wave_lineA - wave_limA), (wave_lineA + wave_limA))
            ax_kms.set_xlim(wave_to_kms(np.array(ax.get_xlim()), wave_lineA))

            ax_kms.set_xlabel('Velocity [km/s]', labelpad=10)
            ax.set_xlabel('Wavelength [$\mathrm{\AA}$]')
                        
            ax.legend()

            return 



