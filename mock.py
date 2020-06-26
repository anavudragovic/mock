#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:49:12 2020

@author: ana
"""

from astropy.io import fits
from astropy.wcs import WCS
#from astropy import units as u
#from photutils import aperture_photometry
#from photutils import CircularAperture, CircularAnnulus
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
import numpy as np 
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.modeling.models import Sersic2D
from astropy.nddata import Cutout2D

#from matplotlib.colors import LogNorm
#from astropy.table import Table
#from photutils.datasets import (make_random_gaussians_table,
#                                make_noise_image,
#                                make_gaussian_sources_image)

import scipy.signal

galaxy = "/Users/ana/Dropbox/dark_swarp.fits"
#segmentation = "/Users/ana/Dropbox/dark_galaxy_seg_1sigma_cutout_no1.fits"
segmentation = "dark_seg_1sig.fits"
psfimg = "/Users/ana/Dropbox/dark_swarp_psf.fits"
psfhdu = fits.open(psfimg)[0]
psf = psfhdu.data
hdu = fits.open(galaxy)[0]
science = hdu.data
hdu_seg = fits.open(segmentation)[0]
seg = hdu_seg.data
#msk = seg.astype(bool)
np.dtype('f8')
wcs = WCS(hdu.header)


# Dark galaxy on cutout image
#x_obj = 1086.7393 
#y_obj = 1106.3072 
# Dark galaxy on the large image
x_obj = 1775.654
y_obj = 2005.323

size_xpix = hdu.header['NAXIS1']
size_ypix = hdu.header['NAXIS2']
#size = size_pix 
position = (x_obj, y_obj)
#print(size)
#center_coordinates = (51, 51)

#science_cps = science/hdu['EXPTIME']
# Create 2D cutout of the image and display it
#cutout = Cutout2D(science, position, size_pix)
plt.figure(figsize=(5,5))
#norm_cutout = simple_norm(cutout.data, 'sqrt', percent=99.)
#plt.xlabel("X [px]")
#plt.ylabel("Y [px]")
#plt.imshow(cutout.data, norm=norm_cutout, cmap = 'Greys')

# We normalize the psf, just in the case it isn't
psf_norm = psf / np.sum(psf)
#plt.figure(figsize=(5,5))
#norm_psf = simple_norm(psf_norm, 'log', percent=99.)
#plt.imshow(psf_norm, norm=norm_psf)

# SB = -2.5 * log10(flux) * m_zp_slope + m_zp_intercept + 5*log10(pix), where:
#  m_zp_slope = 0.94, m_zp_intercept = 30.82 and pix = 0.39
SBlim=29 # variable limit we want to reach
SB0=28.775 # m_zp_intercept + 5*log10(pix) = 30.82 - 2.045 = 28.775
# Bellow we need intensity not the surface brightness
#x,y = np.meshgrid(np.arange(hdu.header['NAXIS1']), np.arange(hdu.header['NAXIS2']))
x,y = np.meshgrid(np.arange(size_xpix), np.arange(size_ypix))


nrows = 3
ncols = 2

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10),
                       squeeze=True)
ax = ax.ravel()
for i in range(nrows*ncols): 
    
    # r_eff["] = 5..105 (step=20) = 5+i*20 -> r_eff[pix] = (5+i*20)/0.39
    mod = Sersic2D(amplitude = 10**(-0.4*(SBlim-SB0)/0.94), r_eff = (5+i*20)/0.39, n=0.6, x_0=x_obj, y_0=y_obj, ellip=0.)

    #image = mod(x,y) + cutout.data
    # We convolve Sersic model with psf 
    image_conv = scipy.signal.fftconvolve(
            np.float64(mod(x,y)), np.float64(psf), mode='same')
    
    # Without convolution with psf: 
    #image = mod(x,y) + cutout.data
    # With psf convolution
    image = image_conv + science#cutout.data

    log_img = np.log10(image)
    #norm = simple_norm(image, 'log', percent=99.)
    ax[i].set_title('Deff='+str(10+i*40)+'" SB='+str(SBlim)+' mag/"^2')
    ax[i].imshow(log_img, cmap='Greys', aspect=1, interpolation='nearest',
               origin='upper', vmin=-1, vmax=2)
plt.tight_layout()
#plt.xscale("log")
plt.savefig('sb29_conv.png', format='png',pdi=500)

print('Deff='+str(10+i*40)+'"')
hdunew = fits.PrimaryHDU(image)
hdunew.header.update(wcs.to_header())
hdunew.writeto('dark_swarp_dwarf.fits', overwrite=True)

from astropy.modeling import models, fitting
#ini_mod = mod(x,y)
#fitter = fitting.LevMarLSQFitter()
#fit_mod = fitter(ini_mod, x, y, maxiter = 1000, estimate_jacobian=False)


ini_mod = Sersic2D(amplitude=10**(-0.4*(SBlim-SB0)/0.94),
r_eff=(5+i*20)/0.39, n=0.6, x_0=x_obj, y_0=y_obj, ellip=0.)

fitter = fitting.LevMarLSQFitter()
goodpix = np.where(seg==0)
fit_mod = fitter(ini_mod, x[goodpix], y[goodpix], image[goodpix], maxiter = 1000, acc=1.0e-7, epsilon=1.0e-6,
estimate_jacobian=False)
norm = simple_norm(image, 'sqrt', percent=95.)
norm_ini = simple_norm(ini_mod(x,y),'sqrt', percent=99.)
norm_fit = simple_norm(fit_mod(x,y),'sqrt', percent=99.)

plt.figure(figsize=(8, 2.5))
plt.subplot(1, 3, 1)
plt.imshow(image,cmap='Greys', norm=norm)
plt.title('Data')
plt.subplot(1, 3, 2)
plt.imshow(ini_mod(x,y),cmap='Greys', norm=norm_ini)
plt.title('Initial guess')
plt.subplot(1, 3, 3)
plt.imshow(fit_mod(x,y),cmap='Greys', norm=norm_fit)
plt.title('Best fit model')

# Generate fake data
np.random.seed(42)
g1 = models.Gaussian1D(1, 0, 0.2)
g2 = models.Gaussian1D(2.5, 0.5, 0.1)
x = np.linspace(-1, 1, 200)
y = g1(x) + g2(x) + np.random.normal(0., 0.2, x.shape)
# Now to fit the data create a new superposition with initial
# guesses for the parameters:
gg_init = models.Gaussian1D(1, 0, 0.1) + models.Gaussian1D(2, 0.5, 0.1)
fitter = fitting.SLSQPLSQFitter()
gg_fit = fitter(gg_init, x, y)
# Plot the data with the best-fit model
plt.figure(figsize=(8,5))
plt.plot(x, y, 'ko')
plt.plot(x, gg_fit(x))
plt.xlabel('Position')
plt.ylabel('Flux')

#vmin, vmax = np.percentile(image, [5, 80])
#print(np.log10(vmin),np.log10(vmax))
#plt.figure()
#plt.imshow(log_img, cmap='Greys', aspect=1, interpolation='nearest',
#           origin='lower', vmin=-1, vmax=2)
#plt.xlabel('x')
#plt.ylabel('y')
#cbar = plt.colorbar()
#cbar.set_label('Log Brightness', rotation=270, labelpad=25)
#cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
#plt.show()
##plt.imshow(np.log10(image), origin='upper')

#largehdu=fits.open("dark_swarp.fits")[0]
#large=largehdu.data
#ncombhdu=fits.open("dark_ncomb.fits")[0]
#ncomb=ncombhdu.data

#sigma2=(large+10000)/ncomb/1.25
#sigma=np.sqrt(sigma2)


#sig=fits.PrimaryHDU(sigma)
#sig.header.update(WCS(largehdu.header).to_header())
#sig.writeto("dark_swarp_sigma.fits",overwrite=True)
