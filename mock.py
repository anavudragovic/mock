#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:49:12 2020

@author: ana
"""

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
import numpy as np 
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.modeling.models import Sersic2D
from astropy.nddata import Cutout2D
import scipy.signal

galaxy = "dark_swarp.fits"
segmentation = "dark_seg_1sig.fits"
psfimg = "dark_swarp_psf.fits"
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
#plt.savefig('sb29_conv.png', format='png',pdi=500)
