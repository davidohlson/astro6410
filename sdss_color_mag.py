# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:46:34 2020

@author: DavidOhlson
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table, join, vstack, Column
import matplotlib.pyplot as plt
from astropy import units as u

### read in data

#t_she = Table.read('data/she2017.fits')
t = Table.read('data/nsa_she_1.fits')

#%% create array of petroflux values (FNugriz)
petroflux = t['PETRO_FLUX']

### create arrays for g and i pflux
pflux_g = np.array(petroflux[:, 3])
pflux_i = np.array(petroflux[:, 5])

#%% create pogson mag arrays for g and i bands
pmag_g = 22.5 - (2.5*np.log10(pflux_g))
pmag_i = 22.5 - (2.5*np.log10(pflux_i))

#%% create array of extinction values
### NSA ext from Schlegel, Finkbeiner, and Davis (1997)
extinction = t['EXTINCTION']

### create ext arrays for g&i bands, corrected to Schlafly and Finkbeiner (2011)
ext_g = extinction[:, 3] * 0.86
ext_i = extinction[:, 5] * 0.86

#%% Distance Modulus

dist = t['Dist'] * 10**6  # need to convert Mpc to pc
DM = 5*np.log10(dist) - 5
t['DISTMOD'] = DM

#%% correct pogson magnitude with extinction values and distance modulus

MagCor_g = pmag_g - DM - ext_g
MagCor_i = pmag_i - DM - ext_i

#%% get L values from corrected mags

Msun_g = 5.11
Msun_i = 4.53

L_g = 10** (-0.4* (MagCor_g - Msun_g))
L_i = 10** (-0.4* (MagCor_i - Msun_i))

#%% calculate log(M/L) values

### using BC03 relations and g-i color for log(M/L)
color = MagCor_g - MagCor_i # (g-i)
m_i = 0.979 # Roediger(2015): Table A1
b_i = -0.831

log_MLCR = (m_i * color) + b_i
MLCR = 10**log_MLCR

#%% multiply by L_i to get Mass

mass = MLCR * L_i
t['MASS'] = mass

t.write('data/nsa_she_2.fits', format='fits', overwrite=True)
