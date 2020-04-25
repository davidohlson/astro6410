# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:31:46 2020

@author: DavidOhlson
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table, join, vstack, Column
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Distance, match_coordinates_sky
from astropy import units as u

#%% read in data

t_she = Table.read('data/she2017.fits')
### she has columns 'RAJ2000', 'DEJ2000', and 'Dist'
t_nsa = Table.read('data/nsa_v1_0_1.fits')
### NSA has columns 'RA', 'DEC'

#%% attempt to crossmatch catalogs based on RA and DEC
ra_she = np.array(t_she['RAJ2000'])
dec_she = np.array(t_she['DEJ2000'])
dist_she = np.array(t_she['Dist'])
ra_nsa = np.array(t_nsa['RA'])
dec_nsa = np.array(t_nsa['DEC'])
c = 299792.458
H_0 = 73
dist_nsa = np.array(t_nsa['ZDIST']* (c/H_0))

cat_she = SkyCoord(ra=ra_she*u.degree, dec=dec_she*u.degree, distance=dist_she*u.Mpc)
cat_nsa = SkyCoord(ra=ra_nsa*u.degree, dec=dec_nsa*u.degree, distance=dist_nsa*u.Mpc)

### get NSA matches to She
idx, d2d, d3d = cat_she.match_to_catalog_sky(cat_nsa)
### can constrain matches by max separation
max_sep = 1.0 * u.arcsec
sep_constraint = d2d < max_sep
she_matches = t_she[sep_constraint]
nsa_matches = t_nsa[idx[sep_constraint]]

"""
### check that tables are matched and have same indexing
plt.scatter(she_matches['RAJ2000'], nsa_matches['RA'])
plt.show()
plt.scatter(she_matches['DEJ2000'], nsa_matches['DEC'])
plt.show()
plt.scatter(she_matches['Dist'], nsa_matches['ZDIST']*(c/H_0))
plt.show()
"""

#%% Combine tables

she_matches['NSAID'] = nsa_matches['NSAID']
nsa_she = join(she_matches, nsa_matches, keys='NSAID', join_type='left')

nsa_she.write('data/nsa_she_1.fits', format='fits', overwrite=True)
