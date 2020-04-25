# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:48:49 2020

@author: DavidOhlson
"""

import numpy as np
from astropy.table import Table, join, vstack, Column
import matplotlib.pyplot as plt
import pdb
from scipy.stats import norm
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

plt.close('all')
ufontsize=12
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
plt.rc('savefig', dpi=300)

#%% create masks for late and early types with Lx detections

a=Table.read('data/nsa_she_2.fits')
det=np.isfinite(a['b_Flux'])
dets0=(np.isfinite(a['b_Flux']) & ([('0' in s) for s in a['HType']]))
detell=(np.isfinite(a['b_Flux']) & ([('E' in s) for s in a['HType']]))
detearly=(dets0 | detell)

s0=np.ma.make_mask([('0' in s) for s in a['HType']])
ell=np.ma.make_mask(([('E' in s) for s in a['HType']]))
early=(s0 | ell)

alls=np.ma.make_mask([('S' in s) for s in a['HType']])
spiral=((alls == True) & (np.ma.make_mask(s0) == False))
detspiral=((alls == True) & (np.ma.make_mask(s0) == False) & (np.isfinite(a['b_Flux'])))

#%% apply masks to data to create data arrays
early = a[detearly]

early_logm = np.log10(early['MASS'])
early_loglx = early['logLX']
early_loglx_u = early['b_loglx_lc']
early_loglx_l = early['b_logLX']

late = a[detspiral]
late_logm = np.log10(late['MASS'])
late_loglx = late['logLX']
late_loglx_u = late['b_loglx_lc']
late_loglx_l = late['b_logLX']

#%% run linear regression on data

logm_test = np.linspace(6.5,12.5,100)
model = linear_model.LinearRegression()

### linear regression for early types
model.fit(early_logm[:, np.newaxis], early_loglx[:, np.newaxis])
earlylx_model = model.predict(logm_test[:, np.newaxis])
print('Early Type Coefficient: \n', model.coef_)

model.fit(early_logm[:, np.newaxis], early_loglx_u[:, np.newaxis]) #fit from upper limit
earlylxu_model = model.predict(logm_test[:, np.newaxis])

model.fit(early_logm[:, np.newaxis], early_loglx_l[:, np.newaxis]) #fit from lower limit
earlylxl_model = model.predict(logm_test[:, np.newaxis])



#%% linear regression for late types
model.fit(late_logm[:, np.newaxis], late_loglx[:, np.newaxis])
latelx_model = model.predict(logm_test[:, np.newaxis])
print('Late Type Coefficient: \n', model.coef_)

mask = ~np.isnan(late_logm) & ~np.isnan(late_loglx_u) # create mask for nan in limits

model.fit(late_logm[:, np.newaxis][mask], late_loglx_u[:, np.newaxis][mask]) #fit from upper limit
latelxu_model = model.predict(logm_test[:, np.newaxis])

model.fit(late_logm[:, np.newaxis][mask], late_loglx_l[:, np.newaxis][mask]) #fit from lower limit
latelxl_model = model.predict(logm_test[:, np.newaxis])

#%% plot data

plt.scatter(early_logm, early_loglx, color='orange')
plt.scatter(late_logm, late_loglx, color='teal')
plt.xlim(7,12)
plt.ylim(35,42)
plt.axes().minorticks_on()
plt.title('X-ray Luminosity vs. stellar mass', fontsize=14)
plt.xlabel('$log M{star} [M_{\odot}]$', fontsize=12)
plt.ylabel('$log L_{x} [erg s^{-1}]$', fontsize=12)
plt.savefig('plots/nsa_Lxm_data.png')
plt.show()

plt.scatter(early_logm, early_loglx, color='red')
plt.plot(logm_test, earlylx_model, color='black')
plt.fill_between(logm_test, earlylxu_model.ravel(), earlylxl_model.ravel(), color='orange', alpha=0.2)
plt.xlim(7,12)
plt.ylim(35,42)
plt.axes().minorticks_on()
plt.text(7.5, 35.5, 'Coefficient: 0.279', fontsize=14)
plt.title('Early-Type: X-ray Luminosity vs. stellar mass', fontsize=14)
plt.xlabel('$log M{star} [M_{\odot}]$', fontsize=12)
plt.ylabel('$log L_{x} [erg s^{-1}]$', fontsize=12)
plt.savefig('plots/nsa_early_LR.png')
plt.show()

plt.scatter(late_logm, late_loglx, color='blue')
plt.plot(logm_test, latelx_model, color='black')
plt.fill_between(logm_test, latelxu_model.ravel(), latelxl_model.ravel(), color='teal', alpha=0.2)
plt.xlim(7,12)
plt.ylim(35,42)
plt.axes().minorticks_on()
plt.text(7.5, 35.5, 'Coefficient: 0.494', fontsize=14)
plt.title('Late-Type: X-ray Luminosity vs. stellar mass', fontsize=14)
plt.xlabel('$log M [M_{\odot}]$', fontsize=12)
plt.ylabel('$log L_{x} [erg s^{-1}]$', fontsize=12)
plt.savefig('plots/nsa_late_LR.png')
plt.show()
