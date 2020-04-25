# ASTRO-6410 Final Project
This project utilizes two primary data sets. The first is a catalog of nearby galaxies with Chandra observations, used in 2017 paper by She+. The second is the current NASA-Sloan Atlas (NSA) catalog. Both were downloaded as FITS files, rather than accessed by remote query.

The combine_table.py code crossmatches the two datasets with ```astropy.coordinates```, using RA and Dec values. Tables of NSA and She matches are left joined and saved as a new FITS file. Next, sdss_color_mag.py reads in the file of matches, and calculates a mass value for each galaxy. The calculated mass values are added as a new column to the table.

Finally, linear_regression.py separates galaxies into early- and late-types. Linear regression is run and plotted for log X-ray Luminosity vs log Stellar Mass. Currently, linear regression is run using ```sklearn.linear_model``` without an upper limit, and plotted without inclusion of std. In the future, linear regression with likely be done using ```LINMIX_ERR``` and will include upper limits. Plotting will also display sigma errors, rather than linear fits of errors.
