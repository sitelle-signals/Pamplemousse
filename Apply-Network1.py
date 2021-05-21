import os
import numpy as np
import matplotlib.pyplot as plt
import time
import keras
from keras import activations
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from orcs.process import SpectralCube
from astropy.io import fits
from pickle import load
from scipy import interpolate
from tqdm import tqdm

#--------------------------------- INPUTS -------------------------------------#
home_dir = '/home/carterrhea/Documents/Benjamin'   # Location weights file
cube_dir = '/home/carterrhea/Documents/Benjamin'  # Path to data cube
cube_name = 'NGC1275-highres'  # don't add .hdf5 extension
output_dir = '/home/carterrhea/Documents/Benjamin'  # Path to output directory
output_name = 'HighRes'  # output file prefix
deep_file = 'NGC1275_zoom_deep_frame'  # Path to deep image fits file: required for header -- don't include .fits extension
#------------------------------------------------------------------------------#


os.chdir(home_dir)

model = keras.models.load_model('R7000-FULL')
# Load cube
cube = SpectralCube(cube_dir+'/'+cube_name+'.hdf5')
# We first need to extract a random spectrum to get the x-axis (Wavenumbers) for the observation
axis, spectrum = cube.extract_spectrum(1000, 1000, 1)
# We need to find the min and max wavenumber that match our min/max from our synthetic spectra
min_ = np.argmin(np.abs(np.array(axis)-14400))
max_ = np.argmin(np.abs(np.array(axis)-15700))
# LOAD IN SPECTRAL INFORMATION
x_min = 0
x_max = cube.shape[0]
y_min = 0
y_max = cube.shape[1]
# Now pull data
dat = cube.get_data(x_min,x_max,y_min,y_max,min_,max_)
# Open our nominal spectrum and get its wavenumbers
ref_spec = fits.open('Reference-Spectrum.fits')
wavenumbers_syn = []
spec_ref_axis = [val[0] for val in ref_spec[1].data]
min_w = np.argmin(np.abs(np.array(spec_ref_axis)-14700))  # For intepolation purposes
max_w =np.argmin(np.abs(np.array(spec_ref_axis)-15600))
spec_ref_axis = np.array(spec_ref_axis)[min_w:max_w]
for i in range(len(spec_ref_axis)):
    wavenumbers_syn.append(spec_ref_axis[i])
ct = 0
# Interpolate background (sky) onto the correct axis
f_sky = interpolate.interp1d(axis[min_:max_], sky[min_:max_], kind='slinear')
sky_int = np.real(f_sky(wavenumbers_syn))
# Create empty 2D numpy array corresponding to x/y pixels
vels = np.zeros((x_max, y_max))
broads = np.zeros((x_max, y_max))
start_time = time.time()
for i in tqdm(range(x_max-x_min)):
    vels_local = []
    broads_local = []
    for j in range(y_max-y_min):
        counts = np.zeros((1,len(wavenumbers_syn)))
        # Interpolate to get only points equivalent to what we have in our synthetic data
        f = interpolate.interp1d(axis[min_:max_], np.real(dat[i,j]), kind='slinear')
        # Get fluxes of interest
        coun = f(wavenumbers_syn) - sky_int
        max_con = np.max(coun)
        coun = [con/max_con for con in coun]
        counts[0] = coun
        # Get into correct format for predictions
        Spectrum = counts
        Spectrum = Spectrum.reshape(1, Spectrum.shape[1], 1)
        predictions = model(Spectrum, training=False)
        vels_local[j] = predictions[0][0]
        broads_local[j] = predictions[0][1]
    vels[i] = vels_local
    broads[i] = broads_local
print(time.time()-start_time)

# Save as Fits File
reg_fits = fits.open(deep_file+'.fits')
header = reg_fits[0].header
fits.writeto(output_dir+'/'+output_name+'_vel.fits', vels.T, header, overwrite=True)
fits.writeto(output_dir+'/'+output_name+'_broad.fits', broads.T, header, overwrite=True)
