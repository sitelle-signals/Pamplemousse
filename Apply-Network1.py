from keras.models import load_model
from orcs.process import SpectralCube
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import activations
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from astropy.io import fits
from pickle import load
from tqdm import tqdm_notebook as tqdm
from scipy import interpolate
import time

#--------------------------------- INPUTS -------------------------------------#
home_dir = '/path/to/main/directory/containing/data'   # Location weights file
cube_dir = '/path/to/cubes'  # Path to data cube
cube_name = 'cube_name'  # don't add .hdf5 extension
output_dir = '/path/to/output'  # Path to output directory
output_name = 'output_name'  # output file prefix
deep_file = '/full/path/to/deep/fits'  # Path to deep image fits file: required for header 
#------------------------------------------------------------------------------#


os.chdir(home_dir)

# Load Scaler
#scaler = load(open('scaler_robust.pkl', 'rb'))
# Model Parameters
activation = 'relu'  # activation function
initializer = 'he_normal'  # model initializer
input_shape = (None, 365, 1)  # shape of input spectra for the input layer
num_filters = [4,16]  # number of filters in the convolutional layers
filter_length = [8,4]  # length of the filters
pool_length = 2  # length of the maxpooling
num_hidden = [256,128]  # number of nodes in the hidden layers
batch_size = 8  # number of data fed into model at once
max_epochs = 25  # maximum number of interations
lr = 0.0007  # initial learning rate
beta_1 = 0.9  # exponential decay rate  - 1st
beta_2 = 0.999  # exponential decay rate  - 2nd
optimizer_epsilon = 1e-08  # For the numerical stability
early_stopping_min_delta = 0.0001
early_stopping_patience = 4
reduce_lr_factor = 0.5
reuce_lr_epsilon = 0.009
reduce_lr_patience = 2
reduce_lr_min = 0.00008
loss_function = 'mean_squared_error'
metrics = ['accuracy', 'mae']

optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_min_delta,
                                       patience=early_stopping_patience, verbose=2, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=reuce_lr_epsilon,
                                  patience=reduce_lr_patience, min_lr=reduce_lr_min, mode='min', verbose=2)


# Define Model
model = Sequential([
    InputLayer(batch_input_shape=input_shape),
    Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[0], kernel_size=filter_length[0]),
    Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[1], kernel_size=filter_length[1]),
    MaxPooling1D(pool_size=pool_length),
    Flatten(),
    Dropout(0.2),
    Dense(units=num_hidden[0], kernel_initializer=initializer, activation=activation),
    Dense(units=num_hidden[1], kernel_initializer=initializer, activation=activation),
    Dense(2),
])
model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
#Load weights
model.load_weights('Sitelle_vel_30k_varyRes.h5')
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
# Create empty 2D numpy array corresponding to x/y pixels
vels = np.zeros((x_max, y_max))
broads = np.zeros((x_max, y_max))
start_time = time.time()
for i in range(x_max-x_min):
    for j in range(y_max-y_min):
        counts = np.zeros((1,len(wavenumbers_syn)))
        # Interpolate to get only points equivalent to what we have in our synthetic data
        f = interpolate.interp1d(axis[min_:max_], dat[i,j], kind='slinear')
        # Get fluxes of interest
        coun = f(wavenumbers_syn)
        max_con = np.max(coun)
        coun = [con/max_con for con in coun]
        counts[0] = coun
        # Get into correct format for predictions
        Spectrum = counts
        Spectrum = Spectrum.reshape(1, Spectrum.shape[1], 1)
        predictions = model.predict(Spectrum)
        vels[i][j] = predictions[0][0]
        broads[i][j] = predictions[0][1]
print(time.time()-start_time)

# Save as Fits File
reg_fits = fits.open(deep_file'.fits')
header = reg_fits[0].header
fits.writeto(output_dir+'/'+output_name+'_vel.fits', vels.T, header, overwrite=True)
fits.writeto(output_dir+'/'+output_name+'_broad.fits', broads.T, header, overwrite=True)
