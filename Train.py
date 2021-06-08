"""
Train models given synthetic spectrum
"""
import matplotlib.pyplot as plt
from astropy.io import fits
import tensorflow as tf
from keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn import preprocessing
from tqdm import tqdm_notebook as tqdm
from pickle import dump
from keras.backend import clear_session
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
import statistics
from scipy import interpolate
from scipy.stats import gaussian_kde
from keras.regularizers import l2,l1
import pickle

def train(resolution, output_file):
    # This gets us the wvenumbers we are going to interpolate on
    # We use this spectra because it is a nice looking (can easily see the lines -- not relevant here)
    # spectra at a resolution of 5000 exactly. We want to keep that sampling :)
    ref_spec = fits.open('Reference-Spectrum-R%s.fits'%(str(resolution)))[1].data
    channel = []
    counts = []
    for chan in ref_spec:  # Only want SN3 region
        channel.append(chan[0])
        counts.append(chan[1])
    min_ = np.argmin(np.abs(np.array(channel)-14700))
    max_ = np.argmin(np.abs(np.array(channel)-15600))
    wavenumbers_syn = channel[min_:max_]

    # READ IN SPECTRA
    spectra = pickle.load( open('R%s.pkl'%(str(resolution)), 'rb'))
    counts_axis = [spec[0] for spec in spectra.values()]
    counts = [spec[1] for spec in spectra.values()]
    labels_spec = [spec[2] for spec in spectra.values()]  # Velocity Labels
    broad_spec = [spec[3] for spec in spectra.values()]  # Broadening Labels
    print('  Interpolating Spectra')
    Counts = []
    for channel,spectrum in zip(counts_axis,counts):
        min_ = np.argmin(np.abs(np.array(channel)-14400))
        max_ = np.argmin(np.abs(np.array(channel)-15700))
        f = interpolate.interp1d(channel[min_:max_], spectrum[min_:max_], kind='slinear')
        # Get fluxes of interest
        coun = f(wavenumbers_syn)
        Counts.append(coun)


    # SETUP MACHINE LEARNING ALGORITHM
    activation = 'relu'  # activation function
    initializer = 'normal'  # model initializer
    input_shape = (None, len(Counts[0]), 1)  # shape of input spectra for the input layer
    num_filters = [4,16]  # number of filters in the convolutional layers
    filter_length = [4,2]  # length of the filters
    pool_length = 4  # length of the maxpooling
    num_hidden = [256, 512]  # number of nodes in the hidden layers
    batch_size = 4  # number of data fed into model at once
    max_epochs = 10  # maximum number of interations
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
    metrics_ = ['mse', 'mae', 'accuracy']

    clear_session()
    model = Sequential([
        InputLayer(batch_input_shape=input_shape),
        Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[0], kernel_size=filter_length[0]),
        Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[1], kernel_size=filter_length[1]),
        MaxPooling1D(pool_size=pool_length),
        Flatten(),
        Dropout(0.2),
        Dense(units=num_hidden[0], kernel_initializer=initializer, activation=activation),# kernel_regularizer=l1(0.00005)),
        #Dropout(0.2),
        Dense(units=num_hidden[1], kernel_initializer=initializer, activation=activation),# kernel_regularizer=l1(0.00005)),
        Dense(2, activation='linear'),
    ])

    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_min_delta,
                                           patience=early_stopping_patience, verbose=2, mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, min_delta=reuce_lr_epsilon,
                                      patience=reduce_lr_patience, min_lr=reduce_lr_min, mode='min',
                                      verbose=2)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_)

    # TRAIN NETWORK
    syn_num_pass = len(Counts)
    train_div = int(0.7*syn_num_pass)  # Percent of synthetic data to use as training
    valid_div = int(0.9*syn_num_pass)  # Percent of synthetic data used as training and validation (validation being the difference between this and train_div)
    test_div = int(1.0*syn_num_pass)
    # Set training set
    TraningSet = np.array(Counts[:train_div])
    TraningSet = TraningSet.reshape(TraningSet.shape[0], TraningSet.shape[1], 1)
    Traning_init_labels = np.array((labels_spec[0:train_div], broad_spec[0:train_div])).T
    # Set validation set
    ValidSet = np.array(Counts[train_div:valid_div])
    ValidSet = ValidSet.reshape(ValidSet.shape[0], ValidSet.shape[1], 1)
    Valid_init_labels = np.array((labels_spec[train_div:valid_div], broad_spec[train_div:valid_div])).T
    # Set Model
    print("  Training Network")
    model.fit(TraningSet, Traning_init_labels, validation_data=(ValidSet, Valid_init_labels),
          epochs=max_epochs, verbose=5, callbacks=[reduce_lr,early_stopping])
    model.save('R%s-PREDICTOR-I/'%(str(resolution)))
    print("  Testing Network")
    # TEST NETWORK
    TestSet = np.array(Counts[valid_div:test_div])
    TestSet = TestSet.reshape(TestSet.shape[0], TestSet.shape[1], 1)
    TestSetLabels = np.array((labels_spec[valid_div:test_div], broad_spec[valid_div:test_div])).T
    test_predictions = model.predict(TestSet)
    resids = test_predictions - TestSetLabels
    errs_temp = [100*(test_predictions[i] - TestSetLabels[i])/TestSetLabels[i] for i in range(len(test_predictions))]

    # SAVE RESULTS
    data_wrapped = list(zip(TestSetLabels[:,0], test_predictions[:,0]))
    df = pd.DataFrame(data_wrapped, columns=["True", "Predicted"])
    s = sns.jointplot(x="True", y="Predicted", data=df, kind="kde", fill=True, color=sns.xkcd_rgb["dusty purple"])
    s.set_axis_labels('True [km/s]', 'Predicted [km/s]', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('velocity_KDE_R%s.png'%(str(resolution)))
    plt.clf()

    data_wrapped = list(zip(TestSetLabels[:,1], test_predictions[:,1]))
    df = pd.DataFrame(data_wrapped, columns=["True", "Predicted"])
    s = sns.jointplot(x="True", y="Predicted", data=df, kind="kde", fill=True, color=sns.xkcd_rgb["dusty purple"])
    s.set_axis_labels('True [km/s]', 'Predicted [km/s]', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('broadening_KDE_R%s.png'%(str(resolution)))
    plt.clf()

    residuals = [TestSetLabels[i,0]-test_predictions[i,0] for i in range(len(test_predictions))]
    sns.distplot(residuals, color=sns.xkcd_rgb["dusty purple"], hist=False, kde_kws={"shade": True, 'bw': 0.5})
    plt.xlim(-200,200)
    plt.text(20, np.max(residuals), 'std: %.2f'%statistics.stdev(residuals), fontsize=15)
    plt.xlabel('Velocity (km/s)', fontsize = 20, fontweight='bold')
    plt.ylabel('Density', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Velocity_dist_R%s.png'%(str(resolution)))
    plt.clf()

    residuals = [TestSetLabels[i,1]-test_predictions[i,1] for i in range(len(test_predictions))]
    sns.distplot(residuals, color=sns.xkcd_rgb["dusty purple"], hist=False, kde_kws={"shade": True, 'bw': 0.5})
    plt.text(20, np.max(residuals), 'std: %.2f'%statistics.stdev(residuals), fontsize=15)
    plt.xlabel('Broadening (km/s)', fontsize = 20, fontweight='bold')
    plt.ylabel('Density', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Broad_dist_R%s.png'%(str(resolution)))
    plt.clf()

    return None


resolutions = [2000, 2500, 3000, 3500, 4000, 4500, 5000]
for resolution in resolutions:
    print('We are on resolution %i'%resolution)
    output_file = 'R'+str(resolution)+'.pkl'
    train(resolution, output_file)
