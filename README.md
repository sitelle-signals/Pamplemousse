This repository contains Jupyter notebooks that will help users recreate
the training set and convolutional neural network used in the paper entitled
"A Machine Learning Approach to Integral Field Unit Spectroscopy Observations:
I. HII region Kinematics" which can be found at: .

In order to create the synthetic data set, users are required to have ORBS installed.
To install ORBS, please  go to the following github page:
https://github.com/thomasorb/orb

In order to compile and train the convolutional neural network, users are required
to have tensorflow and keras. We strongly suggest created an individual tensorflow
environment using the following conda command: `conda install -c conda-forge tensorflow`
Keras can be readily downloaded with the following line: `pip install keras`.

![Convolutional Neural Network](/images/cnn.png)

To run the network on a SITELLE data cube, alter the path information at the top of the
file "Apply-Network1.py". Then you can simply run 'python Apply-Network1.py' to create
the velocity and broadening fits. Be sure to have the network weight file 'Sitelle_vel_30k_varyRes.h5'
and the 'Reference-Spectrum.fits' file
in the directory defined as 'home_dir' in the python code!


The contents of each individual notebook are described below:
1. 1_Generate-Data.ipynb
  - Familiarize the user with the required ORCS command
  - Download pertinent information from the 3MdB site
  - Construct Synthetic Spectra saved as FITS files

2. 2_CNN_Synthetic.ipynb
  - Read in Synthetic Data
  - Initialize CNN
  - Train and Validate CNN on Synthetic Data
  - Test CNN on Synthetic Data
  - Visualize
