This repository contains Jupyter notebooks that will help users recreate
the training set and convolutional neural network used in the paper entitled
"A Machine Learning Approach to Integral Field Unit Spectroscopy Observations:
I. HII region Kinematics" which can be found at: [https://arxiv.org/abs/2008.08093](https://arxiv.org/abs/2008.08093).

Since the initial creation of this repository, we have submitted two additional papers:
 - A Machine Learning Approach to Integral Field Unit Spectroscopy Observations: II. HII Region Line Ratios
 - A Machine Learning Approach to Integral Field Unit Spectroscopy Observations: III. Disentangling Multiple Component Emission

 The first of these (so the second in the series) was accepted to ApJ in 2021 and can be found at
 [https://arxiv.org/abs/2102.06230](https://arxiv.org/abs/2102.06230). The final paper in the series
 has been accepted to ApJ and can be fouund at [https://arxiv.org/abs/2110.00569](https://arxiv.org/abs/2110.00569).

 We have added the notebooks required to use the networks developed in the two
 subsequent papers in the series.

 For information on how to use these networks please see the extensive documentation
 at our website [https://sitelle-signals.github.io/Pamplemousse/index.html](https://sitelle-signals.github.io/Pamplemousse/index.html).

 We also have incorporated these codes into our general purpose line fitting code Luci -
 [https://github.com/crhea93/LUCI](https://github.com/crhea93/LUCI).


 #--------------------- How to use these codes ------------------------#



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

Please note again that all of this has already been automized in `LUCI`!


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

3. 3_Train-Network_PaperII.ipynb
  - Read in Synthetic data for all three filters
  - Initialize network
  - Train and Validate network on Synthetic Data
  - Test network on Synthetic Data
  - Visualize

4. 4_Train-Network_PaperIII.ipynb
    - Read in Synthetic data
    - Initialize network
    - Train and Validate network on Synthetic Data
    - Test network on Synthetic Data
    - Visualize

5. Dynesty.ipynb
  - Describe Bayesian Method and Functions used in Paper III
  - Visualization of Dynesty's results
