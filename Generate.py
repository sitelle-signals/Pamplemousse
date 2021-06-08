"""
Generate synthetic SITELLE data following the procedure outlined in paper 1 for various resolutions
"""

import orb.fit
import pylab as pl
import numpy as np
from orb.core import Lines
import random
from astropy.io import fits
import datetime
from tqdm import tqdm_notebook as tqdm
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle


def create_spectra(resolution, output_file, ampls):
    spectra_data = {}  # {spectrum_ct: [spectrum, velocity, broadening, resolution, SNR]}
    # Set Input Parameters
    #output_dir = 'R2000/'
    # Set observation parameters
    step = 2943
    order = 8
    #resolution = 2000
    vel_num = 5000  # Number of Velocity Values Sampled
    broad_num = 1000  # Number of Broadening Values Sampled
    theta_num = 1  # Number of Theta Values Sampled
    num_syn = 50000  # Number of Synthetic Spectra

    # Sample velocity
    vel_ = np.random.uniform(-200,3000,vel_num)
    # Sample broadening
    broad_ = np.random.uniform(10,100,broad_num)
    # Same resolution
    res_ = np.random.uniform(resolution-300, resolution, 300)

    # Now we need to get our emission lines
    halpha_cm1 = Lines().get_line_cm1('Halpha')
    NII6548_cm1 = Lines().get_line_cm1('[NII]6548')
    NII6583_cm1 = Lines().get_line_cm1('[NII]6583')
    SII6716_cm1 = Lines().get_line_cm1('[SII]6716')
    SII6731_cm1 = Lines().get_line_cm1('[SII]6731')
    # We now can model the lines. For the moment, we will assume all lines have the same velocity and broadening
    # Do this for randomized combinations of vel_ and broad_
    for spec_ct in range(num_syn):
        if spec_ct%10000 == 0:
            print("    We are on spectrum number %i"%spec_ct)
        pick_new = True
        # Randomly select velocity and broadening parameter and theta
        velocity = random.choice(vel_)
        broadening = random.choice(broad_)
        resolution = random.choice(res_)
        theta = 11.96
        axis_corr = 1 / np.cos(np.deg2rad(theta))
        # Randomly Select a M3db simulation
        #while pick_new:
        sim_num = random.randint(0,len(ampls)-1)
        sim_vals = ampls.iloc[sim_num]
        # Now add all of the lines with amplitudes relative to Halpha...
        spectrum = orb.fit.create_cm1_lines_model([halpha_cm1], [sim_vals['h1']/sim_vals['h1']],
                                                  step, order, resolution, theta, fmodel='sincgauss',
                                                  sigma=broadening, vel=velocity)
        spectrum += orb.fit.create_cm1_lines_model([NII6548_cm1], [sim_vals['n1']/sim_vals['h1']],
                                                  step, order, resolution, theta, fmodel='sincgauss',
                                                  sigma=broadening, vel=velocity)
        spectrum += orb.fit.create_cm1_lines_model([NII6583_cm1], [sim_vals['n2']/sim_vals['h1']],
                                                  step, order, resolution, theta, fmodel='sincgauss',
                                                  sigma=broadening, vel=velocity)
        spectrum += orb.fit.create_cm1_lines_model([SII6716_cm1], [sim_vals['s1']/sim_vals['h1']],
                                                  step, order, resolution, theta, fmodel='sincgauss',
                                                  sigma=broadening, vel=velocity)
        spectrum += orb.fit.create_cm1_lines_model([SII6731_cm1], [sim_vals['s2']/sim_vals['h1']],
                                                  step, order, resolution, theta, fmodel='sincgauss',
                                                  sigma=broadening, vel=velocity)
        # We now add noise
        SNR = np.random.uniform(3,30)
        spectrum += np.random.normal(0.0,1.0/SNR,spectrum.shape)
        spectrum_axis = orb.utils.spectrum.create_cm1_axis(np.size(spectrum), step, order, corr=axis_corr)
        min_ = np.argmin(np.abs(np.array(spectrum_axis)-14400))
        max_ = np.argmin(np.abs(np.array(spectrum_axis)-15700))
        spectrum = spectrum[min_:max_]  ## at R = 5000 -> spectrum[214:558]
        spectrum_axis = spectrum_axis[min_:max_]
        # Normalize Spectrum Values
        spec_max = np.max(spectrum)
        spectrum = [spec_/spec_max for spec_ in spectrum]
        spectra_data[sim_num] = [spectrum_axis, spectrum, velocity, broadening, resolution, SNR]
    # Save to pickle
    pickle.dump(spectra_data, open(output_file, 'wb'))



# We must alo get our flux values from 3mdb
# First we load in the parameters needed to login to the sql database
MdB_HOST='3mdb.astro.unam.mx'
MdB_USER='OVN_user'
MdB_PASSWD='oiii5007'
MdB_PORT='3306'
MdB_DBs='3MdBs'
MdB_DBp='3MdB'
MdB_DB_17='3MdB_17'
# Now we connect to the database
co = pymysql.connect(host=MdB_HOST, db=MdB_DB_17, user=MdB_USER, passwd=MdB_PASSWD)
# Now we get the lines we want
ampls = pd.read_sql("select H__1_656281A as h1, N__2_654805A as n1, N__2_658345A as n2, \
                  S__2_673082A  as s1, S__2_671644A as s2,   \
                  com1 as U, com2 as gf, com4 as ab \
                  from tab_17 \
                  where ref = 'BOND'"
                    , con=co)
resolutions = [2000, 2500, 3000, 3500, 4000, 4500, 5000]
for resolution in resolutions:
    print('We are on resolution %i'%resolution)
    output_file = 'R'+str(resolution)+'.pkl'
    create_spectra(resolution, output_file, ampls)
