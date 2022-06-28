import orb.fit
import pylab as pl
import numpy as np
from orb.core import Lines
import random
from astropy.io import fits
import datetime
import pymysql
import os
import pandas as pd
import warnings
from AmpToFlux import ampToFlux
import scipy.special as ss
warnings.filterwarnings("ignore")
from astropy.io import fits
import matplotlib.pyplot as pl
import dill as pickle
import numpy as np
from joblib import Parallel, delayed
from redden import *
from tqdm import tqdm


"""
DEREDENNING: redden_line requires the first argument to be in angstroms so 1e7/wavenumber
"""
# ------------------------------- FUNCTIONS --------------------------#
"""
Calculate the Flux of an emission line for SITELLE given the amplitude, broadening,
resolution, and wavenumber
"""
def ampToFlux(ampl, broad, res, wvn):
    """
    ampl - amplitude
    broad - broadening
    res - spectral resolution
    wvn - wavenumber of emission line (cm^-1)
    """
    num = np.sqrt(2*np.pi)*ampl*broad
    den = ss.erf((2*wvn*broad)/(1.20671*res))
    flux = num/den
    return flux

def fluxToAmp(flux, broad, res, wvn):
    num = ss.erf((2*wvn*broad)/(1.20671*res))*flux
    den = np.sqrt(2*np.pi)*broad
    ampl = num/den
    return ampl
# ------------------- IMPORTS -----------------------------------------#
# Set Input Parameters
name = '10k-red-broad'
output_dir = '/home/carterrhea/Dropbox/CFHT/Analysis-Paper2/SyntheticData/'+name+'/'
plot_dir = '/home/carterrhea/Dropbox/CFHT/Analysis-Paper2/SyntheticData/Plots/'+name+'/'
# Set observation parameters
step = 2943
order = 8
resolution_SN3 = 5000
resolution_SN2 = 1000
vel_num = 2000  # Number of Velocity Values Sampled
broad_num = 1000  # Number of Broadening Values Sampled
theta_num = 1  # Number of Theta Values Sampledr
num_syn = 50000  # Number of Synthetic Spectra
#-----------------------------------------------------------------------------------#
# Randomize values
# Sample theta parameter
thetas_ = np.random.uniform(11.96,11.96,theta_num)#11.8,19.6,theta_num)
# Sample velocity
vel_ = np.random.uniform(-200,200,vel_num)
# Sample broadening
broad_ = np.random.uniform(10,50,broad_num)
# Same resolution
res_SN3 = np.random.uniform(resolution_SN3-200, resolution_SN3, 200)
res_SN2 = np.random.uniform(resolution_SN2-100, resolution_SN2, 100)

# Now we need to get our emission lines
halpha_cm1 = Lines().get_line_cm1('Halpha')
NII6548_cm1 = Lines().get_line_cm1('[NII]6548')
NII6583_cm1 = Lines().get_line_cm1('[NII]6583')
SII6716_cm1 = Lines().get_line_cm1('[SII]6716')
SII6731_cm1 = Lines().get_line_cm1('[SII]6731')
OIII5007_cm1 = Lines().get_line_cm1('[OIII]5007')
OIII4959_cm1 = Lines().get_line_cm1('[OIII]4959')
OII3726_cm1 = Lines().get_line_cm1('[OII]3726')
OII3729_cm1 = Lines().get_line_cm1('[OII]3729')
hbeta_cm1 = Lines().get_line_cm1('Hbeta')
print('# -- Connecting to 3MdB -- #')
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
print('# -- Obtaining Amplitudes -- #')
# Now we get the lines we want
print('# -- HII --#')
HII_ampls = pd.read_sql("select H__1_656281A as ha, N__2_654805A as n1, N__2_658345A as n2, \
                  S__2_673082A  as s2, S__2_671644A as s1, O__2_372603A as O2, O__3_500684A as O3, O__3_495891A as O3_2, \
                  H__1_486133A as hb, O__2_372881A as O2_2  \
                  from tab_17 \
                  where ref = 'BOND'"
                    , con=co)
print('# -- PNe --#')
PNe_ampls = pd.read_sql("select H__1_656281A as ha, O__3_500684A as O3, N__2_658345A as n2, N__2_654805A as n1, \
                  S__2_673082A  as s2, S__2_671644A as s1, H__1_486133A as hb, O__1_630030A as O1,  O__3_495891A as O3_2,   \
                  O__2_372603A as O2, O__2_372881A as O2_2 \
                  from tab_17 \
                  where ref = 'PNe_2020'"
                    , con=co)
co = pymysql.connect(host=MdB_HOST, db=MdB_DBs, user=MdB_USER, passwd=MdB_PASSWD)
print('# -- SNR --#')
SNR_ampls = pd.read_sql("""SELECT shock_params.shck_vel AS shck_vel,
                         shock_params.mag_fld AS mag_fld,
                         emis_VI.NII_6548 AS n1,
                         emis_VI.NII_6583 AS n2,
                         emis_VI.HI_6563 AS ha,
                         emis_VI.OIII_5007 AS O3,
                         emis_VI.OIII_4959 as O3_2,
                         emis_VI.HI_4861 AS hb,
                         emis_VI.SII_6716 AS s1,
                         emis_VI.SII_6731 AS s2,
                         emis_VI.OII_3726 AS O2,
                         emis_VI.OII_3729 as O2_2
                         FROM shock_params
                         INNER JOIN emis_VI ON emis_VI.ModelID=shock_params.ModelID
                         INNER JOIN abundances ON abundances.AbundID=shock_params.AbundID
                         WHERE emis_VI.model_type='shock'
                         AND abundances.name='Allen2008_Solar'
                         ORDER BY shck_vel, mag_fld;""", con=co)
# We will now filter out values that are non representative of our SIGNALS sample
print('# -- Starting Creation of Spectra -- #')
## We now can model the lines. For the moment, we will assume all lines have the same velocity and broadening
# Do this for randomized combinations of vel_ and broad_
ct = 0
nm_laser = 543.5 # wavelength of the calibration laser, in fact it can be any real positive number (e.g. 1 is ok)
#for spec_ct in range(num_syn):
def create_spectrum(spec_ct, ampls, ampls_str):
    spectrum = None  # Intialize
    pick_new = True
    # Randomly select velocity and broadening parameter and theta
    velocity = random.choice(vel_)
    broadening = random.choice(broad_)
    resolution_sn3 = random.choice(res_SN3)
    resolution_sn2 = random.choice(res_SN2)  # SN2 and SN1 since they have the same resoution
    theta = 11.96#random.choice(thetas_)
    axis_corr =  orb.utils.spectrum.theta2corr(theta)

    # Randomly Select a M3db simulation
    while pick_new:
        sim_num = random.randint(0,len(ampls)-1)
        sim_vals = ampls.iloc[sim_num]
        # Redden Spectra
        Balmer_dec = np.random.uniform(2,6)  # Randomly select Balmer Decrement Value
        # Calculate tau value
        tauV = redden_tau(Balmer_dec)
        # Calculate Halpha redenned for comparisons
        Ha_red_flux = redden_line(1e7/halpha_cm1, ampToFlux(sim_vals['ha'], broadening, resolution_sn3, halpha_cm1), tauV)
        # Only pick simulation if the n1,n2,s1 and s2 lines are 12% that of Halpha -> SNR 3
        # Must be checked verse redenned values!!! Otherwise lines will be lost in the noise!
        pass_ct = 0  # Number of lines for which the criteria is meet
        for line,name in zip([sim_vals['n2'],sim_vals['s1'],sim_vals['s2'],sim_vals['O2'], sim_vals['O2_2'],sim_vals['O3'],sim_vals['hb']],[NII6583_cm1, SII6716_cm1, SII6731_cm1, OII3726_cm1, OII3729_cm1, OIII5007_cm1, hbeta_cm1]):
            if line in [sim_vals['n2'], sim_vals['s1'], sim_vals['s2']]:
                line_reddened = redden_line(1e7/name, ampToFlux(line, broadening, resolution_sn3, name), tauV)
            else:
                line_reddened = redden_line(1e7/name, ampToFlux(line, broadening, resolution_sn2, name), tauV)
            if line_reddened/Ha_red_flux > 0.12:  # If greater than 12%
                pass_ct += 1  # Another line has passed
            else:
                break  # No need to check any others
        if pass_ct == 7:  # If all lines and ratios pass
            pick_new = False  # Don't loop!

    Ha_red_flux = redden_line(1e7/halpha_cm1, ampToFlux(sim_vals['ha'], broadening, resolution_sn3, halpha_cm1), tauV)
    Ha_red_ampl = fluxToAmp(Ha_red_flux, broadening, resolution_sn3, halpha_cm1)
    n1_red_flux = redden_line(1e7/NII6548_cm1, ampToFlux(sim_vals['n1'], broadening, resolution_sn3, NII6548_cm1), tauV)
    n1_red_ampl = fluxToAmp(n1_red_flux, broadening, resolution_sn3, NII6548_cm1)
    n2_red_flux = redden_line(1e7/NII6583_cm1, ampToFlux(sim_vals['n2'], broadening, resolution_sn3, NII6583_cm1), tauV)
    n2_red_ampl = fluxToAmp(n2_red_flux, broadening, resolution_sn3, NII6583_cm1)
    s1_red_flux = redden_line(1e7/SII6716_cm1, ampToFlux(sim_vals['s1'], broadening, resolution_sn3, SII6716_cm1), tauV)
    s1_red_ampl = fluxToAmp(s1_red_flux, broadening, resolution_sn3, SII6716_cm1)
    s2_red_flux = redden_line(1e7/SII6731_cm1, ampToFlux(sim_vals['s2'], broadening, resolution_sn3, SII6731_cm1), tauV)
    s2_red_ampl = fluxToAmp(s2_red_flux, broadening, resolution_sn3, SII6731_cm1)
    O2_red_flux = redden_line(1e7/OII3726_cm1, ampToFlux(sim_vals['O2'], broadening, resolution_sn2, OII3726_cm1), tauV)
    O2_red_ampl = fluxToAmp(O2_red_flux, broadening, resolution_sn2, OII3726_cm1)
    O2_2_red_flux = redden_line(1e7/OII3729_cm1, ampToFlux(sim_vals['O2_2'], broadening, resolution_sn2, OII3729_cm1), tauV)
    O2_2_red_ampl = fluxToAmp(O2_2_red_flux, broadening, resolution_sn2, OII3729_cm1)
    O3_red_flux = redden_line(1e7/OIII5007_cm1, ampToFlux(sim_vals['O3'], broadening, resolution_sn2, OIII5007_cm1), tauV)
    O3_red_ampl = fluxToAmp(O3_red_flux, broadening, resolution_sn2, OIII5007_cm1)
    O3_2_red_flux = redden_line(1e7/OIII4959_cm1, ampToFlux(sim_vals['O3_2'], broadening, resolution_sn2, OIII4959_cm1), tauV)
    O3_2_red_ampl = fluxToAmp(O3_2_red_flux, broadening, resolution_sn2, OIII4959_cm1)
    Hb_red_flux = redden_line(1e7/hbeta_cm1, ampToFlux(sim_vals['hb'], broadening, resolution_sn2, hbeta_cm1), tauV)
    Hb_red_ampl = fluxToAmp(Hb_red_flux, broadening, resolution_sn2, hbeta_cm1)
    # Now add all of the lines...

    Halpha_spec = orb.fit.create_cm1_lines_model([halpha_cm1], [Ha_red_ampl],
                                              step, order, resolution_sn3, theta, fmodel='sincgauss',
                                              sigma=broadening, vel=velocity)
    n1_spec = orb.fit.create_cm1_lines_model([NII6548_cm1], [n1_red_ampl],
                                              step, order, resolution_sn3, theta, fmodel='sincgauss',
                                              sigma=broadening, vel=velocity)
    n2_spec = orb.fit.create_cm1_lines_model([NII6583_cm1], [n2_red_ampl],
                                              step, order, resolution_sn3, theta, fmodel='sincgauss',
                                              sigma=broadening, vel=velocity)
    s1_spec = orb.fit.create_cm1_lines_model([SII6716_cm1], [s1_red_ampl],
                                              step, order, resolution_sn3, theta, fmodel='sincgauss',
                                              sigma=broadening, vel=velocity)
    s2_spec = orb.fit.create_cm1_lines_model([SII6731_cm1], [s2_red_ampl],
                                              step, order, resolution_sn3, theta, fmodel='sincgauss',
                                              sigma=broadening, vel=velocity)
    OII_spec = orb.fit.create_cm1_lines_model([OII3726_cm1], [O2_red_ampl],
                                              1647, order, resolution_sn2, theta, fmodel='sincgauss',
                                              sigma=broadening, vel=velocity)
    OII_2_spec = orb.fit.create_cm1_lines_model([OII3729_cm1], [O2_2_red_ampl],
                                              1647, order, resolution_sn2, theta, fmodel='sincgauss',
                                              sigma=broadening, vel=velocity)
    OIII_spec = orb.fit.create_cm1_lines_model([OIII5007_cm1], [O3_red_ampl],
                                              1680, 6, resolution_sn2, theta, fmodel='sincgauss',
                                              sigma=broadening, vel=velocity)
    OIII_2_spec = orb.fit.create_cm1_lines_model([OIII4959_cm1], [O3_2_red_ampl],
                                              1680, 6, resolution_sn2, theta, fmodel='sincgauss',
                                              sigma=broadening, vel=velocity)
    Hbeta_spec = orb.fit.create_cm1_lines_model([hbeta_cm1], [Hb_red_ampl],
                                              1680, 6, resolution_sn2, theta, fmodel='sincgauss',
                                              sigma=broadening, vel=velocity)
    # Let's add all of our components together
    spectrum_SN3 = Halpha_spec + n1_spec + n2_spec + s1_spec + s2_spec
    spectrum_SN2 = OIII_spec + OIII_2_spec + Hbeta_spec
    spectrum_SN1 = OII_spec + OII_2_spec
    # Normalize Spectrum Values to Halpha to add noise because we want the noise to be wrt Halpha
    spec_max = np.max(spectrum_SN3)
    spectrum_SN3 = np.array([spec_/spec_max for spec_ in spectrum_SN3])
    spectrum_SN2 = np.array([spec_/spec_max for spec_ in spectrum_SN2])
    spectrum_SN1 = np.array([spec_/spec_max for spec_ in spectrum_SN1])
    # We now add noise
    SNR = np.random.uniform(5,30)
    spectrum_SN3 += np.random.normal(0.0,1.0/SNR,spectrum_SN3.shape)
    spectrum_SN2 += np.random.normal(0.0,1.0/SNR,spectrum_SN2.shape)
    spectrum_SN1 += np.random.normal(0.0,1.0/SNR,spectrum_SN1.shape)
    spectrum_axis_SN3 = orb.utils.spectrum.create_cm1_axis(np.size(spectrum_SN3), step, order, corr=axis_corr)
    spectrum_axis_SN2 = orb.utils.spectrum.create_cm1_axis(np.size(spectrum_SN2), 1680, 6, corr=axis_corr)
    spectrum_axis_SN1 = orb.utils.spectrum.create_cm1_axis(np.size(spectrum_SN1), 1647, order, corr=axis_corr)

    # Now normalize to the max value. This must be done for later comparison with real data
    # Since we are normalizing the spectra to the max value
    spec_max = np.max(spectrum_SN3)
    if np.max(spectrum_SN2) > spec_max: spec_max = np.max(spectrum_SN2)
    if np.max(spectrum_SN1) > spec_max: spec_max = np.max(spectrum_SN1)
    spectrum_SN3 = np.array([spec_/spec_max for spec_ in spectrum_SN3])
    spectrum_SN2 = np.array([spec_/spec_max for spec_ in spectrum_SN2])
    spectrum_SN1 = np.array([spec_/spec_max for spec_ in spectrum_SN1])

    # Correct flux for each filter
    #for ct,wl in enumerate(spectrum_axis_SN1):
    #    spectrum_SN1[ct] = redden_line(1e7/wl, spectrum_SN1[ct], tauV)
    #for ct,wl in enumerate(spectrum_axis_SN2):
    #    spectrum_SN2[ct] = redden_line(1e7/wl, spectrum_SN2[ct], tauV)
    #for ct,wl in enumerate(spectrum_axis_SN3):
    #    spectrum_SN3[ct] = redden_line(1e7/wl, spectrum_SN3[ct], tauV)
    #SN3 FITS
    # Gather information to make Fits file
    col1 = fits.Column(name='Wavenumber', format='E', array=spectrum_axis_SN3)
    col2 = fits.Column(name='Flux', format='E', array=spectrum_SN3)
    cols = fits.ColDefs([col1, col2])
    hdu = fits.BinTableHDU.from_columns(cols)
    # Header info
    hdr = fits.Header()
    hdr['OBSERVER'] = 'Carter Rhea'
    hdr['COMMENT'] = "Synthetic Spectrum SN3 Number: %i"%spec_ct
    hdr['TIME'] =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr['VELOCITY'] = velocity
    hdr['BROADEN'] = broadening
    hdr['THETA'] = theta
    hdr['SIM'] = ampls_str
    hdr['SIM_NUM'] = spec_ct  # sim_num
    hdr['RES'] = resolution_sn3
    hdr['SNR'] = SNR
    hdr['Halpha'] = Ha_red_flux
    hdr['NII6548'] = n1_red_flux
    hdr['NII6583'] = n2_red_flux
    hdr['SII6716'] = s1_red_flux
    hdr['SII6731'] = s2_red_flux
    hdr['tauV'] = tauV
    empty_primary = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([empty_primary, hdu])
    hdul.writeto(output_dir+ampls_str+'/Spectrum_SN3_%i.fits'%spec_ct, overwrite=True)


    #SN2 FITS
    # Gather information to make Fits file
    col1 = fits.Column(name='Wavenumber', format='E', array=spectrum_axis_SN2)
    col2 = fits.Column(name='Flux', format='E', array=spectrum_SN2)
    cols = fits.ColDefs([col1, col2])
    hdu = fits.BinTableHDU.from_columns(cols)
    # Header info
    hdr = fits.Header()
    hdr['OBSERVER'] = 'Carter Rhea'
    hdr['COMMENT'] = "Synthetic Spectrum SN2 Number: %i"%spec_ct
    hdr['TIME'] =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr['VELOCITY'] = velocity
    hdr['BROADEN'] = broadening
    hdr['THETA'] = theta
    hdr['SIM'] = ampls_str
    hdr['SIM_NUM'] = spec_ct  # sim_num
    hdr['RES'] = resolution_sn2
    hdr['OIII5007'] = O3_red_flux
    hdr['OIII4959'] = O3_2_red_flux
    hdr['Hbeta'] = Hb_red_flux
    empty_primary = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([empty_primary, hdu])
    hdul.writeto(output_dir+ampls_str+'/Spectrum_SN2_%i.fits'%spec_ct, overwrite=True)


    #SN1 FITS
    # Gather information to make Fits file
    col1 = fits.Column(name='Wavenumber', format='E', array=spectrum_axis_SN1)
    col2 = fits.Column(name='Flux', format='E', array=spectrum_SN1)
    cols = fits.ColDefs([col1, col2])
    hdu = fits.BinTableHDU.from_columns(cols)
    # Header info
    hdr = fits.Header()
    hdr['OBSERVER'] = 'Carter Rhea'
    hdr['COMMENT'] = "Synthetic Spectrum SN1 Number: %i"%spec_ct
    hdr['TIME'] =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr['VELOCITY'] = velocity
    hdr['BROADEN'] = broadening
    hdr['THETA'] = theta
    hdr['SIM'] = ampls_str
    hdr['SIM_NUM'] = spec_ct  # sim_num
    hdr['RES'] = resolution_sn2
    hdr['OII3726'] = O2_red_flux
    hdr['OII3729'] = O2_2_red_flux
    empty_primary = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([empty_primary, hdu])
    hdul.writeto(output_dir+ampls_str+'/Spectrum_SN1_%i.fits'%spec_ct, overwrite=True)


    # save figure
    if spec_ct%1000 == 0:
        fig, axs = pl.subplots(3,1, figsize=(10,8))
        axs[0].plot([1e7/spec for spec in spectrum_axis_SN1], spectrum_SN1)
        axs[1].plot([1e7/spec for spec in spectrum_axis_SN2], spectrum_SN2)
        axs[2].plot([1e7/spec for spec in spectrum_axis_SN3], spectrum_SN3)
        print('Ha/Hb initial: '+str(sim_vals['ha']/sim_vals['hb']))
        print('Ha/Hb: '+str(Ha_red_flux/Hb_red_flux))
        pl.savefig(output_dir+ampls_str+'_'+str(spec_ct)+'.png')
        pl.clf()



for ampl, ampl_str in zip([HII_ampls, PNe_ampls, SNR_ampls],['BOND', 'PNe_2020', 'SNR_2008']):
    print("# -- Creating %s Spectra -- #"%ampl_str)
    if not os.path.exists(output_dir+ampl_str):
        os.makedirs(output_dir+ampl_str)
    for spec_ct in tqdm(range(num_syn)):
        create_spectrum(spec_ct, ampl, ampl_str)
    #Parallel(n_jobs=1)(delayed(create_spectrum)(spec_ct, ampl, ampl_str) for spec_ct in tqdm(range(num_syn)))
