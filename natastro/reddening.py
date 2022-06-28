'''
Created on 08/24/2016

@author: ariel

Provides functions to correct spectra for galactic extinction

---

Note from Natalia - 14/Aug/2020

This is originally from pycasso2.redenning:
    from pycasso2.reddening import calc_redlaw
which can be downloaded at https://bitbucket.org/streeto/pycasso2
'''

from .wcs import get_galactic_coordinates_rad

import numpy as np
from os import path
from astropy import log

__all__ = ['Galactic_reddening_corr', 'calc_redlaw', 'get_EBV',
           'Cardelli_RedLaw', 'Charlot_RedLaw',
           'Calzetti_RedLaw', 'CCC_RedLaw']


def get_EBV_map(file_name):
    '''

    Reads E(B-V) HEALPix map from Planck's dust map using healpy, if the map
    file is not found, it will be downloaded from:
    http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_CompMap_ThermalDustModel_2048_R1.20.fits


    '''
    import urllib.request, urllib.parse, urllib.error
    import healpy as hp

    if not path.exists(file_name):
        log.info(
            'Downloading dust map (1.5GB), this is probably a good time to check XKCD.')
        url = 'http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_CompMap_ThermalDustModel_2048_R1.20.fits'
        log.debug('Map: %s' % url)
        urllib.request.urlretrieve(url, file_name)

    log.info('Reading E(B-V) map from ' + file_name)
    EBV_map = hp.read_map(file_name, field=2)

    return EBV_map


def get_EBV(wcs, file_name):
    import healpy as hp

    l, b = get_galactic_coordinates_rad(wcs)
    EBV_map = get_EBV_map(file_name)
    # Get the corresponting HEALPix index and the E(B-V) value:
    index = hp.ang2pix(nside=2048, theta=(np.pi / 2) - b, phi=l)
    return EBV_map[index]



def Cardelli_RedLaw(l, R_V=None):
    '''
        @summary: Calculates q(lambda) to the Cardelli, Clayton & Mathis 1989 reddening law.
        Converted to Python from STALIGHT.
        @param l: Wavelenght vector (Angstroms)
        @param R_V: R_V factor. Default is R_V = 3.1.
    '''
# ###########################################################################
#     q = A_lambda / A_V for Cardelli et al reddening law
#     l = lambda, in Angstrons
#     x = 1 / lambda in 1/microns
#     q = a + b / R_V; where a = a(x) & b = b(x)
#     Cid@INAOE - 6/July/2004
#
    if R_V is None:
        R_V = 3.1
    a = np.zeros(np.shape(l))
    b = np.zeros(np.shape(l))
    F_a = np.zeros(np.shape(l))
    F_b = np.zeros(np.shape(l))
    x = np.zeros(np.shape(l))
    y = np.zeros(np.shape(l))
#
    for i in range(0,len(l)):
        x[i]=10000. / l[i]
        y[i]=10000. / l[i] - 1.82
#
#    x = 10000. / l
#
#     Far-Ultraviolet: 8 <= x <= 10 ; 1000 -> 1250 Angs 
    inter = np.bitwise_and(x >= 8, x <= 10)

    a[inter] = -1.073 - 0.628 * (x[inter] - 8.) + 0.137 * (x[inter] - 8.)**2 - 0.070 * (x[inter] - 8.)**3
    b[inter] = 13.670 + 4.257 * (x[inter] - 8.) - 0.420 * (x[inter] - 8.)**2 + 0.374 * (x[inter] - 8.)**3

#     Ultraviolet: 3.3 <= x <= 8 ; 1250 -> 3030 Angs 

    inter =  np.bitwise_and(x >= 5.9, x < 8)
    F_a[inter] = -0.04473 * (x[inter] - 5.9)**2 - 0.009779 * (x[inter] - 5.9)**3
    F_b[inter] =  0.2130 * (x[inter] - 5.9)**2 + 0.1207 * (x[inter] - 5.9)**3
    
    inter =  np.bitwise_and(x >= 3.3, x < 8)
    
    a[inter] =  1.752 - 0.316 * x[inter] - 0.104 / ((x[inter] - 4.67)**2 + 0.341) + F_a[inter]
    b[inter] = -3.090 + 1.825 * x[inter] + 1.206 / ((x[inter] - 4.62)**2 + 0.263) + F_b[inter]

#     Optical/NIR: 1.1 <= x <= 3.3 ; 3030 -> 9091 Angs ; 
    inter = np.bitwise_and(x >= 1.1, x < 3.3)
    
#    y = 10000. / l - 1.82
    
    a[inter] = 1.+ 0.17699 * y[inter] - 0.50447 * y[inter]**2 - 0.02427 * y[inter]**3 + 0.72085 * y[inter]**4 + 0.01979 * y[inter]**5 - 0.77530 * y[inter]**6 + 0.32999 * y[inter]**7
    b[inter] = 1.41338 * y[inter] + 2.28305 * y[inter]**2 + 1.07233 * y[inter]**3 - 5.38434 * y[inter]**4 - 0.62251 * y[inter]**5 + 5.30260 * y[inter]**6 - 2.09002 * y[inter]**7


#     Infrared: 0.3 <= x <= 1.1 ; 9091 -> 33333 Angs ; 
    inter = np.bitwise_and(x >= 0.3, x < 1.1)
    
    a[inter] =  0.574 * x[inter]**1.61
    b[inter] = -0.527 * x[inter]**1.61
    
    q = a + b / R_V

    return q


def CharlotFall_RedLaw(l, mu=0.3):
    '''
    Returns the attenuation curve from the two-component dust model by Charlot & Fall (2000). 
    
    Parameters
    ----------
    l : 1-D sequence of floats
        The \lambda wavelength array (in Angstroms). 
    
    mu : float
         Fraction of extinction contributed by the ambient ISM
            
    Returns
    -------
    q = A_lambda / A_V for Charlot & Fall (2000) et al attenuation curve.

    Natalia@St Andrews - 29/Nov/2018
    '''
    
    q_ISM = mu * np.power((l/5500.),-0.7)
    q_BC  = (1. - mu) * np.power((l/5500.),-1.3)
    q = q_ISM + q_BC
    
    return q


def Calzetti_RedLaw(l, R_V=None):
    '''
    Calculates the reddening law by Calzetti et al. (1994).
    Original comments in the .for file:
    
    q = A_lambda / A_V for Calzetti et al reddening law (formula from hyperz-manual).
    l = lambda, in Angstrons
    x = 1 / lambda in 1/microns
    Cid@INAOE - 6/July/2004
    
    Parameters
    ----------
    l : array like
        Wavelength in Angstroms.
    
    R_V : float, optional
        Selective extinction parameter (roughly "slope").
    
    Returns
    -------
    q : array
        Extinction A_lamda / A_V. Array of same length as l.
        
    '''
    if R_V is None:
        R_V = 4.05
    if not isinstance(l, np.ma.MaskedArray):
        l = np.asarray(l, 'float64')
        
    x = 1e4 / l
    aux = np.zeros_like(l)

    # UV -> Optical: 1200 -> 6300 Angs
    inter = (l >= 1200.) & (l <= 6300.)
    aux[inter] = (-2.156 + 1.509 * x[inter] - 0.198 * x[inter]**2 + 0.011 * x[inter]**3)

    # Red -> Infrared
    inter = (l >= 6300.) & (l <= 22000.)
    aux[inter] = (-1.857 + 1.040 * x[inter])

    # Get q
    q = 1. + aux * 2.659 / R_V

    # Issue a warning if lambda falls outside 1200->22000 Angs range
    if ((l < 1200.) | (l > 22000.)).any():
        log.warn('[Calzetti_RedLaw] WARNING! some lambda outside valid range (1200, 22000.)')
    return q


def CCC_RedLaw(l, R_V=None):
    '''
    CCC = "Calzetti + Claus + Cid" - law: Returns $2 = q(l) = A(l) / AV; for
    given l = (in Angs).  Strictly valid from 970 to 22000, but
    extrapolated to 912 -> 31149 Angs.  UV: I switch from the
    Leitherer to the Calzetti law at 1846 Angs. This is a bit longer
    than the nominal limit of the Leitherer law, but is is where the
    two equations meet and so the transition is smooth. In the IR,
    extrapolating the Calzetti law to l > 22000 we find that it
    becomes = 0 at 31149.77 Angs.  I thus set q = 0 for l > 3149.

    Cid@Granada - 22/Jan/2010
    '''
    if R_V is None:
        R_V = 4.05

    q = np.zeros_like(l)
    x   = 10000. / l

    # FUV: 912 -> 1846.  Leitherer et al 2002 (ApJS, 140, 303), eq 14,
    # valid from 970 to 1800 Angs, but extended to 912 -> 1846. OBS:
    # They use E(B_V)_stars in their eq, so I divide by R_V, assumed to
    # be the same as for the Calzetti-law.
    inter = (l >= 912.) & (l <= 1846.)

    q[inter] = (5.472 + 0.671 * x[inter] - 9.218e-3 * x[inter]**2 + 2.620e-3 * x[inter]**3)/R_V

    # UV -> optical: 1846 -> 6300. Calzetti law.

    inter = (l > 1846.) & (l <= 6300.)

    q[inter] = (2.659 / R_V) * (-2.156 + 1.509 * x[inter] - 0.198 * x[inter]**2 + \
                                0.011 * x[inter]**3) + 1.

    # Red -> Infrared: 6300 -> 31149. Calzetti law, originally valid up
    # to 22000 Angs, but stretched up to 31149 Angs, where it becomes
    # zero.
    inter = (l > 6300.) & (l <= 31149.)

    q[inter] = (2.659 / R_V) * (-1.857 + 1.040 * x[inter]) + 1.

    return q

def HowarthMW_RedLaw(l, R_V=None):
    '''
        @summary: Calculates q(lambda) for Howarth (1983) Milky Way reddening law.
        UV: for LMC but based on Seaton's Galactic extinction.
        Optical and IR: for MW.
        Note he also has an optical LMC law (not implemented).
        @param l: Wavelenght vector (Angstroms)
        @param R_V: R_V factor. Default is R_V = 3.1.

        TO INVESTIGATE: Discontinuity between UV and optical.
    '''
# ###########################################################################
#     q = A_lambda / A_V
#     l = lambda, in Angstrons
#     x = 1 / lambda in 1/microns

    if R_V is None:
        R_V = 3.1

    x = 1.e4 / np.array(l)
    q = np.zeros_like(x)


    # UV
    ff = (x >= 2.75) & (x <= 9.0)
    def q_UV(x):
        print (x)
        return R_V - 0.236 + 0.462 * x + 0.105 * x**2 + 0.454/ ((x - 4.557)**2 + 0.293)
    q[ff] = q_UV(x[ff])
    
    # Optical
    ff = (x >= 1.83) & (x <= 2.75)
    def q_opt(x):
        return R_V + 2.56 * (x - 1.83) - 0.993 * (x -1.83)**2
    q[ff] = q_opt(x[ff])

    # IR
    ff = (x <= 1.83)
    def q_IR(x):
        return ((1.86 - 0.48 * x) * x - 0.1) * x
    q[ff] = q_IR(x[ff])

    # Normalize
    q /= q_IR(1.e4/5500.)
    
    # Issue a warning if lambda falls outside >1111 Angs range
    if (l < 1111.).any():
        log.warn('[HowarthMW_RedLaw] WARNING! some lambda outside valid range (1111, infinity.)')
    
    return q
    
def bla():
    a = np.zeros(np.shape(l))
    b = np.zeros(np.shape(l))
    F_a = np.zeros(np.shape(l))
    F_b = np.zeros(np.shape(l))
    x = np.zeros(np.shape(l))
    y = np.zeros(np.shape(l))
#
    for i in range(0,len(l)):
        x[i]=10000. / l[i]
        y[i]=10000. / l[i] - 1.82
#
#    x = 10000. / l
#
#     Far-Ultraviolet: 8 <= x <= 10 ; 1000 -> 1250 Angs 
    inter = np.bitwise_and(x >= 8, x <= 10)

    a[inter] = -1.073 - 0.628 * (x[inter] - 8.) + 0.137 * (x[inter] - 8.)**2 - 0.070 * (x[inter] - 8.)**3
    b[inter] = 13.670 + 4.257 * (x[inter] - 8.) - 0.420 * (x[inter] - 8.)**2 + 0.374 * (x[inter] - 8.)**3

#     Ultraviolet: 3.3 <= x <= 8 ; 1250 -> 3030 Angs 

    inter =  np.bitwise_and(x >= 5.9, x < 8)
    F_a[inter] = -0.04473 * (x[inter] - 5.9)**2 - 0.009779 * (x[inter] - 5.9)**3
    F_b[inter] =  0.2130 * (x[inter] - 5.9)**2 + 0.1207 * (x[inter] - 5.9)**3
    
    inter =  np.bitwise_and(x >= 3.3, x < 8)
    
    a[inter] =  1.752 - 0.316 * x[inter] - 0.104 / ((x[inter] - 4.67)**2 + 0.341) + F_a[inter]
    b[inter] = -3.090 + 1.825 * x[inter] + 1.206 / ((x[inter] - 4.62)**2 + 0.263) + F_b[inter]

#     Optical/NIR: 1.1 <= x <= 3.3 ; 3030 -> 9091 Angs ; 
    inter = np.bitwise_and(x >= 1.1, x < 3.3)
    
#    y = 10000. / l - 1.82
    
    a[inter] = 1.+ 0.17699 * y[inter] - 0.50447 * y[inter]**2 - 0.02427 * y[inter]**3 + 0.72085 * y[inter]**4 + 0.01979 * y[inter]**5 - 0.77530 * y[inter]**6 + 0.32999 * y[inter]**7
    b[inter] = 1.41338 * y[inter] + 2.28305 * y[inter]**2 + 1.07233 * y[inter]**3 - 5.38434 * y[inter]**4 - 0.62251 * y[inter]**5 + 5.30260 * y[inter]**6 - 2.09002 * y[inter]**7


#     Infrared: 0.3 <= x <= 1.1 ; 9091 -> 33333 Angs ; 
    inter = np.bitwise_and(x >= 0.3, x < 1.1)
    
    a[inter] =  0.574 * x[inter]**1.61
    b[inter] = -0.527 * x[inter]**1.61
    
    q = a + b / R_V

    return q

def calc_redlaw(l, redlaw, R_V=None, **kwargs):
    l = np.atleast_1d(l)
    if redlaw == 'CCM':
        return Cardelli_RedLaw(l, R_V)
    elif redlaw == 'CAL':
        return Calzetti_RedLaw(l, R_V)
    elif redlaw == 'CCC':
        return CCC_RedLaw(l, R_V)
    elif redlaw == 'CF00':
        return CharlotFall_RedLaw(l, **kwargs)
    elif redlaw == 'HMW83':
        return HowarthMW_RedLaw(l, R_V)
    else:
        raise Exception('Unknown reddening law %s.' % redlaw)


def Galactic_reddening_corr(wave, EBV, R_V=3.1):
    Av = R_V * EBV
    A_lambda = Av * calc_redlaw(wave, redlaw='CCM', R_V=R_V)
    tau_lambda = A_lambda / (2.5 * np.log10(np.exp(1.)))
    return np.exp(tau_lambda)
