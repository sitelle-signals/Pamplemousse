'''
Created on 26/06/2015

@author: andre
'''

from astropy import log, wcs
import numpy as np

__all__ = ['get_axis_coordinates', 'get_wavelength_coordinates', 'write_WCS',
           'get_reference_pixel', 'get_galactic_coordinates_rad',
           'get_pixel_area', 'get_pixel_area_srad', 'get_pixel_scale', 'get_pixel_scale_rad',
           'get_Naxis', 'replace_wave_WCS', 'shift_celestial_WCS', 'scale_celestial_WCS']


rad_per_deg = np.pi / 180.0
one_angstrom = 1e-10


def get_celestial(w):
    wc = w.celestial
    if wc.wcs.naxis != 2:
        log.warn('No celestial axes found, using axes 1,2.')
        wc = w.sub(2)
    return wc


def proj_plane_pixel_area(w):
    '''
    More tolerant version of `astropy.wcs.utils.proj_plane_pixel_area`.
    '''
    w = get_celestial(w)
    psm = w.pixel_scale_matrix
    a = np.abs(np.linalg.det(psm))
    x_unit, y_unit = w.wcs.cunit
    if x_unit == '' or y_unit == '':
        a /= 3600**2
    return a


def get_wavelength_coordinates(w, Nwave):
    w = w.sub([3])
    pix_coords = np.arange(Nwave)
    wave_coords = w.wcs_pix2world(pix_coords[:, np.newaxis], 0)
    if w.wcs.cunit[0] == 'm':
        wave_coords /= one_angstrom
    return np.squeeze(wave_coords)


def get_Naxis(header, i):
    return header['NAXIS%d' % i]


def get_celestial_coordinates(w, Nx, Ny, relative=True):
    # FIXME: Not sure how to make the coordinates relative for all cases. Nor
    # how to return arcsec always.
    w = get_celestial(w)
    x0_world, y0_world = w.wcs.crval
    make_relative = (x0_world != 0.0 and y0_world != 0.0 and relative)

    if make_relative:
        w.wcs.crval = (180.0, 0.0)
    xx_pix = np.arange(Nx) + 1
    yy_pix = np.arange(Ny) + 1
    x0_pix, y0_pix = np.rint(w.wcs.crpix).astype('int')
    xx_world, _ = w.wcs_pix2world(xx_pix, np.zeros_like(xx_pix) + y0_pix, 1)
    _, yy_world = w.wcs_pix2world(np.zeros_like(yy_pix) + x0_pix, yy_pix, 1)
    if make_relative:
        xx_world -= 180.0
        x_unit, y_unit = w.wcs.cunit
        if x_unit == 'deg' or y_unit == 'deg':
            xx_world *= 3600.0
            yy_world *= 3600.0
    return xx_world, yy_world


def get_galactic_coordinates_rad(w):
    '''

    Input: header with WCS information.
    Returns: Galactic coordinates l and b in radians to compare to HEALPix 
    maps.

    Note: l is consistent with HEALPix's phi, while HEALPix's theta will be 
    given by theta = pi/2 - b.

    '''

    w = get_celestial(w)
    x0, y0 = w.wcs.crpix
    coords = wcs.utils.pixel_to_skycoord(x0, y0, w, origin=1, mode='wcs')
    galcoords = coords.transform_to('galactic')

    l, b = galcoords.l.radian, galcoords.b.radian

    return l, b


def get_reference_pixel(w, as_int=False):
    crpix = w.wcs.crpix - 1
    if as_int:
        crpix = np.rint(crpix).astype('int')
    return (crpix[2], crpix[1], crpix[0])


def get_wavelength_sampling(w):
    w = w.sub([3])
    s = wcs.utils.proj_plane_pixel_scales(w)
    if w.wcs.cunit[0] == 'm':
        s /= one_angstrom
    return np.asscalar(s)


def get_pixel_area(w):
    return proj_plane_pixel_area(w)


def get_pixel_area_srad(w):
    a = get_pixel_area(w)
    return a * (rad_per_deg * rad_per_deg)


def get_pixel_scale(w):
    w = get_celestial(w)
    s = wcs.utils.proj_plane_pixel_scales(w)
    s = s.mean()
    x_unit, y_unit = w.wcs.cunit
    if x_unit == '' or y_unit == '':
        s /= 3600
    return s


def get_pixel_scale_rad(w):
    s = get_pixel_scale(w)
    return s * rad_per_deg


def write_WCS(header, w):
    if w is None:
        return

    wcs_garbage = ['CRPIX1', 'CRPIX2', 'CRPIX3',
                   'CRVAL1', 'CRVAL2', 'CRVAL3',
                   'CDELT1', 'CDELT2', 'CDELT3',
                   'CUNIT1', 'CUNIT2', 'CUNIT3',
                   'PC1_1', 'PC1_2', 'PC1_3',
                   'PC2_1', 'PC2_2', 'PC2_3',
                   'PC3_1', 'PC3_2', 'PC3_3',
                   'CD1_1', 'CD1_2', 'CD1_3',
                   'CD2_1', 'CD2_2', 'CD2_3',
                   'CD3_1', 'CD3_2', 'CD3_3',
                   ]
    for h in wcs_garbage:
        if h in header:
            del header[h]

    header.extend(w.to_header(), update=True)


def replace_wave_WCS(w, crpix_wave, crval_wave, cdelt_wave):
    w = w.copy()
    if w.wcs.cunit[2] == 'm':
        crval_wave *= one_angstrom
        cdelt_wave *= one_angstrom
    crpix = w.wcs.crpix
    w.wcs.crpix = (crpix[0], crpix[1], crpix_wave + 1)
    crval = w.wcs.crval
    w.wcs.crval = crval[0], crval[1], crval_wave
    if w.wcs.has_cd():
        w.wcs.cd[2, 2] = cdelt_wave
    elif w.wcs.has_pc():
        w.wcs.pc[2, 2] = cdelt_wave
    else:
        w.wcs.cdelt[2] = cdelt_wave
    return w

def shift_celestial_WCS(w, dx, dy):
    w = w.copy()
    crpix = w.wcs.crpix
    crpix = (crpix[0] - dx, crpix[1] - dy, crpix[2])
    w.wcs.crpix = crpix
    return w

def scale_celestial_WCS(w, scaling):
    w = w.copy()
    crpix = w.wcs.crpix
    transform = lambda p: (p - 0.5) / scaling + 0.5
    w.wcs.crpix = (transform(crpix[0]), transform(crpix[1]), crpix[2])
    if w.wcs.has_cd():
        w.wcs.cd[0:2, 0:2] *= scaling
    elif w.wcs.has_pc():
        w.wcs.pc[0:2, 0:2] *= scaling
    else:
        w.wcs.cdelt[0:2] *= scaling
    return w
