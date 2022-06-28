'''
Some useful functions

Natalia@UFSC - 07/May/2014
'''

import os
import sys
import time
import functools

import numpy as np
import astropy.table
import scipy.sparse

import h5py
#import hickle as hkl

def ind2flag(array, indices):
    # Transform indices array into boolean flag array
    flag = np.zeros_like(array, 'bool')
    flag[indices] = 1
    return flag

def arbitrary_round(x, prec = 2, base = 0.05):
    # http://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
    return round(base * round(float(x)/base), prec)

def mask_minus999(var, thereshold = -990., fill = None):
    a = np.ma.masked_array(var, mask=(var <= thereshold))
    if fill == None:
        return a
    else:
        return a.filled(fill)

def mask_nonfinite(var, fill = None):
    a = np.ma.masked_array(var, mask=~np.isfinite(var))
    if fill == None:
        return a
    else:
        return a.filled(fill)

def safe_sqrt(x):
    sqrt_x = np.where( (x >= 0), np.sqrt(abs(x + (x < 0))), -999. )
    sqrt_x = np.ma.masked_where((sqrt_x < -990), sqrt_x)
    return sqrt_x

def safe_log10(x):
    log10_x = np.where( (x > 0), np.log10(abs(x + (x == 0))), -999. )
    log10_x = np.ma.masked_where((log10_x < -990), log10_x)
    if isinstance(x, np.ma.MaskedArray):
        log10_x.mask |= x.mask
    return log10_x

def safe_ln(x):
    ln_x = np.where( (x > 0), np.log(abs(x + (x == 0))), -999. )
    ln_x = np.ma.masked_where((ln_x < -990), ln_x)
    if isinstance(x, np.ma.MaskedArray):
        ln_x.mask |= x.mask
    return ln_x

def safe_div(a, b, mask999 = True, fill = None):
    aob = np.where( (b != 0), a / ( b + (b == 0)), -999. )
    if mask999:
        aob = np.where( (b > -990) & (a > -990), aob, -999. )
    aob = np.ma.masked_where((aob < -990), aob)
    if fill == None:
        return aob
    else:
        return aob.filled(fill)

def safe_x(x):
    x = np.where(np.isfinite(x), x, -999.)
    x = np.ma.masked_where((x < -990), x)
    return x

def safe_sum(a, b):
    a = safe_x(a)
    b = safe_x(b)
    apb = np.where( ((a > -999.) & (b > -999.)), a + b, -999. )
    apb = np.ma.masked_where((apb < -990), apb)
    return apb

def safe_sub(a, b):
    a = safe_x(a)
    b = safe_x(b)
    amb = np.where( ((a > -999.) & (b > -999.)), a - b, -999. )
    amb = np.ma.masked_where((amb < -990), amb)
    return amb

def safe_pow(a, b):
    a = safe_x(a)
    b = safe_x(b)
    apb = np.where( ((a > -999.) & (b > -999.)), a**b + (a <= -999.), -999. )
    apb = np.where( ((a == 0.) & (b < 0.)), -999., apb )
    apb = np.ma.masked_where((apb < -990), apb)
    return apb

def safe_mult(x, f, log = False):
    if log:
        xf = x + f
    else:
        xf = x * f
    xf = np.where( ((x > -999.) & (f > -999.)), xf, -999. )
    return xf

def quad_safe_sum(x1, x2=None):
    if x2 is None:
        x = np.array(x1)
    else:
        x = np.hstack((np.array(x1), np.array(x2)))
    aux = safe_pow(x, 2)
    qs = safe_sqrt(np.sum(aux))
    return qs

def unique_tol(x, rtol=0., atol=1.e-8):
    '''
    Return unique values in an array ignoring values that are too close.

    rtol = relative tolerance
    atol = absolute tolerance
    (see numpy.isclose documentation for details)
    '''

    x_uniq = np.unique(x)

    aux = np.isclose(x_uniq[:-1], x_uniq[1:], rtol=rtol, atol=atol)
    flag_isclose = np.concatenate([[False], aux])

    x_uniq_tol = x_uniq[~flag_isclose]

    return x_uniq_tol

def calcLineRatio(F1, F2, dF1 = 0., dF2 = 0., fluxInLog = True, returnAll = True):
    '''
    Calc a line ratio and its uncertainty. Returns F1/F2, d(F1/F2), log F1/F2 and d(log F1/F2).

    Natalia@UFSC - 05/Dec/2014
    '''

    if fluxInLog:

        logF1  = F1
        dlogF1 = dF1
        F1 = safe_pow(10, logF1)
        dF1 = np.log(10) * F1 * dlogF1

        logF2  = F2
        dlogF2 = dF2
        F2 = safe_pow(10, logF2)
        dF2 = np.log(10) * F2 * dlogF2

        logF1F2 = safe_sub(logF1, logF2)
        F1F2 = safe_pow(10, logF1F2.data)

    else:

        F1F2 = safe_div(F1, F2)
        logF1F2 = safe_log10(F1F2)
        dlogF1 = safe_div( safe_div(dF1, F1), np.log(10.))
        dlogF2 = safe_div( safe_div(dF2, F2), np.log(10.))

    # Error propagation
    dlogF1F2 = np.where( ((dlogF1 > -999) & (dlogF2 > -999)), safe_pow((dlogF1**2 + dlogF2**2), 0.5), -999. )
    dF1F2 = np.where( ((dlogF1F2 > -999.) & (F1F2.data > -999.)), np.log(10.) * F1F2 * dlogF1F2, -999. )

    if returnAll:
        return mask_minus999(F1F2), mask_minus999(dF1F2), mask_minus999(logF1F2), mask_minus999(dlogF1F2)
    elif fluxInLog:
        return mask_minus999(logF1F2), mask_minus999(dlogF1F2)
    else:
        return mask_minus999(F1F2), mask_minus999(dF1F2)


def calcLineSum(F1, F2, dF1 = 0., dF2 = 0., fluxInLog = True, returnAll = True):
    '''
    Calc the sum of two lines and its uncertainty. Returns F1+F2 and d(F1+F2), log (F1+F2) and d(log (F1+F2))
    Calc a line ratio and its uncertainty. Returns F1/F2, d(F1/F2), log F1/F2 and d(log F1/F2).

    Natalia@UFSC - 10/Dec/2014
    '''

    if fluxInLog:

        logF1  = F1
        dlogF1 = dF1
        F1 = safe_pow(10, logF1)
        dF1 = np.log(10) * F1 * dlogF1

        logF2  = F2
        dlogF2 = dF2
        F2 = safe_pow(10, logF2)
        dF2 = np.log(10) * F2 * dlogF2

    # Calc sum and log sum
    F12 = safe_sum(F1, F2)
    logF12 = safe_log10(F12)

    # Error propagation
    dF12 = safe_pow( (dF1**2 + dF2**2), 0.5 )
    dlogF12 = safe_div(dF12, F12 * np.log(10.))

    if returnAll:
        return mask_minus999(F12), mask_minus999(dF12), mask_minus999(logF12), mask_minus999(dlogF12)
    elif fluxInLog:
        return mask_minus999(logF12), mask_minus999(dlogF12)
    else:
        return mask_minus999(F12), mask_minus999(dF12)


def calcLineFracDev(F1, F2, fluxInLog = True):
    '''
    Calculates the fraction deviation between two lines:
    (F1 - F2) / F1
    '''

    if fluxInLog:
        logF1  = F1
        F1 = safe_pow(10, logF1)
        logF2  = F2
        F2 = safe_pow(10, logF2)

    dF = safe_sum(F1, -F2)
    dev = safe_div(dF, F1)

    return dev


def read_hdf5_to_table(fileName, debug = False):
    datasets = load_hdf5(fileName, debug = debug)

    tables = {}
    for dataset, arr in datasets.items():
        if debug: print ("@@> Converting dataset %s to Table..." % dataset)
        t = astropy.table.Table(arr, copy=False)
        t.table_name = dataset
        tables[dataset] = t

    if (len(tables.keys()) == 1):
        tables = tables[dataset]

    if debug: print ("@@> Done.")
    return tables


def load_hdf5(fileName, debug = False):
    '''
    Reads an hdf5 file to memory; much faster than atpy or astropy
    '''

    f = h5py.File(fileName, 'r')

    datasets = {}
    for dataset in f.keys():
        if debug:
            print ("@@> Reading dataset %s..." % dataset)
        dset = f.get(dataset)
        arr = np.empty(dset.shape, dtype=dset.dtype)
        dset.read_direct(arr)
        datasets[dataset] = arr

    if debug:
        print ("@@> Done.")

    f.close()

    return datasets


def saveTableToFile(table, saveFile = None, fileType = 'hdf5', table_name = 'table', overwrite = False, **kwargs):
    '''
    Save a grid or source table to a file.
    This is much faster than atpy or astropy for creating hdf5 files.
    '''

    if (fileType == 'hdf5'):
        kwargs['path'] = table_name

    if saveFile is None:
        print ('!!> No file name given, not saving to any file.')
    else:
        if (not overwrite) and (os.path.exists(saveFile)):
            print ('!!> Output file %s already exists.\n    Use the overwrite option if needed.' % saveFile)
        else:
            if (os.path.exists(saveFile)):
                os.remove(saveFile)
            print ("@@> Saving table %s to file %s as type %s" % (table_name, saveFile, fileType))
            table.write(saveFile, format=fileType, **kwargs)


def saveDictToFile(dictArrays, fileName, overwrite = False, **kwargs):
    '''
    Save a dictionary of numpy arrays to an hdf5 file.
    '''

    if (not overwrite) and (os.path.exists(fileName)):
        print ('!!> Output file %s already exists.\n    Use the overwrite option if needed.' % fileName)
    else:
        if (os.path.exists(fileName)):
            os.remove(fileName)
        hkl.dump(dictArrays, fileName, **kwargs)


class timeit(object):
    def __init__(self, function, debug = True, *args, **kwargs):
        self.debug = debug
        self.function = function
        functools.update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        t1 = time.time()
        f = self.function(*args, **kwargs)
        if self.debug:
            print ('**> %s took %s seconds.' % (self.function.__name__, time.time() - t1)    )
        return f

def method_dec(decorator):
    """
    Converts a function decorator into a method decorator.

    From: https://groups.google.com/forum/#!msg/django-developers/MCSxSkf1QE0/s768z_V2afMJ
    Linked from: http://stackoverflow.com/questions/1288498/using-the-same-decorator-with-arguments-with-functions-and-methods#comment2594561_1288936
    """
    def _dec(func):
        def _wrapper(self, *args, **kwargs):
            def bound_func(*args2, **kwargs2):
                return func(self, *args2, **kwargs2)
            # Change function name (added by Natalia)
            bound_func.__name__ = func.__name__
            # bound_func has the signature that 'decorator' expects i.e.  no
            # 'self' argument, but it is a closure over self so it can call
            # 'func' correctly.
            return decorator(bound_func)(*args, **kwargs)
        return functools.wraps(func)(_wrapper)
    functools.update_wrapper(_dec, decorator)
    # Change the name, to avoid confusion, as this is *not* the same
    # decorator.
    _dec.__name__ = 'method_dec(%s)' % decorator.__name__
    return _dec

# Method decorators
mtimeit = method_dec(timeit)



def cartesian(arrays, out=None):
    """
    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


# --------------------------------------------------------------------
class tableHDFRAM(object):

    def __init__(self, *args, **kwargs):
        self._tableRAM = astropy.table.Table()
        self._tableHDF = astropy.table.Table()
        self._readHdf5(*args, **kwargs)

    def __setitem__(self, item, value):
        self._tableRAM[item] = value

    def __len__(self):
        return len(self._tableRAM)

    def __get__(self, instance, owner):
        '''
        TO FIX
        '''
        print('get')
        return self._tableRAM

    def keysRAM(self):
        return self._tableRAM.keys()

    def keysHDF(self):
        return list(self._tableHDF.dtype.names)

    def keys(self):
        keys = self.keysRAM() + self.keysHDF()
        return keys

    def __getitem__(self, item):
        '''
        Based on astropy.table
        '''
        import six

        # Get keys from RAM and HDF5 file
        #keysRAM = self._tableRAM.keys()
        #keysHDF = list(self._tableHDF.dtype.names)
        #keys = keysRAM + keysHDF


        if isinstance(item, six.string_types):
            # For a single column: read this column into RAM
            if item in self.keysRAM():
                return self._tableRAM[item]
            elif item in self.keysHDF():
                self._tableRAM[item] = self._tableHDF[item]
                return self._tableRAM[item]

        elif isinstance(item, (tuple, list)) and all(isinstance(x, six.string_types) for x in item):
            # For many columns: read columns into RAM

            bad_names = [x for x in item if x not in self.keys()]
            if bad_names:
                raise ValueError('Slice name(s) {0} not valid column name(s)'
                                 .format(', '.join(bad_names)))

            for k in item:
                if (k in self.keysHDF()) & (k not in self.keysRAM()):
                    self._tableRAM[k] = self._tableHDF[k]

            return self._tableRAM[item]

        elif (isinstance(item, (int, np.integer)) or
              isinstance(item, slice) or
              isinstance(item, np.ndarray) or
              isinstance(item, list) or
              isinstance(item, tuple) and all(isinstance(x, np.ndarray) for x in item)):
            # For rows: read rows from RAM and disk, but do not copy to RAM
            keysHDF = [key for key in self.keysHDF() if key not in self.keysRAM()]
            t1 = self._tableRAM[item]
            if len(keysHDF) == 0:
                t2 = astropy.table.Table()
            else:
                t2 = astropy.table.Table(data = self._tableHDF[item][keysHDF])
            t3 = astropy.table.hstack([t2, t1])
            return t3

        else:
            raise ValueError('Illegal type {0} for table item access'
                             .format(type(item)))

    def sort(self, *args, **kwargs):
        for k in self.keysHDF():
            if k not in self.keysRAM():
                self.__getitem__(k)

        self._tableRAM.sort(*args, **kwargs)

    def _readHdf5(self, fileName = None, key = None, delTableRam = True):

        if fileName is not None:
            # Read hdf5 file but do not load it to memory
            a = h5py.File(fileName, 'r')
            if key is None:
                key = a.keys()[0]
            self._tableHDF = a[key]

            # Copy first array from HDF5 to RAM
            k = self._tableHDF.dtype.names[0]
            self._tableRAM = astropy.table.Table(data = {k: self._tableHDF[k]})

    def write(self, *args, **kwargs):
        table = self[:]
        table.write(*args, **kwargs)
# --------------------------------------------------------------------



# --------------------------------------------------------------------
def multiply_sparse(sparse_matrix, array, toCOO = False, toCSR = False, toCSC = False, debug = False):
    '''
    From:
    http://stackoverflow.com/questions/12237954/multiplying-elements-in-a-sparse-array-with-rows-in-matrix
    '''

    out_matrix = sparse_matrix.copy()

    if (toCSR + toCSC + toCOO) > 1:
        print ('@@> More than one argument to transfor the sparse matrix passed (%s, %s, %s). Giving up.' % (toCSR, toCSC, toCOO))
        sys.exit(1)
    if (toCOO) & (not isinstance(out_matrix, scipy.sparse.coo_matrix)):
        out_matrix = out_matrix.tocoo()
    if (toCSR) & (not isinstance(out_matrix, scipy.sparse.csr_matrix)):
        out_matrix = out_matrix.tocsr()
    if (toCSC) & (not isinstance(out_matrix, scipy.sparse.csc_matrix)):
        out_matrix = out_matrix.tocsc()


    if isinstance(out_matrix, scipy.sparse.coo_matrix):
        if debug: print ('COO')
        out_matrix.data *= array[out_matrix.col]

    elif isinstance(out_matrix, scipy.sparse.csr_matrix):
        if debug: print ('CSR')
        out_matrix.data *= array[out_matrix.indices]

    elif isinstance(out_matrix, scipy.sparse.csc_matrix):
        if debug: print ('CSC')
        out_matrix.data *= array.repeat(np.diff(out_matrix.indptr))

    else:
        raise ValueError('Matrix must be a scipy.sparse CSR, CSC or COO.')

    # Check that the multiplication worked
    if debug:
        test = sparse_matrix.multiply(array)
        print (np.all(np.isclose(out_matrix.toarray(), test)))


    return out_matrix
# --------------------------------------------------------------------

# --------------------------------------------------------------------
def ReSamplingMatrixNonUniform_sparseArray(bins_resam = None, bins_orig = None,
                                           fullArray = False, debug = False):
    '''
    Sparse matrices make a huge difference here!
    '''

    br_low, br_upp, br_cen, br_size, Nr = getBinsLims(bins_resam)
    bo_low, bo_upp, bo_cen, bo_size, No = getBinsLims(bins_orig)


    # Find out where the indices of each original bin edge falls into the resampled bins
    irlow__o = np.digitize(bo_low, br_low, right = False) - 1
    irupp__o = np.digitize(bo_upp, br_low, right = False) - 1
    #print di, irlow__o, irupp__o


    # Check if any resampled bin edge is on the border the original range
    io = np.arange(No)

    il = np.where(irlow__o == -1)[0]
    fl = (bo_low[il] < br_low[0])
    f_lowerEdge = np.in1d(io, il[fl])

    iu = np.where(irupp__o == (Nr-1))[0]
    fu = (bo_upp[iu] > br_upp[-1])
    f_upperEdge = np.in1d(io, iu[fu])


    # Fix the upper edge
    irupp__o[f_upperEdge] += 1

    # Calculate how many resampled bins the original bins span
    di = (irupp__o - irlow__o)


    # Check if any resampled bin edge is outside the original range
    il = np.where(irupp__o == -1)[0]
    fl = (bo_upp[il] < br_low[0])

    iu = np.where(irlow__o == (Nr-1))[0]
    fu = (bo_low[iu] > br_upp[-1])

    fok = (~np.in1d(io, il[fl])) & (~np.in1d(io, iu[fu]))


    # Get the flags for each case.

    # Case 1: Original bin is smaller than or equal to the resampled bin.
    f1__o = (di <= 0) & (fok) & (~f_upperEdge) & (~f_lowerEdge)
    N1 = (f1__o).sum()

    # Case 2: Original bin is greater than the resampled bin.
    # We need to consider three subcases.
    f2__o = (di > 0) & (fok)

    # Case 2a: Original bin covers the left part of the resampling bin.
    f2a__o = (f2__o) & (~f_upperEdge)
    N2a = f2a__o.sum()

    # Case 2b: Original bin covers the right part of the resampling bin.
    f2b__o = (f2__o) & (~f_lowerEdge)
    N2b = f2b__o.sum()

    # Case 2d: resampling window is larger than the edges of the original window.
    f2d__o = (f2__o) & (di > 1)
    N2d = (di[f2d__o]-1).sum()

    # Total non-zero elements in the matrix.
    Nt = N1 + N2a + N2b + N2d
    #print Nt, N1, N2a, N2b, N2d


    # Start numpy arrays, which should be much faster than lists
    # to hold the columns, rows and data of the sparse matrix
    cols__o = np.zeros(Nt, 'int')
    rows__r = np.zeros(Nt, 'int')
    fracs__ro = np.zeros(Nt)

    def update_arrays(iini, ifin, col__o, row__r, frac__ro):
        cols__o[iini:ifin] = col__o
        rows__r[iini:ifin] = row__r
        fracs__ro[iini:ifin] = frac__ro

    # Updating arrays for Case 1: Original bin is smaller than or equal to the resampled bin.
    col__o = np.where(f1__o)[0]
    row__r = irlow__o[f1__o]
    frac__ro = (bo_size[col__o] / br_size[row__r])
    iini = 0
    ifin = iini + N1
    update_arrays(iini, ifin, col__o, row__r, frac__ro)


    # Updating arrays for Case 2a: Original bin covers the left part of the resampling bin.
    col__o = np.where(f2a__o)[0]
    row__r = irupp__o[f2a__o]
    frac__ro = (bo_upp[col__o] - br_low[row__r]) / br_size[row__r]
    iini = ifin
    ifin = iini + N2a
    update_arrays(iini, ifin, col__o, row__r, frac__ro)


    # Updating arrays for Case 2b: Original bin covers the right part of the resampling bin.
    col__o = np.where(f2b__o)[0]
    row__r = irlow__o[f2b__o]
    frac__ro = (br_upp[row__r] - bo_low[col__o]) / br_size[row__r]
    iini = ifin
    ifin = iini + N2b
    update_arrays(iini, ifin, col__o, row__r, frac__ro)


    # Updating arrays for Case 2d: resampling window is larger than the original window.
    row__r = np.empty(N2d, 'int')
    col__o = np.empty(N2d, 'int')

    _di = di[f2d__o]
    _irlow__o = irlow__o[f2d__o]
    _inds__o = np.where(f2d__o)[0]

    _f = (_di > 1)
    Nf = _f.sum()
    j = 0

    while (Nf > 0):
        row__r[j:j+Nf] = (_irlow__o + _di - 1)[_f]
        col__o[j:j+Nf] = _inds__o[_f]

        j += Nf
        _di -= 1
        _f = (_di > 1)
        Nf = _f.sum()

    frac__ro = np.ones(len(col__o))

    iini = ifin
    ifin = iini + N2d
    update_arrays(iini, ifin, col__o, row__r, frac__ro)

    # Create sparse matrix
    fracBin__ro = scipy.sparse.coo_matrix((fracs__ro, (rows__r, cols__o)), shape=(Nr, No))

    if fullArray:
        fracBin__ro = fracBin__ro.toarray()

    return fracBin__ro
# --------------------------------------------------------------------

# --------------------------------------------------------------------
def ReSamplingMatrixNonUniform_fullArray(bins_resam = None, bins_orig = None, debug = False):
    '''
    Based on ReSamplingMatrixNonUniform from pycasso2, but vectorized.

    Assumes the resampled bins are ordered.

    Natalia@Corrego - 30/Dec/2015
    '''

    br_low, br_upp, br_cen, br_size, Nr = getBinsLims(bins_resam)
    bo_low, bo_upp, bo_cen, bo_size, No = getBinsLims(bins_orig)

    # Check to see if this density function contributes to this bin (the easy part)
    flagX__ro =  (br_low[..., np.newaxis] < bo_upp[np.newaxis, ...]) & (br_upp[..., np.newaxis] > bo_low[np.newaxis, ...])

    # Conserving probabilities (like conserving flux in spectral resampling)
    # This only makes sense if the original bin contributes to the resampled.
    fracBin__ro = np.zeros(flagX__ro.shape)

    # Case 1: resampling window is smaller than or equal to the original window.
    f1__ro = flagX__ro & (br_low[..., np.newaxis] >= bo_low[np.newaxis, ...]) & (br_upp[..., np.newaxis] <= bo_upp[np.newaxis, ...])
    fracBin__ro[f1__ro] = 1.

    # Case 2: resampling window is larger than the original window.
    f2__ro = flagX__ro & (br_low[..., np.newaxis] <= bo_low[np.newaxis, ...]) & (br_upp[..., np.newaxis] >= bo_upp[np.newaxis, ...])
    fracBin__ro[f2__ro] = (bo_size[np.newaxis, ...] / br_size[..., np.newaxis])[f2__ro]

    # Case 3: resampling window is on the right of the original window.
    f3__ro = flagX__ro & (br_low[..., np.newaxis] >= bo_low[np.newaxis, ...]) & (br_upp[..., np.newaxis] >= bo_upp[np.newaxis, ...])
    fracBin__ro[f3__ro] = ((bo_upp[np.newaxis, ...] - br_low[..., np.newaxis]) / br_size[..., np.newaxis])[f3__ro]

    # Case 4: resampling window is on the left of the original window.
    f4__ro = flagX__ro & (br_low[..., np.newaxis] <= bo_low[np.newaxis, ...]) & (br_upp[..., np.newaxis] <= bo_upp[np.newaxis, ...])
    fracBin__ro[f4__ro] = ((br_upp[..., np.newaxis] - bo_low[np.newaxis, ...]) / br_size[..., np.newaxis])[f4__ro]

    return fracBin__ro
# --------------------------------------------------------------------

# --------------------------------------------------------------------
def testsFor_ReSamplingMatrix(test = 1):

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()

    if test == 0:
        bins_o = np.array([0., 2, 4])
        a__o   = np.array([3, 2, 0])
        bins_r = bins_o

    if test == 1:
        bins_o = np.array([0., 2, 4])
        a__o   = np.array([3, 2, 0])
        bins_r = np.array([0., 1, 2, 3, 4])

    if test == 2:
        bins_o = np.array([0., 1, 2, 3, 4])
        a__o   = np.array([3, 3, 2, 2, 0])
        bins_r = np.array([0., 2, 4])

    if test == 3:
        bins_o = np.array([0., 2, 4])
        a__o   = np.array([3, 2, 0])
        bins_r = np.array([0., 1.5, 2, 3, 4])

    if test == 4:
        bins_o = np.array([0., 2, 4, 6])
        a__o   = np.array([3, 2, 2, 0])
        bins_r = np.array([0., 1.5, 2, 3, 4, 6])

    if test == 5:
        bins_o = np.array([0., 2, 4, 6])
        a__o   = np.array([3, 2, 2, 0])
        bins_r = np.array([0., 0.1, 0.2, 1, 2, 3, 4, 6])

    if test == 6:
        bins_o = np.array([0., 2, 4, 6])
        a__o   = np.array([3, 2, 2, 0])
        bins_r = np.array([0., 1., 4, 6])

    if test == 7:
        bins_o = np.array([0., 2, 4, 6])
        a__o   = np.array([3, 2, 2, 0])
        bins_r = np.linspace(-2, 8, 50)

    if test == 8:
        bins_o = np.array([0.01391141, 0.0565045,  0.17295018, 0.47385343])
        a__o   = np.array([1.33764282, 1.54277143, 1.73827933, 0])
        bins_r = np.array([0.09966678, 0.23994778, 0.50652996, 0.69347715])

    if test == 9:
        bins_o = np.array([ 0.07724977, 0.11431462, 0.66589757, 0.99628085])
        a__o   = np.array([ 1.14393907, 1.40455105, 2.54546141, 0])
        bins_r = np.array([ 0.03293035, 0.32737579, 0.84678898, 0.94612517])

    if test == 10:
        bins_o = np.array([ 0.28746837, 0.552113  , 0.61751934, 0.67614172])
        a__o   = np.array([ 0.5591539 , 1.31533187, 1.70198906, 0])
        bins_r = np.array([ 0.01708962, 0.45912355, 0.50794393, 0.95216739])

    if test == 11:
        bins_o = np.array([ 0.37363107, 0.89876499, 0.96997939, 0.98359356])
        a__o   = np.array([ 0.12227489, 1.4299947 , 2.6660467 , 0])
        bins_r = np.array([ 0.21842809, 0.29216228, 0.37951693, 0.75237436])

    if test == 12:
        bins_o = np.array([ 0.38063107, 0.89876499, 0.96997939, 0.98359356])
        a__o   = np.array([ 0.12227489, 1.4299947 , 2.6660467 , 0])
        bins_r = np.array([ 0.21842809, 0.29216228, 0.37951693, 0.75237436])

    if test == 13:
        bins_o = np.sort(np.random.rand(4))
        a__o = 3 * np.sort(np.random.rand(4))
        bins_r = np.sort(np.random.rand(4))
        print (bins_o)
        print (a__o)
        print (bins_r)

    plt.step(bins_o, a__o,  'o-', where='post')
    plt.ylim(0, 4)
    plt.xlim(-2, 8)
    #print (a__o[:-1] * (bins_o[1:] - bins_o[:-1])).sum()

    R__ro = ReSamplingMatrixNonUniform_fullArray(bins_resam = bins_r, bins_orig = bins_o)
    a__r = R__ro.dot(a__o[:-1].T)
    a__r = np.hstack([a__r, 0])
    plt.step(bins_r, a__r, '*-', ms=10, where='post')
    #print (a__r[:-1] * (bins_r[1:] - bins_r[:-1])).sum()

    R2__ro = ReSamplingMatrixNonUniform_sparseArray(bins_resam = bins_r, bins_orig = bins_o)
    a2__r = R2__ro.dot(a__o[:-1].T)
    a2__r = np.hstack([a2__r, 0])
    plt.step(bins_r, a2__r, '^-', ms=10, where='post')
    #print (a2__r[:-1] * (bins_r[1:] - bins_r[:-1])).sum()

    print ('Test', test, np.all(R2__ro == R__ro))
    #print R__ro
    #print R2__ro.toarray()
# --------------------------------------------------------------------

# --------------------------------------------------------------------
def getBinsLims(bins):
    '''
    Simply returns lower and upper edge of bins, plus bin centres and number of bins.
    '''

    if len(bins.shape) == 1:
        bins_low = bins[:-1]
        bins_upp = bins[1:]
        bins_size = bins_upp - bins_low
    else:
        bins_low = bins[0]
        bins_upp = bins[1]
        bins_size = bins[2]

    bins_cen = (bins_upp + bins_low) / 2.
    Nbins = len(bins_cen)

    return bins_low, bins_upp, bins_cen, bins_size, Nbins
# --------------------------------------------------------------------

# --------------------------------------------------------------------
def calcCovMatrix(sig_x, sig_y, rho_xy):

    var_x = sig_x**2
    var_y = sig_y**2

    var_xy = sig_x * sig_y * rho_xy

    cov__xy = np.array([ [var_x, var_xy], [var_xy, var_y] ])

    return cov__xy
# --------------------------------------------------------------------
