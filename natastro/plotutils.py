import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib import ticker
import time

from . import utils

#=========================================================================
def label(key, labels, short_labels = None, short = False):
    if key in labels.keys():
        lab = labels[key]
        if short:
            lab = short_labels[key]
    else:
        lab = key.replace('_', ' ')
    return lab
#=========================================================================

#=========================================================================
def label_panels(fig, x = 0.05, y = 0.95, labels = None, addToLabel = None, exclude_subp = None, va='top', **kwargs):
    if exclude_subp is None:
        exclude_subp = []

    import string
    if labels is None:
        labels = list(string.ascii_lowercase)
        labels = ['(%s)' % a for i, a in enumerate(labels)]

    if addToLabel is not None:
        labels = ['%s %s' % (a, b) for a, b in zip(labels, addToLabel)]

    for i, ax in enumerate(fig.get_axes(), start=1):
        if i not in exclude_subp:
            ax.text(x, y, labels[0], transform=ax.transAxes, va = va, **kwargs)
            labels.pop(0)
#=========================================================================

#=========================================================================
# http://stackoverflow.com/questions/14907062/matplotlib-aspect-ratio-in-subplots-with-various-y-axes
def adjust_subplots(fig, exclude_subp = None, **kwargs):
    if exclude_subp is None:
        exclude_subp = []

    for i, ax in enumerate(fig.get_axes(), start=1):
        if i not in exclude_subp:
            adjust_subplot(ax, **kwargs)

def adjust_subplot(ax, desired_box_ratioN = 1.):
    temp_inverse_axis_ratioN = abs( (ax.get_xlim()[1] - ax.get_xlim()[0])/(ax.get_ylim()[1] - ax.get_ylim()[0]) )
    ax.set(aspect = desired_box_ratioN * temp_inverse_axis_ratioN, adjustable='box-forced')
#=========================================================================

#=========================================================================
def title_date(title = None, y = 0.999, **kwargs):
    t = time.strftime("%d/%b/%Y %H:%M:%S")
    if title is None:
        suptitle = t
    else:
        suptitle = r'%s %s' % (title, t)

    suptitle = suptitle.replace('_', ' ')
    plt.suptitle(suptitle, y = y, **kwargs)
#=========================================================================

#=========================================================================
def fix_ticks(ax, nx = 4, ny = 4, fx = 5, fy = 5, steps = None, x = True, y = True, **kwargs):
    if steps is None:
        steps = [1, 2, 5, 10]

    if (fx <= 0):
        ax.tick_params(axis=u'x', which=u'both', length=0)
    elif x:
        ax.xaxis.set_major_locator( ticker.MaxNLocator(nbins =    nx+1, steps=steps, **kwargs) )
        ax.xaxis.set_minor_locator( ticker.MaxNLocator(nbins = fx*nx+1, steps=steps, **kwargs) )
        
    if (fy <= 0):
        ax.tick_params(axis=u'y', which=u'both', length=0)
    if y:
        ax.yaxis.set_major_locator( ticker.MaxNLocator(nbins =    ny+1, steps=steps, **kwargs) )
        ax.yaxis.set_minor_locator( ticker.MaxNLocator(nbins = fy*ny+1, steps=steps, **kwargs) )
#=========================================================================

#=========================================================================
def fix_colorbar_ticks(cb = None, ny = 4, steps = None, **kwargs):
    if steps is None:
        steps = [1, 2, 5, 10]
    if cb is None:
        cb = plt.colorbar()
    tick_locator = ticker.MaxNLocator(nbins = ny+1, steps=steps, **kwargs)
    cb.locator = tick_locator
    cb.update_ticks()
#=========================================================================


#=========================================================================
def tight_colorbar(im, size='7%', pad='3%', zorder=10, ax=None, loc='right', **kwargs):
    if ax is None:
        ax = plt.gca()
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size, pad=pad, zorder = zorder)
    cbar = plt.colorbar(im, cax=cax, **kwargs)
    if (loc == 'top'):
        cax.xaxis.set_label_position(loc)
        cax.xaxis.set_ticks_position(loc)
    return cbar
#=========================================================================


#=========================================================================
def plot_identity(x, y, plot_fit = False, color='k', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    global_min = min(x.min(), y.min())
    global_max = max(x.max(), y.max())
    dx = global_max - global_min
    xmin = global_min - 0.1*dx
    xmax = global_max + 0.1*dx

    ax.plot([xmin, xmax], [xmin, xmax], color=color, **kwargs)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    if plot_fit:
        flag = (x > -999) & (y > -999)
        x = x[flag]
        y = y[flag]

        # Linear fit y(x) = a1 x + b1
        p = np.polyfit(x, y, 1)
        _x1 = np.array([xmin, xmax])
        a1, b1 = p[0], p[1]
        _y1 = a1 * _x1 + b1
        ax.plot(_x1, _y1, 'k:')

        # Linear fit x(y) = a2 y + b2
        p = np.polyfit(y, x, 1)
        _y2 = np.array([xmin, xmax])
        a2, b2 = p[0], p[1]
        _x2 = a2 * _y2 + b2
        ax.plot(_x2, _y2, 'k:')

        # Bisector fit: Formulae provided by Laerte
        # (except that I use y = a x + b whereas y = a + b x for him)
        a2 = 1 / a2
        a = ( a1 * a2 - 1 + np.sqrt( (1 + a1**2) * ( 1 + a2**2) ) ) / (a1 + a2)
        xm = np.mean(x)
        ym = np.mean(y)
        b = ym - (a * xm)
        _x3 = np.array([xmin, xmax])
        _y3 = a * _x3 + b
        ax.plot(_x3, _y3, 'k--')
#=========================================================================


#=========================================================================
def save(fig, outFile, pdfFromEps = False, epsFromPdf = False, pdftocairo = False, **kwargs):

    if pdfFromEps:
        epsFile = os.path.splitext(outFile)[0] + '_tmp.eps'
        fig.savefig(epsFile, **kwargs)
        os.system("ps2pdf -dEPSCrop %s %s" % (epsFile, outFile))
        os.remove(epsFile)
    elif epsFromPdf:
        pdfFile = os.path.splitext(outFile)[0] + '_tmp.pdf'
        fig.savefig(pdfFile, **kwargs)
        os.system("pdftocairo -ps %s %s" % (pdfFile, outFile))
        os.remove(pdfFile)
    elif pdftocairo:
        tmpFile = os.path.splitext(outFile)[0] + '_tmp.pdf'
        fig.savefig(tmpFile, **kwargs)
        os.system("pdftocairo -pdf %s %s" % (tmpFile, outFile))
        os.remove(tmpFile)
    else:
        fig.savefig(outFile, **kwargs)
#=========================================================================

#=========================================================================
def save_figs_paper(fig, fileName, raster=True, dpi=300, savePDF = True, saveEPS = True,
                    pdftocairo = True, epsFromPdf = True):
    if raster == True:

        for ax in fig.get_axes():
            ax.set_rasterization_zorder(1)
        if savePDF:
            save(fig, '%s.pdf' % fileName, rasterized=True, dpi=dpi, pdftocairo=pdftocairo)
        if saveEPS:
            save(fig, '%s.eps' % fileName, rasterized=True, dpi=dpi, epsFromPdf=epsFromPdf)

    else:

        for ax in fig.get_axes():
            ax.set_rasterization_zorder(-10)
        save(fig, '%s.pdf' % fileName, rasterized=False, dpi=dpi, pdftocairo=True)
#=========================================================================

#=========================================================================
def plotLatex(fn):
    def _plotLatex(override_params = {}, **kwargs):
        # Load normal setup
        plotSetup(**kwargs)

        # Add LaTeX configs
        params = {
            'text.usetex': True,
            'text.latex.preamble': [
                r'\usepackage{amsmath}',
                r'\usepackage{amssymb}',
                r'\usepackage{siunitx}',
                r'\sisetup{detect-all}',
            ],
        }
        plt.rcParams.update(params)

        # Load special function setup
        fn()

        # Load overridden paramters
        plt.rcParams.update(override_params)

    return _plotLatex
#=========================================================================

#=========================================================================
def plotSetupHershey(**kwargs):
    plotSetup(**kwargs)
    plt.rcParams['font.family'] = ['AVHershey Complex']
#=========================================================================

#=========================================================================
@plotLatex
def plotSetupMinion():
    params = {
        'text.latex.preamble': [
            r'\usepackage[lf, mathtabular]{MinionPro}',
            r'\usepackage{MnSymbol}',
        ]
    }
    plt.rcParams.update(params)

#plt.rcParams.update(params)
#=========================================================================

#=========================================================================
def plotSetup(fig_width_pt=448.07378, aspect=0.0, fontsize=None, lw=1.2, override_params={}):
    # stolen from Andre...

    # From http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
    inches_per_pt = 1.0 / 72.27
    if aspect == 0.0:
        golden_mean = (np.sqrt(5) - 1.0) / 2.0
        aspect = golden_mean
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * aspect
    fig_size = (fig_width, fig_height)

    # Default fontsize
    if (fontsize is None):
        fontsize = int(0.08 * fig_width / inches_per_pt)

    params = {'backend': 'pdf',
              'interactive'  : True,

              'axes.labelsize': fontsize,
              'axes.titlesize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'legend.fontsize': fontsize,

              'font.size': fontsize,
              'font.family': 'serif',
              'font.serif': ['Times New Roman', ],
              'font.sans-serif': ['Helvetica', ],
              'mathtext.fontset': 'stix',
              'text.usetex': False,

              'xtick.major.size': 6,
              'ytick.major.size': 6,
              'xtick.minor.size': 3,
              'ytick.minor.size': 3,

              'xtick.direction': 'in',
              'ytick.direction': 'in',
              'xtick.top':   True,
              'ytick.right': True,

              'figure.subplot.hspace': 0.2,
              'figure.subplot.wspace': 0.2,
              'figure.subplot.left':   0.125,
              'figure.subplot.right':  0.9,
              'figure.subplot.bottom': 0.1,
              'figure.subplot.top':    0.9,

              'figure.facecolor' : 'white' ,
              'figure.figsize'   : fig_size,

              'lines.linewidth'        : 1.0 * lw,    # line width in points
              'lines.markeredgewidth'  : 0.5 * lw,    # the line width around the marker symbol
              'patch.linewidth'        : 1.0 * lw,    # edge width in points
              'axes.linewidth'         : 1.0 * lw,    # edge linewidth
              'xtick.major.width'      : 0.5 * lw,    # major tick width in points
              'xtick.minor.width'      : 0.5 * lw,    # minor tick width in points
              'ytick.major.width'      : 0.5 * lw,    # major tick width in points
              'ytick.minor.width'      : 0.5 * lw,    # minor tick width in points
              'grid.linewidth'         : 0.5 * lw,    # in points

              'savefig.dpi': 300,
              }

    plt.rcdefaults()
    plt.rcParams.update(params)
    plt.rcParams.update(override_params)
#=========================================================================


#=========================================================================
def sample_plot():

    x1 = [1,2,3]
    y1 = [4,5,6]
    x2 = [1,2,3]
    y2 = [5,5,5]

    plt.clf()

    # @@@> Some usefulf tricks: line width, legend location, latex for label, fontsize
    fig311 = plt.subplot(3, 1, 1)
    plt.plot(x1, y1, label=r"bidu", ls="-", color='r', lw = 5)
    plt.plot(x1, y2, label=r"bugu", ls="-", color='y')
    plt.legend(loc = 2)
    plt.xlabel(r'H$\mathrm{H}\alpha^V_0$ [km s$^{-1}$]')
    plt.ylabel(r'$\langle \log \, t_\star \rangle_M$')
    plt.text(2, 5.5, r"an equation: $E=mc^2$", color = "b")

    # @@@> Tickmarks
    plt.minorticks_on()
    plt.xticks([1.1, 1.7, 2.3, 2.9])
    plt.yticks(np.arange(4.,6.1,0.5))

    # @@@> More tickmarks
    fig312 = plt.subplot(3, 1, 2)

    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    majorLocator   = MultipleLocator(20)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator   = MultipleLocator(5)

    t = np.arange(0.0, 100.0, 0.1)
    s = np.sin(0.1*np.pi*t) * np.exp(-t*0.01)

    plt.plot(t,s)

    fig312.xaxis.set_major_locator(majorLocator)
    fig312.xaxis.set_major_formatter(majorFormatter)
    fig312.xaxis.set_minor_locator(minorLocator)


    fig313 = plt.subplot(3, 1, 3)


    # @@@> Adjust xgutter/ygutter
    plt.subplots_adjust(hspace=.5)
#=========================================================================


#=========================================================================
def plot_stat_Nbins(x, y, Nbins = 10, **kwargs):
    # https://stackoverflow.com/questions/39418380/histogram-with-equal-number-of-points-in-each-bin
    f = (x > -999.) & (y > -999.)
    x = x[f]
    y = y[f]
    Nx = len(x)
    x_bins = np.interp(np.linspace(0, Nx, Nbins + 1), np.arange(Nx), np.sort(x))
    return _plot_stat(x, y, x_bins, **kwargs)

def plot_stat_Npoi(x, y, Npoi = 10000, **kwargs):
    # https://stackoverflow.com/questions/39418380/histogram-with-equal-number-of-points-in-each-bin
    Nx = len(x)
    Nbins = int(len(x) / Npoi)
    x_bins = np.interp(np.linspace(0, Nx, Nbins + 1), np.arange(Nx), np.sort(x))
    return _plot_stat(x, y, x_bins, **kwargs)

def _plot_stat(x, y, x_bins, colour='k', stat='median', plot=True, ax=None, **kwargs):
    if stat.startswith('p.'):
        p = float(stat.replace('p', '')) * 100.
        stat = lambda x: np.percentile(x, p)

    bin_stats, bin_edges, binnumber = scipy.stats.binned_statistic(x, y, statistic=stat,    bins=x_bins)
    counts,    bin_edges, binnumber = scipy.stats.binned_statistic(x, y, statistic='count', bins=x_bins)

    x_mid = (x_bins[:-1] + x_bins[1:]) / 2
    _f = (counts >= 5)

    if np.any(_f):
        if plot:
            if ax is None:
                ax = plt.gca()
            ax.plot(x_mid[_f], bin_stats[_f], color=colour, **kwargs)
        return x_mid[_f], bin_stats[_f]
    else:
        return None

def plot_median_bins(x, y, z, dz=0.15, zlim = None, cinv = False, zlab = None,
                     labBin = True, labBin_x = 0.95, **kwargs):
    import seaborn as sns

    f = (x > -999.) & (y > -999.) & (z > -999.)
    x = x[f]
    y = y[f]
    z = z[f]

    # Calc z bins
    if zlim is None:
        zmin, zmax = z.min(), z.max()
    else:
        zmin, zmax = zlim

    # Bins in z
    z_bins = np.arange(zmin, zmax+dz, dz)
    zmid = (z_bins[1:] + z_bins[:-1]) / 2.
    Nz = len(z_bins)
    colours = sns.hls_palette(Nz, l=.4, s=.95)[:-1]
    if cinv:
        colours = colours[::-1]

    ax = plt.gca()
    if zlab is not None:
        ax.text(labBin_x+0.02, 0.07*(Nz), zlab, transform=ax.transAxes, ha='right', fontsize='xx-small')

    for i_bin, (bin_low, bin_upp) in enumerate(zip(z_bins[:-1], z_bins[1:])):
        colour = colours[i_bin]
        _f = (z >= bin_low) & (z < bin_upp)

        if (_f.sum() > 10):
            x_bin = x[_f]
            y_bin = y[_f]

            plot_stat_Nbins(x_bin, y_bin, colour=colour, **kwargs)
            if labBin:
                ax.text(labBin_x, 0.07*(i_bin + 1), '%.2f' % zmid[i_bin], transform=ax.transAxes, color=colour, ha='right', fontsize='xx-small')
#=========================================================================

#=========================================================================
def plot_stat_dx(x, y, dx=None, Nbins=5, min_pts=10, **kwargs):
    f = (x > -999.) & (y > -999.)
    x = x[f]
    y = y[f]
    if dx is None:
        dx = (x.max() - x.min()) / Nbins
    x_bins = np.arange(x.min(), x.max()+dx, dx)
    Nx = np.digitize(x, x_bins)
    #ff = (Nx >= min_pts)
    return _plot_stat(x, y, x_bins, **kwargs)

def plot_median_dx_bins(x, y, z, dz=0.15, dx=0.15, zlim = None, cinv = False, zlab = None,
                        labBin = True, labBin_x = 0.95, **kwargs):
    import seaborn as sns

    f = (x > -999.) & (y > -999.) & (z > -999.)
    x = x[f]
    y = y[f]
    z = z[f]

    # Calc z bins
    if zlim is None:
        zmin, zmax = z.min(), z.max()
    else:
        zmin, zmax = zlim

    # Bins in z
    z_bins = np.arange(zmin, zmax+dz, dz)
    zmid = (z_bins[1:] + z_bins[:-1]) / 2.
    Nz = len(z_bins)
    colours = sns.hls_palette(Nz, l=.4, s=.95)[:-1]
    if cinv:
        colours = colours[::-1]

    ax = plt.gca()
    if zlab is not None:
        ax.text(labBin_x+0.02, 0.07*(Nz), zlab, transform=ax.transAxes, ha='right', fontsize='xx-small')

    for i_bin, (bin_low, bin_upp) in enumerate(zip(z_bins[:-1], z_bins[1:])):
        colour = colours[i_bin]
        _f = (z >= bin_low) & (z < bin_upp)

        if (_f.sum() > 10):
            x_bin = x[_f]
            y_bin = y[_f]

            plot_stat_dx(x_bin, y_bin, dx, colour=colour, **kwargs)
            if labBin:
                ax.text(labBin_x, 0.07*(i_bin + 1), '%.2f' % zmid[i_bin], transform=ax.transAxes, color=colour, ha='right', fontsize='xx-small')
#=========================================================================

#=========================================================================
def clean_vars(x, y, flag=None):
    # Clean data
    f = (~utils.mask_minus999(x).mask) & (~utils.mask_minus999(y).mask)
    if isinstance(x, np.ma.MaskedArray):
        f &= (~x.mask)
    if isinstance(y, np.ma.MaskedArray):
        f &= (~y.mask)
    if flag is not None:
        f &= flag
    if isinstance(f, np.ma.MaskedArray):
        f &= (~f.mask)
    xx = x[f]
    yy = y[f]
    ff = utils.mask_minus999(f, fill=0)
    return xx, yy, ff

def clean_vars_table(t, varx, vary, **kwargs):
    return clean_vars(t[varx], t[vary], **kwargs)
#=========================================================================

#=========================================================================
def plot_scatter_xy(ax, t, varx, vary, color=None, labels=None, flag=None,
                    xlim=None, ylim=None, axx=None, axy=None,
                    pltIdentity=False,
                    **kwargs):
    '''
    Loosely emulates plot_xyz from sm.
    Natalia@Krakow - 12/Apr/2019.
    '''

    # Clean data
    x, y, f = clean_vars_table(t, varx, vary, flag=flag)

    # Create axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if axx is None:
        divider = make_axes_locatable(ax)
        axx = divider.append_axes('top'  , size= '20%', pad=0.0, sharex=ax)
    if axy is None:
        axy = divider.append_axes('right', size= '20%', pad=0.0, sharey=ax)

    ax.scatter(x, y, c=color, **kwargs)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    Np = f.sum()

    if (Np >= 1):
        bins = 20
        if (Np == 1): bins = 1
        hist, bin_lims = np.histogram(x, bins=bins)
        axx.hist(x, bins=bins, histtype='step', color=color, weights=np.full(len(x), 1./np.max(hist)))
        axx.set_ylim(0, 1.2)
        hist, bin_lims = np.histogram(y, bins=bins)
        axy.hist(y, bins=bins, histtype='step', color=color, weights=np.full(len(y), 1./np.max(hist)), orientation='horizontal')
        axy.set_xlim(0, 1.2)

    fix_ticks(axx, fx = 10, fy = 0)
    fix_ticks(axy, fx = 0, fy = 10)
    fix_ticks(ax, fx = 10, fy = 10)

    plt.setp(axx.get_xticklabels(), visible=False)
    plt.setp(axy.get_yticklabels(), visible=False)
    plt.setp(axx.get_yticklabels(), visible=False)
    plt.setp(axy.get_xticklabels(), visible=False)

    if labels is None:
        labels = {}
    ax.set_xlabel(label(varx, labels))
    ax.set_ylabel(label(vary, labels))

    if pltIdentity:
        plot_identity(x, y, ax=ax)

    return ax, axx, axy, Np
#=========================================================================
