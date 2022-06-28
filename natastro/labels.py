'''
Clever calculator for dust stuff.

Natalia@UFSC - 30/Aug/2018
'''

# --------------------------------------------------------------------
def get_labels(ps='stix'):
    
    labels = {
        'at_flux'               : r'$\langle \log t_\star \rangle \;\mathrm{[yr]}$'                                                             ,
        'AV_Balmer'             : r'$A_V^\mathrm{neb} \;\mathrm{[mag]}$'                                                                        ,
        'D4000'                 : r'$D_n(4000)$'                                                                                                ,
        'El_EW_6563'            : r'$W_\mathrm{H\alpha}^\mathrm{obs} \;\mathrm{[\AA]}$'                                                         ,
        'El_F_6563'             : r'$F_\mathrm{H\alpha}^\mathrm{obs} \;\mathrm{[10^{-17} erg s^{-1} cm^{-2}]}$'                                 ,
        'El_SN_4861'            : r'$S/N_\mathrm{H\beta}$'                                                                                      ,
        'HaHb'                  : r'$\mathrm{H\alpha}/\mathrm{H\beta}$'                                                                         ,
        'LHa_I/LHa_G'           : r'$L_\mathrm{IFS}/L_\mathrm{G}$'                                                                              ,
        'LHa_obs/Mcor'          : r'$L(\mathrm{H\alpha})_\mathrm{G,obs}/M_\mathrm{\star,G} \;\mathrm{[L_\odot \, M_\odot^{-1}]}$'               ,
        'Lobn'                  : r'$L_\mathrm{\lambda=5635\,\normalsize\AA} \;\mathrm{[L_\odot \;\AA^{-1}]}$'                                  ,
        'S2Ha'                  : r'[S {\scshape ii}]/$\mathrm{H\alpha}$'                                                                       ,
        'log a/a_ave'           : r'$\log\, e^{\tau_\mathrm{j}}/ \langle e^{\tau_\mathrm{j}} \rangle$'                                          ,
        'log_FHb_ave'           : r'$\log \, \langle F_\mathrm{H\beta} \rangle$ [erg s$^{-1}$ cm$^{-2}$]'                                       ,
        'log_LHa'               : r'$\log L_\mathrm{H\alpha} \;\mathrm{[L_\odot]}$'                                                             ,
        'log_LHa_G'             : r'$\log L_\mathrm{G}$'                                                                                        ,
#        'log_N2Ha'              : r'log [N {\scshape ii}]/$\mathrm{H\alpha}$'                                                                   ,
#        'log_O3Hb'              : r'log [O {\scshape iii}]/$\mathrm{H\beta}$'                                                                   ,
        'log_N2Ha'              : r'log [N II]/$\mathrm{H\alpha}$'                                                                              ,
        'log_O3Hb'              : r'log [O III]/$\mathrm{H\beta}$'                                                                              ,
        'log_SFR_Ha'            : r'$\log \mathrm{SFR} \;\mathrm{[M_\odot yr^{-1}]}$'                                                           ,
        'log_surfHa'            : r'$\log \Sigma_\mathrm{H\alpha} \;\mathrm{[L_\odot\, kpc^{-2}]}$'                                             ,
        'log_surfHa_dr'         : r'$\log \Sigma_\mathrm{H\alpha}^\mathrm{dr} \;\mathrm{[L_\odot\, kpc^{-2}]}$'                                 ,
        'log_WHa'               : r'$\log W_\mathrm{H\alpha} \;\mathrm{[\AA]}$'                                                                 ,
        'log_WHa_dr'            : r'$\log W_\mathrm{H\alpha}^\mathrm{dr} \;\mathrm{[\AA]}$'                                                     ,
        'Mcor'                  : r'$M_\mathrm{\star} \;\mathrm{[M_\odot]}$'                                                                    ,
        'pixel_scale_pc'        : r'Sampling [pc]'                                                                                              ,
        'R_V'                   : r'$R_V$'                                                                                                      ,
        'SurfMass'              : r'$\Sigma_\mathrm{M_\star} \;\mathrm{[M_\odot\, kpc^{-2}]}$'                                                  ,
        'surfHa'                : r'$\Sigma_\mathrm{H\alpha} \;\mathrm{[L_\odot\, kpc^{-2}]}$'                                                  ,
        }
    
    if (ps != 'minion'):
        labels = {k: r'$\mathdefault{%s}$' % v.replace('$', '').replace('\log', '\log\,') for k, v in labels.items()}

    return labels
# --------------------------------------------------------------------
