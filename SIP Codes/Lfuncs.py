import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d

from astropy.table import Table, Column, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits

from sklearn.neighbors import KDTree
from astroML.stats import binned_statistic_2d

import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
import corner

import pickle
import math
import matplotlib.colors as mc
import colorsys
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import scipy

#-------SETUP-------
GLPDF_PREFACTOR = -0.5*np.log(2*np.pi)
glpdf = lambda x, mu, sig: GLPDF_PREFACTOR - np.log(sig) - 0.5*((x-mu)/sig)**2

y_lim_factor = 0.3 # multiplicative factor to add to the max of the 50th % PDF for the y limit for model/data velocity histogram plots
MINF = -np.inf
FAIL_RET = MINF, (None, None)  #useful if blobs come into play...

# flat_prior_widths = [pri_frotdiskmax, (1-pri_frothalomin), (pri_disk_dispmax-1), (pri_halo_dispmax-1), .5]
# FLAT_PRIOR_LN = np.sum(-np.log(flat_prior_widths))
#
# beta=1
# flat_prior=True
def load_disk_model(diskmodel_path):
    return Table.read(diskmodel_path, format='ascii',
                      names=['Radius_arcmin', 'Radius_kpc', 'Vrot_kms', 'Delta_Vrot', 'i_deg', 'PA_deg'])

diskmodel = load_disk_model('./Kam2017_table4.dat')
histylabel = 'Probability Density'
legloc='best'
ylim_pdf_pad = 0.2
pri_disk_dispmax = 50.
pri_halo_dispmax = 100.
pri_hfrac_max = 0.85
pri_frothalomin = -1.3 # allows halo to counter-rotate. maximum inherently 1: can't have halo rotating faster than disk
pri_frotdiskmax = 1.5 # allows disk to rotate faster than HI. minimum inherently 0: don't allow disk to counter-rotate relative to stars

#-------------Set centre properties, load HI model------------
def distance(dmod):
    dist = 10.**((dmod + 5.)/5.)/10**3
    return dist

m33_sys = -180.0 # pm 1 km/s; vanderMarel et al. 2008; Kam et al. Table 1 \pm 3
# m33_sys_wunit = m33_sys*u.km/u.s
m33_dmod = 24.67 # what Anil assumed  # Kam et al 2017 uses 840 kpc; but tilted ring model in arcmin, so this is not crucial. Alt 24.54 or 24.75: closest/furthest distances quoted in Kam 2015
m33_dist = distance(m33_dmod)
m33coord = SkyCoord(ra='01h33m50.9s', dec='+30d39m36s', distance=m33_dist*u.kpc, unit=(u.hourangle, u.deg))

m33_pa = (22.5 + 180.0)*u.deg # \pm 1 ; Table 1 Kam 2017; +180 to match definition used in HI model
m33_pa_amanda = (22 + 180.0)*u.deg  # what Amanda assumes for radial deprojection; +180 to match definition used in HI model
m33_inclination_kam = 52.0*u.deg # \pm 3 ; Table 1 Kam 2017
m33_inclination = 54.0*u.deg # what Amanda assumes for radial deprojection

#---------Compare disk(lagging+nolag) to disk+(static)halo, SET 2--------
# I.E.: is it preferred to have a lagging disk or not, with or without a static halo?
def loglike_Ldiskhalo(paramsL,vels,modv,extras):
	disklag, disksig, fhalo, halocen, halosig = paramsL
	fdisk = 1 - fhalo
	#incl,posang,f_vrot = extras
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

    #per-component llikelihoods
	halolpdf = glpdf(vels, halocen, halosig)
	disklpdf = glpdf(vels, modv_dlag, disksig)
	blob = np.c_[halolpdf,disklpdf]
	both_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(fhalo)+halolpdf)
	return np.sum(both_likelihood),blob

def loglike_diskhalo(paramsL,vels,modv,extras):
	disklag, disksig, fhalo, halocen, halosig = paramsL
	fdisk = 1 - fhalo

	#per-component llikelihoods
	halolpdf = glpdf(vels, halocen, halosig)
	disklpdf = glpdf(vels, modv, disksig)
	blob = np.c_[halolpdf,disklpdf]
	both_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(fhalo)+halolpdf)
	return np.sum(both_likelihood),blob

def loglike_Ldisknohalo(paramsL,vels,modv,extras):
	disklag, disksig, fhalo, halocen, halosig = paramsL
	fdisk = 1 - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	disklpdf = glpdf(vels, modv_dlag, disksig)
	blob = np.c_[np.zeros(len(disklpdf)),disklpdf]
	return np.sum(disklpdf),blob

def loglike_disknohalo(paramsL,vels,modv,extras):
	disklag, disksig, fhalo, halocen, halosig = paramsL
	fdisk = 1 - fhalo
	disklpdf = glpdf(vels, modv, disksig)
	blob = np.c_[np.zeros(len(disklpdf)),disklpdf]
	return np.sum(disklpdf),blob

#---------Compare disk (allowed to lag HI) to disk+(static)halo, SET 1--------
# I.E.: is it preferred to have a halo component at all?
params = 'disklag, disksig, fhalo, halocen, halosig'.split(', ')

def priortrans(x):  #read these as 'how do I get from 0 to the minimum bound, and 1 to the maximum bound?'
    disklag_prit = x[0] * pri_frotdiskmax #uniform between 0 and frotdiskmax
    disksig_prit = x[1] * (pri_disk_dispmax-1) + 1 #uniform between 1 and 100
    fhalo_prit = x[2] * 0.5 #uniform between 0 and 0.5 (at least50% of things must be disk)
    halocen_prit = x[3] * 200 - 300 #uniform between -300 and -100
    halosig_prit = x[4] * (pri_halo_dispmax-1) + 1
    return np.array([disklag_prit, disksig_prit, fhalo_prit, halocen_prit, halosig_prit])
#Replace the two below with equiv funcs above, since identical
# def loglike_both(params,vels,modv,extras):
#     disklag, disksig, fdisk, halocen, halosig = params
#     fdisk = 1 - fhalo
# 	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])
#
#     #per-component llikelihoods
#     halolpdf = glpdf(vels, halocen, halosig)
#     disklpdf = glpdf(vels, modv_dlag, disksig)
#
#     both_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(fhalo)+halolpdf)
#     return np.sum(both_likelihood)

# def loglike_nohalo(params,vels,modv,extras):
#     disklag, disksig, fdisk, halocen, halosig = params
#     fdisk = 1 - fhalo
# 	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])
#
#     disklpdf = glpdf(vels, modv_dlag, disksig)
#     return np.sum(disklpdf)

#---------Compare disk+(static)halo to disk+(rotating,lagging)halo, SET 3------
#I.E.: Is it preferred to have a rotating or non-rotating halo?
params2 = 'disklag, disksig, fhalo, halocen, halosig, halolag'.split(', ')

def priortrans2(x):
    disklag_prit = x[0] * pri_frotdiskmax #uniform between 0 and frotdiskmax
    disksig_prit = x[1] * (pri_disk_dispmax-1) + 1
    fhalo_prit = x[2] * pri_hfrac_max
    halocen_prit = x[3] * 250 - 300 #was 200
    halosig_prit = x[4] * (pri_halo_dispmax-1) + 1
    halolag_prit = x[5] * (1-pri_frothalomin) + pri_frothalomin
    return np.array([disklag_prit, disksig_prit, fhalo_prit, halocen_prit, halosig_prit, halolag_prit])

def loglike_rothalo(paramsL,vels,modv,extras):
	disklag, disksig, fhalo, halocen, halosig, halolag = paramsL
	fdisk = 1 - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])
	modv_hlag = m33_sys+halolag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if halosig < disksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

    #per-component llikelihoods
	rhalolpdf = glpdf(vels, modv_hlag, halosig)
	disklpdf = glpdf(vels, modv_dlag, disksig)
	blob = np.c_[rhalolpdf,disklpdf]

	both_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(fhalo)+rhalolpdf)
	return np.sum(both_likelihood),blob

def loglike_nrothalo(paramsL,vels,modv,extras):
	disklag, disksig, fhalo, halocen, halosig, halolag = paramsL
	fdisk = 1 - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if halosig < disksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	#per-component llikelihoods
	halolpdf = glpdf(vels, halocen, halosig)
	disklpdf = glpdf(vels, modv_dlag, disksig)
	blob = np.c_[halolpdf,disklpdf]

	both_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(fhalo)+halolpdf)
	return np.sum(both_likelihood),blob

def loglike_nohalo(paramsL,vels,modv,extras):
	disklag, disksig, fhalo, halocen, halosig, halolag = paramsL
	fdisk = 1 - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if halosig < disksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	disklpdf = glpdf(vels, modv_dlag,disksig)
	blob = np.c_[np.zeros(len(disklpdf)),disklpdf]
	return np.sum(disklpdf),blob

#-------Compare 2-component disk+(static)halo to 1-component disk+(static)halo, SET 4-------
#I.E.: Could we explain halo rotation instead as an extra disk component otherwise missing?
params3 = 'disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, fhalo'.split(', ')

def priortrans3(x):
    disklag_prit = x[0] * pri_frotdiskmax #uniform between 0 and frotdiskmax
    disksig_prit = x[1] * (pri_disk_dispmax-1) + 1

    tdisklag_prit = x[2] * pri_frotdiskmax #uniform between 0 and frotdiskmax
    tdisksig_prit = x[3] * (pri_disk_dispmax-1) + 1

    halocen_prit = x[5] * 250 - 300
    halosig_prit = x[6] * (pri_halo_dispmax-1) + 1

    ftdisk_prit = x[4] * 0.5
    fhalo_prit = x[7] * 0.5

    return np.array([disklag_prit, disksig_prit, tdisklag_prit, tdisksig_prit, ftdisk_prit, halocen_prit, halosig_prit, fhalo_prit, halolag_prit])

def loglike_all3(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, fhalo = paramsL
	fdisk = 1 - ftdisk - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])
	modv_tdlag = m33_sys+tdisklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if tdisksig < disksig: #or halosig < tdisksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	#per-component llikelihoods
	tdiskpdf = glpdf(vels, modv_tdlag, tdisksig)
	disklpdf = glpdf(vels, modv_dlag, disksig)
	halolpdf = glpdf(vels, halocen, halosig)

	disks_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(ftdisk)+tdiskpdf)
	blob = np.c_[halolpdf,disks_likelihood]
	all_likelihood = np.logaddexp(disks_likelihood, np.log(fhalo)+halolpdf)
	return np.sum(all_likelihood),blob

def loglike_nohalo3(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, fhalo = paramsL
	fdisk = 1 - ftdisk - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])
	modv_tdlag = m33_sys+tdisklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if tdisksig < disksig:# or halosig < tdisksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	#per-component llikelihoods
	tdiskpdf = glpdf(vels, modv_tdlag, tdisksig)
	disklpdf = glpdf(vels, modv_dlag, disksig)

	disks_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(ftdisk)+tdiskpdf)
	blob = np.c_[np.zeros(len(disks_likelihood)),disks_likelihood]
	return np.sum(disks_likelihood),blob

'''
#----------Compare disk+(static)halo to disk+(static)halo+uniform foreground, SET 6-----
#I.E.: is there a foreground pop we're missing that could be preferred to a rotating halo?
paramsbg = 'disklag, disksig, fhalo, halocen, halosig, fforeground'.split(', ')

def priortrans_ubkg(x):
    disklag_prit = x[0] * pri_frotdiskmax #uniform between 0 and frotdiskmax
    disksig_prit = x[1] * (pri_disk_dispmax-1) + 1
    fhalo_prit = x[2] * 0.5 + 0.5
    halocen_prit = x[3] * 200 - 300
    halosig_prit = x[4] * (pri_halo_dispmax-1) + 1
    fforegroud_prit = x[5] * 0.01
    return np.array([disklag_prit, disksig_prit, fhalo_prit, halocen_prit, halosig_prit, fforegroud_prit])

def loglike_both_ubkg(params,vels,modv,extras,vfg):
    disklag, disksig, fhalo, halocen, halosig, fforeground = params
    fdisk = 1 - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

    #per-component llikelihoods
    halolpdf = glpdf(vels, halocen, halosig)
    disklpdf = glpdf(vels, modv_dlag, disksig)
    foregroundpdf = vfg

    both_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(fhalo)+halolpdf)
    combined_likelihood = np.logaddexp(np.log(1-fforeground)+both_likelihood, np.log(fforeground)+foregroundpdf)
    return np.sum(combined_likelihood)

def loglike_nohalo_ubkg(params,vels,modv,extras,vfg):
    disklag, disksig, fhalo, halocen, halosig, fforeground = params
    fdisk = 1 - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

    disklpdf = glpdf(vels, modv_dlag, disksig)
    foregroundpdf = vfg

    combined_likelihood = np.logaddexp(np.log(1-fforeground)+disklpdf, np.log(fforeground)+foregroundpdf)
    return np.sum(combined_likelihood)
'''
#-------BIGMODEL, SET 5-------
#I.E. Try fitting 2-component lagging(or not)disk+rotlag halo - what combination of these has the most evidence?
# Note thick disk allowed to lag - tests whether 'thin' disk preferred to lag or not (since no evidence of lag in young pops)
params4 = 'disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo'.split(', ')

def priortrans4(x):
    disklag_prit = x[0] * pri_frotdiskmax #uniform between 0 and frotdiskmax
    disksig_prit = x[1] * (pri_disk_dispmax-1) + 1

    tdisklag_prit = x[2] * pri_frotdiskmax #uniform between 0 and frotdiskmax
    tdisksig_prit = x[3] * (pri_disk_dispmax-1) + 1

    # halocen_prit = x[5] * 250 - 300
    halocen_prit = x[5] * 5
    halosig_prit = x[6] * (pri_halo_dispmax-1) + 1
    halolag_prit = x[7] * (1-pri_frothalomin) + pri_frothalomin

    ftdisk_prit = x[4] * 0.5
    fhalo_prit = x[8] * 0.5

    return np.array([disklag_prit, disksig_prit, tdisklag_prit, tdisksig_prit, ftdisk_prit, halocen_prit, halosig_prit, halolag_prit, fhalo_prit])

def loglike_all4(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo = paramsL
	fdisk = 1 - ftdisk - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])
	modv_hlag = m33_sys+halolag*extras[2]*np.cos(extras[1])*np.sin(extras[0])
	modv_tdlag = m33_sys+tdisklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if tdisksig < disksig or halosig < tdisksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

    #per-component llikelihoods
	tdiskpdf = glpdf(vels, modv_tdlag, tdisksig)
	disklpdf = glpdf(vels, modv_dlag, disksig)
	rhalolpdf = glpdf(vels, modv_hlag, halosig)

	disks_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(ftdisk)+tdiskpdf)
	blob = np.c_[rhalolpdf,disks_likelihood]
	# all_likelihood = np.logaddexp(disks_likelihood, np.log(fhalo)+rhalolpdf)
	all_likelihood = scipy.special.logsumexp([np.log(fdisk)+disklpdf, np.log(ftdisk)+tdiskpdf,np.log(fhalo)+rhalolpdf],axis=0)
	return np.sum(all_likelihood),blob

def loglike_nrothalo4(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo = paramsL
	fdisk = 1 - ftdisk - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])
	modv_tdlag = m33_sys+tdisklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if tdisksig < disksig: #or halosig < tdisksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	#per-component llikelihoods
	halolpdf = glpdf(vels, halocen, halosig)
	tdiskpdf = glpdf(vels, modv_tdlag, tdisksig)
	disklpdf = glpdf(vels, modv_dlag, disksig)

	disks_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(ftdisk)+tdiskpdf)
	blob = np.c_[halolpdf,disks_likelihood]
	all_likelihood = np.logaddexp(disks_likelihood, np.log(fhalo)+halolpdf)
	return np.sum(all_likelihood),blob

def loglike_nohalo4(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo = paramsL
	fdisk = 1 - ftdisk - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])
	modv_tdlag = m33_sys+tdisklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if tdisksig < disksig:# or halosig < tdisksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	#per-component llikelihoods
	tdiskpdf = glpdf(vels, modv_tdlag, tdisksig)
	disklpdf = glpdf(vels, modv_dlag, disksig)
	disks_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(ftdisk)+tdiskpdf)
	blob = np.c_[np.zeros(len(disks_likelihood)),disks_likelihood]
	return np.sum(disks_likelihood),blob

def loglike_1disk4(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo = paramsL
	fdisk = 1 - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])
	modv_hlag = m33_sys+halolag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if halosig < disksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	#per-component llikelihoods
	disklpdf = glpdf(vels, modv_dlag, disksig)
	rhalolpdf = glpdf(vels, modv_hlag, halosig)
	blob = np.c_[rhalolpdf,disklpdf]

	all_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(fhalo)+rhalolpdf)
	return np.sum(all_likelihood),blob

def loglike_1disknrothalo4(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo = paramsL
	fdisk = 1 - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if halosig < disksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	#per-component llikelihoods
	halolpdf = glpdf(vels, halocen, halosig)
	disklpdf = glpdf(vels, modv_dlag, disksig)
	blob = np.c_[halolpdf,disklpdf]

	all_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(fhalo)+halolpdf)
	return np.sum(all_likelihood),blob

def loglike_1disknohalo4(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo = paramsL
	fdisk = 1 - fhalo
	modv_dlag = m33_sys+disklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	disklpdf = glpdf(vels, modv_dlag, disksig)
	blob = np.c_[np.zeros(len(disklpdf)),disklpdf]
	return np.sum(disklpdf),blob

def loglike_all4noL(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo = paramsL
	fdisk = 1 - ftdisk - fhalo
	modv_hlag = m33_sys+halolag*extras[2]*np.cos(extras[1])*np.sin(extras[0])
	modv_tdlag = m33_sys+tdisklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if tdisksig < disksig: #or halosig < tdisksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	#per-component llikelihoods
	tdiskpdf = glpdf(vels, modv_tdlag, tdisksig)
	disklpdf = glpdf(vels, modv, disksig)
	rhalolpdf = glpdf(vels, modv_hlag, halosig)

	disks_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(ftdisk)+tdiskpdf)
	blob = np.c_[rhalolpdf,disks_likelihood]
	all_likelihood = np.logaddexp(disks_likelihood, np.log(fhalo)+rhalolpdf)
	return np.sum(all_likelihood),blob

def loglike_nrothalo4noL(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo = paramsL
	fdisk = 1 - ftdisk - fhalo
	modv_tdlag = m33_sys+tdisklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if tdisksig < disksig: #or halosig < tdisksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	#per-component llikelihoods
	halolpdf = glpdf(vels, halocen, halosig)
	tdiskpdf = glpdf(vels, modv_tdlag, tdisksig)
	disklpdf = glpdf(vels, modv, disksig)

	disks_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(ftdisk)+tdiskpdf)
	blob = np.c_[halolpdf,disks_likelihood]
	all_likelihood = np.logaddexp(disks_likelihood, np.log(fhalo)+halolpdf)
	return np.sum(all_likelihood),blob

def loglike_nohalo4noL(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo = paramsL
	fdisk = 1 - ftdisk - fhalo
	modv_tdlag = m33_sys+tdisklag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if tdisksig < disksig:# or halosig < tdisksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	#per-component llikelihoods
	tdiskpdf = glpdf(vels, modv_tdlag, tdisksig)
	disklpdf = glpdf(vels, modv, disksig)

	disks_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(ftdisk)+tdiskpdf)
	blob = np.c_[np.zeros(len(disks_likelihood)),disks_likelihood]
	return np.sum(disks_likelihood),blob

def loglike_1disk4noL(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo = paramsL
	fdisk = 1 - fhalo
	modv_hlag = m33_sys+halolag*extras[2]*np.cos(extras[1])*np.sin(extras[0])

	if halosig < disksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	#per-component llikelihoods
	disklpdf = glpdf(vels, modv, disksig)
	rhalolpdf = glpdf(vels, modv_hlag, halosig)
	blob = np.c_[rhalolpdf,disklpdf]

	all_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(fhalo)+rhalolpdf)
	return np.sum(all_likelihood),blob

def loglike_1disknrothalo4noL(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo = paramsL
	fdisk = 1 - fhalo

	if halosig < disksig:
		return -np.inf,np.ones([len(vels),2])*np.nan

	#per-component llikelihoods
	halolpdf = glpdf(vels, halocen, halosig)
	disklpdf = glpdf(vels, modv, disksig)
	blob = np.c_[halolpdf,disklpdf]

	all_likelihood = np.logaddexp(np.log(fdisk)+disklpdf, np.log(fhalo)+halolpdf)
	return np.sum(all_likelihood),blob

def loglike_1disknohalo4noL(paramsL,vels,modv,extras):
	disklag, disksig, tdisklag, tdisksig, ftdisk, halocen, halosig, halolag, fhalo = paramsL
	fdisk = 1 - fhalo

	disklpdf = glpdf(vels, modv, disksig)
	blob = np.c_[np.zeros(len(disklpdf)),disklpdf]
	return np.sum(disklpdf),blob

#-------------------------------------------
def run_nest(vels, modv, extras, loglike_func, priortrans_func, paramlist, fnam, abridged, backflag=False):
	# if backflag==True:
	# 	sampler = dynesty.DynamicNestedSampler(loglike_func, priortrans_func, len(params3),logl_args=[vels,modv,extras,vfg])
	# else:
	# print(loglike_func, priortrans_func, len(paramlist))
	sampler = dynesty.DynamicNestedSampler(loglike_func, priortrans_func, len(paramlist),logl_args=[vels,modv,extras],blob=True)
	# print([vels,modv,extras])

	sampler.run_nested(print_progress=False)
	result = sampler.results
	result_weights = np.exp(result.logwt - result.logz[-1])  # normalized weights. critical for resampling for corner plot.
	samples_equal = dyfunc.resample_equal(result.samples, result_weights)
	# if abridged!='T':
	f2 = corner.corner(samples_equal, labels=paramlist, quantiles=[.16, .5, .84], show_titles=True) #Spits out junk for the 'unfitted' parametes, which you can ignore.
	plt.close(f2)

	quantiles = [dyfunc.quantile(samps, [0.16, 0.5, 0.84]) for samps in samples_equal.T]

	if not abridged:
		#replaces plot_autocorr and plot_chains to ensure fit has converged: first checks evidence converged, second checks all parameters converged
		fig, axes = dyplot.runplot(result, color='tab:blue', mark_final_live=False, logplot=True)
		fig.tight_layout()
		plt.close(fig)
		fig, axes = dyplot.traceplot(result, show_titles=True, trace_cmap='viridis', quantiles=None)
		fig.tight_layout()
		plt.close(fig)

	return [result.logz[-1], result.logzerr[-1]], np.array(quantiles), samples_equal,result['blob']
	#returns [evidence, error on evidence], [-1\sigma,50%,+1sigma] for each fitted par, sample "run"/chain (CORRECTLY EQUALLY WEIGHTED), and sample 'blobs' (do these need weighting???? hopefully not since recorded like likelihood)

def plot_hist_voffset(samplechain, quantiles, vels, modelvels, extras, vdatalbl, figname, optflags, ndraws=False, y_max=y_lim_factor, abridged=False):
    if not ndraws:
        ndraws = 50
    vel_offset = modelvels - vels
    # make an array of velocity parameters, randomly drawing from the chains
    nthsample = np.random.randint(0, len(samplechain), size=ndraws)
    modelparams = samplechain[nthsample]

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.hist(vel_offset, label=vdatalbl, color=crgb, bins='auto', alpha=alpha_comb, density=True, zorder=2, histtype='stepfilled', edgecolor=crgb)

    xmin, xmax, xsteps = -500, 500, 1.
    xarray = np.arange(xmin, xmax, xsteps)

    for v in modelparams:
        dm, tdm, mdm, hm, dphm, hf = genpdf_voffset(xarray, v, extras, modelvels, optflags)
        ax.plot(xarray, mdm, c=cdiskdraws, lw=0.5, alpha=0.4, label='_nolabel_')
        ax.plot(xarray, hm * hf, c=chalodraws, lw=0.5, alpha=0.4, label='_nolabel_')
        ax.plot(xarray, mdm + hm * hf, c=ctotaldraws, lw=0.5, alpha=0.4, label='_nolabel_')

    diskmodel_pdf, tdiskmodel_pdf, multidiskmodel_pdf, halomodel_pdf, diskplushalo_pdf, halofrac = genpdf_voffset(
        xarray, quantiles[:, 1], extras, modelvels, optflags)
    ax.plot(xarray, multidiskmodel_pdf, lw=3, linestyle=':', c=cdisk, label='Stellar Disk Model')
    ax.plot(xarray, halomodel_pdf * halofrac, lw=3, linestyle='--', c=chalo, label='Stellar Halo Model')
    ax.plot(xarray, diskplushalo_pdf, lw=3, linestyle='-.', c=ctotal, label='Combined Disk+Halo model')

    ymax = max(diskplushalo_pdf) + ylim_pdf_pad * max(diskplushalo_pdf)
    ax.axvline(0, 0, 1, ls='-', c=csys, lw=1, label=r"H$_{\mathrm{I}}$ Velocity")
    ax.legend(loc=legloc)
    ax.set_xlim(-250, 250)

    ax.xaxis.set_minor_locator(plt.MultipleLocator(25))
    ax.set_xlabel(r"Velocity Offset from H$_{\mathrm{I}}$ Disk Model (km s$^{-1}$)")
    ax.set_ylabel(r"Probability Density")
    ax.set_ylabel(histylabel)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return

def plot_hist_vlos(samplechain, quantiles, vels, modelvels, extras, vdatalbl, figname, optflags, ndraws=False, y_max=y_lim_factor, m33centric=False, abridged=False):
    if not ndraws:
        ndraws = 50
    if m33centric:
        vchange = -1.*m33_sys  # want to add this value
    else:
        vchange = 0.

    # make an array of velocity parameters, randomly drawing from the chains
    nthsample = np.random.randint(0, len(samplechain), size=ndraws)
    modelparams = samplechain[nthsample]

    fig, ax = plt.subplots(figsize=(12, 10))
    step = 15
    numax = math.ceil((np.max(vels+vchange)-np.min(vels+vchange))/step)*step+np.min(vels+vchange)+1
    binsofwidth = np.arange(np.min(vels+vchange), numax, step)
    plt.hist(vels+vchange, label=vdatalbl, color=crgb, bins=binsofwidth, alpha=alpha_comb, density=True, zorder=2, histtype='stepfilled', edgecolor=crgb)

    xmin, xmax, xsteps = -500, 500, 1.
    xarray = np.arange(xmin, xmax, xsteps)

    for v in modelparams:
        dm, tdm, mdm, hm, dphm, hf = genpdf_vlos(xarray, v, extras, modelvels, optflags)
        ax.plot(xarray+vchange, mdm, c=cdiskdraws, lw=0.5, alpha=0.4, label='_nolabel_')
        ax.plot(xarray+vchange, hm*hf, c=chalodraws, lw=0.5, alpha=0.4, label='_nolabel_')
        ax.plot(xarray+vchange, mdm + hm*hf, c=ctotaldraws, lw=0.5, alpha=0.4, label='_nolabel_')

    diskmodel_pdf, tdiskmodel_pdf, multidiskmodel_pdf, halomodel_pdf, diskplushalo_pdf, halofrac = genpdf_vlos(
        xarray, quantiles[:, 1], extras, modelvels, optflags)
    ax.plot(xarray+vchange, multidiskmodel_pdf, lw=3, linestyle=':', c=cdisk, label='Stellar Disk Model')
    ax.plot(xarray+vchange, halomodel_pdf*halofrac, lw=3, linestyle='--', c=chalo, label='Stellar Halo Model')
    ax.plot(xarray+vchange, diskplushalo_pdf, lw=3, linestyle='-.', c=ctotal, label='Combined Disk+Halo model')

    ymax = max(diskplushalo_pdf) + ylim_pdf_pad*max(diskplushalo_pdf)
    ax.axvline(m33_sys + vchange, 0, 1, ls='-', c=csys, lw=1, label='M33 Systemic Velocity')
    ax.legend(loc=legloc)
    ax.set_xlim(-370 + vchange, 20 + vchange)
    # ax.set_ylim(top=ymax)

    ax.xaxis.set_minor_locator(plt.MultipleLocator(25))
    ax.set_xlabel('Heliocentric Velocity (km s$^{-1}$)')
    ax.set_ylabel('Probability Density')
    # if abridged=='NP':
    #     ax.axes.yaxis.set_ticklabels([])
    # else:
    ax.set_ylabel(histylabel)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return

def compute_majaxis_model(gasdiskmodel, skycoord,extras, parlist, optflags, ndraws=1000):
	# given an input list of sky coordinate objects, the HI gas disk model data, and the 50th percentile values for stellar model, randomly draw samples of the stellar velocity model (from the sky positions of the stars) for plotting on a distance along major axis vs. heliocentric velocity plot.
	# returns: array of major axis distances of the targets, and arrays for draws from the models: major axis distances, and velocities

	#use the same function as used for disk model
	a,b = major_minor_transform(skycoord, m33_pa, centercoords=m33coord)
	majax_dist = a*60. # degrees to arcmin

	# at each star's sky position
	majax_mod = []
	veldraws_mod = []

	for s,ma,k in zip(skycoord,majax_dist,range(len(skycoord))):
		if len(parlist)<=6:
			if optflags[1]=='T':
				voffset_d = np.random.normal(0, parlist[1],int(ndraws*(1.-parlist[2])))
				if optflags[2]=='T':
					voffset_h = np.random.normal(0.0, parlist[4],int(ndraws*parlist[2]))
					veldraws_halo = m33_sys + extras[2][k]*parlist[5]*np.cos(extras[1][k])*np.sin(extras[0][k]) + voffset_h
				else:
					veldraws_halo = np.random.normal(parlist[3], parlist[4],int(ndraws*parlist[2]))
			else:
				voffset_d = np.random.normal(0, parlist[1], ndraws)

			if optflags[0]=='T':
				veldraws_disk = m33_sys + extras[2][k]*parlist[0]*np.cos(extras[1][k])*np.sin(extras[0][k]) + voffset_d
			else:
				veldraws_disk = m33_sys + extras[2][k]*np.cos(extras[1][k])*np.sin(extras[0][k]) + voffset_d

			if optflags[1]=='T':
				tmp_v = np.concatenate((veldraws_disk,veldraws_halo), axis=None)
				veldraws_mod.append(tmp_v)
				total_draws = int(ndraws*parlist[2]) + int(ndraws*(1 - parlist[2]))
				majax_mod.append([ma]*total_draws)
			else:
				veldraws_mod.append(veldraws_disk)
				majax_mod.append([ma]*ndraws)

		elif len(parlist)==8:
			voffset_d2 = np.random.normal(0, parlist[3], int(ndraws*parlist[4]))
			if optflags[1]=='T':
				voffset_d1 = np.random.normal(0, parlist[1], int(ndraws*(1.-parlist[4]-parlist[7])))
				veldraws_halo = np.random.normal(parlist[5], parlist[6],int(ndraws*parlist[7]))
			else:
				voffset_d1 = np.random.normal(0, parlist[1], int(ndraws*(1.-parlist[4])))

			veldraws_disk1 = m33_sys + extras[2][k]*parlist[0]*np.cos(extras[1][k])*np.sin(extras[0][k]) + voffset_d1
			veldraws_disk2 = m33_sys + extras[2][k]*parlist[2]*np.cos(extras[1][k])*np.sin(extras[0][k]) + voffset_d2

			if optflags[1]=='T':
				tmp_v = np.concatenate((veldraws_disk1,veldraws_disk2,veldraws_halo), axis=None)
				veldraws_mod.append(tmp_v)
				total_draws = int(ndraws*parlist[4]) + int(ndraws*parlist[7]) + int(ndraws*(1 - parlist[4]-parlist[7]))
				majax_mod.append([ma]*total_draws)
			else:
				tmp_v = np.concatenate((veldraws_disk1,veldraws_disk2), axis=None)
				veldraws_mod.append(tmp_v)
				total_draws = int(ndraws*parlist[4]) + int(ndraws*(1 - parlist[4]))
				majax_mod.append([ma]*total_draws)
		elif len(parlist)==9:
			if optflags[3]=='T':
				voffset_d2 = np.random.normal(0, parlist[3], int(ndraws*parlist[4]))
				veldraws_disk2 = m33_sys + extras[2][k]*parlist[2]*np.cos(extras[1][k])*np.sin(extras[0][k]) + voffset_d2
				if optflags[1]=='T':
					voffset_d1 = np.random.normal(0, parlist[1], int(ndraws*(1.-parlist[4]-parlist[8])))
					if optflags[2]=='T':
						voffset_h = np.random.normal(0.0, parlist[6],int(ndraws*parlist[8]))
						veldraws_halo = m33_sys + extras[2][k]*parlist[7]*np.cos(extras[1][k])*np.sin(extras[0][k]) + voffset_h
					else:
						veldraws_halo = np.random.normal(parlist[5], parlist[6],int(ndraws*parlist[8]))
				else:
					voffset_d1 = np.random.normal(0, parlist[1], int(ndraws*(1.-parlist[4])))
			else:
				if optflags[1]=='T':
					voffset_d1 = np.random.normal(0, parlist[1], int(ndraws*(1.-parlist[8])))
					if optflags[2]=='T':
						voffset_h = np.random.normal(0.0, parlist[6],int(ndraws*parlist[8]))
						veldraws_halo = m33_sys + extras[2][k]*parlist[7]*np.cos(extras[1][k])*np.sin(extras[0][k]) + voffset_h
					else:
						veldraws_halo = np.random.normal(parlist[5], parlist[6],int(ndraws*parlist[8]))
				else:
					voffset_d1 = np.random.normal(0, parlist[1], ndraws)

			if optflags[0]=='T':
				veldraws_disk1 = m33_sys + extras[2][k]*parlist[0]*np.cos(extras[1][k])*np.sin(extras[0][k]) + voffset_d1
			else:
				veldraws_disk1 = m33_sys + extras[2][k]*np.cos(extras[1][k])*np.sin(extras[0][k]) + voffset_d1

			if optflags[3]=='T':
				if optflags[1]=='T':
					tmp_v = np.concatenate((veldraws_disk1,veldraws_disk2,veldraws_halo), axis=None)
					veldraws_mod.append(tmp_v)
					total_draws = int(ndraws*parlist[8]) + int(ndraws*parlist[7]) + int(ndraws*(1 - parlist[4]-parlist[8]))
					majax_mod.append([ma]*total_draws)
				else:
					tmp_v = np.concatenate((veldraws_disk1,veldraws_disk2), axis=None)
					veldraws_mod.append(tmp_v)
					total_draws = int(ndraws*parlist[4]) + int(ndraws*(1 - parlist[4]))
					majax_mod.append([ma]*total_draws)
			else:
				if optflags[1]=='T':
					tmp_v = np.concatenate((veldraws_disk1,veldraws_halo), axis=None)
					veldraws_mod.append(tmp_v)
					total_draws = int(ndraws*parlist[8]) + int(ndraws*(1 -parlist[8]))
					majax_mod.append([ma]*total_draws)
				else:
					veldraws_mod.append(veldraws_disk1)
					majax_mod.append([ma]*ndraws)

	majax_model = np.ndarray.flatten(np.array(majax_mod))
	veldraws_model = np.ndarray.flatten(np.array(veldraws_mod))
	return majax_dist, majax_model, veldraws_model

def plot_majaxis_model(majaxis_dist, majaxis_model, veldraws_model, obs_vel, probs, figname, stellarlabel='Observed Stellar Velocity', cbarlabel='Disk Probability'):
    gap = 1  # add a little space to the figure so model and points not up against x axis limits
    xedges = [majaxis_dist.min(), majaxis_dist.max()]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot([xedges[0] - gap, xedges[-1] + gap], [m33_sys, m33_sys], color=csys, alpha=0.5, zorder=1, linestyle='-', linewidth=1, label='M33 Systemic Velocity')
    dc = ax.scatter(majaxis_dist, obs_vel, c=probs, s=15, alpha=0.75, zorder=2, label=stellarlabel)

    ax.set_xlim(xedges[0] - gap, xedges[-1] + gap)
    if len(probs) > 1:
        cbar = plt.colorbar(dc)
        cbar.set_label(cbarlabel)
        cbar.set_alpha(1)
    ax.legend(loc=legloc)
    ax.set_xlabel('Distance Along Major Axis (arcmin)')
    ax.set_ylabel('Heliocentric Velocity (km s$^{-1}$)')
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return

def plot_cmd_prob(tbdata, probs, figname):
    probinds = probs.argsort()
    probsort = probs[probinds[::-1]]  # order from max disk prob to min (halo on top)
    tbsort = tbdata[probinds[::-1]]

    m_cfht = (~np.isnan(tbsort['G0_CFHT']))
    m_475 = (~np.isnan(tbsort['F475W0_ACS']))
    m_606 = (~np.isnan(tbsort['F606W0_ACS']))

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].scatter(tbsort['G0_CFHT'][m_cfht] - tbsort['I0_CFHT'][m_cfht], tbsort['I0_CFHT'][m_cfht], c=probsort[m_cfht], alpha=0.8, vmin=0, vmax=1)
    ax[0].set_xlabel(r'(g - i)$_0$')
    ax[0].set_ylabel('i$_0$')

    dc = ax[1].scatter(tbsort['F475W0_ACS'][m_475] - tbsort['F814W0_ACS'][m_475], tbsort['F814W0_ACS'][m_475], c=probsort[m_475], alpha=0.8, vmin=0, vmax=1)
    ax[1].set_xlabel(r'(F475W - F814W)$_0$')
    ax[1].set_ylabel(r'F814W$_0$')

    ax[2].scatter(tbsort['F606W0_ACS'][m_606] - tbsort['F814W0_ACS'][m_606], tbsort['F814W0_ACS'][m_606], c=probsort[m_606], alpha=0.8, vmin=0, vmax=1)
    ax[2].set_xlabel(r'(F606W - F814W)$_0$')
    ax[2].set_ylabel(r'F814W$_0$')

    for a in ax:
        a.invert_yaxis()

    if len(probs) > 1:
        cbar = plt.colorbar(dc)
        cbar.set_label('Disk Probability')
        cbar.set_alpha(1)
    fig.tight_layout()
    if hasattr(argmts, 'savecmd') and argmts.savecmd == 'T':
        fn = figname[:-4]
        pickle.dump(fig, open(f'{spec_direct}/dyplots/{figname}.pickle', 'wb'))
    else:
        fig.savefig(f'{spec_direct}/dyplots/{figname}', bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return

def plot_pos_prob(tbdata, probs, figname):
    probinds = probs.argsort()
    probsort = probs[probinds[::-1]]  # order from max disk prob to min (halo on top)
    tbsort = tbdata[probinds[::-1]]

    fig, ax = plt.subplots(1, figsize=(8, 8))
    dc = ax.scatter(tbsort['coord'].ra.deg, tbsort['coord'].dec.deg, c=probsort, alpha=0.8)
    if len(probs) > 1:
        cbar = plt.colorbar(dc)
        cbar.set_label('Disk Probability')
        cbar.set_alpha(1)
    ax.set_xlabel('RA (deg)')
    ax.invert_xaxis()
    ax.set_ylabel('Dec (deg)')
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return

def calcprobs_slowrotator(parlist,dat,trres,extras,optflags):
	# compute the probability an individual star is halo or disk using values of model parameters passed as parlist. dat = velocities.
	# output: the log likelihood for the set of parameters parlist

	if len(parlist)<=6:
		if optflags[0]=='T':
			vdisk = m33_sys + parlist[0]*extras[2]*np.cos(extras[1])*np.sin(extras[0])
		else:
			vdisk=trres

		if optflags[1]=='T':
			if optflags[2]=='T':
				vhalo = m33_sys + parlist[5]*extras[2]*np.cos(extras[1])*np.sin(extras[0])
			else:
				vhalo = parlist[3]

			lngauss_m33halo = np.log(parlist[2]) - np.log(parlist[4]) - np.log(np.sqrt(2.*np.pi)) - (np.array(dat)-np.array(vhalo))**2./(2.0*parlist[4]**2.) # halo likelihood
			lngauss_m33disk = np.log(1. - parlist[2]) - np.log(parlist[1]) - np.log(np.sqrt(2.*np.pi)) -(np.array(dat)-np.array(vdisk))**2./(2.0*parlist[1]**2.) # disk likelihood
			lngauss_m33term = scipy.special.logsumexp([lngauss_m33halo, lngauss_m33disk], axis = 0)

			post_prob_d = np.exp(lngauss_m33disk - lngauss_m33term)
			post_prob_h = np.exp(lngauss_m33halo - lngauss_m33term)
		else:
			post_prob_d = np.ones(len(vdisk))
			post_prob_h = np.zeros(len(vdisk))

	elif len(parlist)==8:
		vdisk = m33_sys + parlist[2]*extras[2]*np.cos(extras[1])*np.sin(extras[0])
		vdisk2 = m33_sys + parlist[0]*extras[2]*np.cos(extras[1])*np.sin(extras[0])

		if optflags[1]=='T':
			vhalo = parlist[5]

			lngauss_m33halo = np.log(parlist[7]) - np.log(parlist[6]) - np.log(np.sqrt(2.*np.pi)) - (np.array(dat)-np.array(vhalo))**2./(2.0*parlist[4]**2.) # halo likelihood
			lngauss_m33disk1 = np.log(1-parlist[4]-parlist[7]) - np.log(parlist[1]) - np.log(np.sqrt(2.*np.pi)) -(np.array(dat)-np.array(vdisk))**2./(2.0*parlist[1]**2.) # disk likelihood
			lngauss_m33disk2 = np.log(parlist[4]) - np.log(parlist[3]) - np.log(np.sqrt(2.*np.pi)) -(np.array(dat)-np.array(vdisk2))**2./(2.0*parlist[3]**2.) # disk likelihood

			lngauss_m33term = scipy.special.logsumexp([lngauss_m33halo, lngauss_m33disk1,lngauss_m33disk2], axis = 0)
			post_prob_d = np.exp(lngauss_m33disk1+lngauss_m33disk2 - lngauss_m33term)
			post_prob_h = np.exp(lngauss_m33halo - lngauss_m33term)
		else:
			lngauss_m33disk1 = np.log(1-parlist[4]) - np.log(parlist[1]) - np.log(np.sqrt(2.*np.pi)) -(np.array(dat)-np.array(vdisk))**2./(2.0*parlist[1]**2.) # disk likelihood
			lngauss_m33disk2 = np.log(parlist[4]) - np.log(parlist[3]) - np.log(np.sqrt(2.*np.pi)) -(np.array(dat)-np.array(vdisk2))**2./(2.0*parlist[3]**2.) # disk likelihood
			lngauss_m33term = scipy.special.logsumexp([lngauss_m33disk1, lngauss_m33disk2], axis = 0)

			post_prob_d = np.exp(lngauss_m33disk1+lngauss_m33disk2 - lngauss_m33term)
			post_prob_h = np.zeros(len(vdisk))

	elif len(parlist)==9:
		if optflags[0]=='T':
			vdisk1 = m33_sys + parlist[0]*extras[2]*np.cos(extras[1])*np.sin(extras[0])
		else:
			vdisk1=trres

		if optflags[3]=='T':
			vdisk2 = m33_sys + parlist[2]*extras[2]*np.cos(extras[1])*np.sin(extras[0])
			lngauss_m33disk2 = np.log(parlist[4]) - np.log(parlist[3]) - np.log(np.sqrt(2.*np.pi)) -(np.array(dat)-np.array(vdisk2))**2./(2.0*parlist[3]**2.) # disk likelihood

		if optflags[1]=='T':
			if optflags[2]=='T':
				vhalo = m33_sys + parlist[7]*extras[2]*np.cos(extras[1])*np.sin(extras[0])
			else:
				vhalo = parlist[5]
			lngauss_m33halo = np.log(parlist[8]) - np.log(parlist[6]) - np.log(np.sqrt(2.*np.pi)) - (np.array(dat)-np.array(vhalo))**2./(2.0*parlist[6]**2.) # halo likelihood

			if optflags[3]=='T':
				lngauss_m33disk1 = np.log(1.-parlist[8]-parlist[4]) - np.log(parlist[1]) - np.log(np.sqrt(2.*np.pi)) -(np.array(dat)-np.array(vdisk1))**2./(2.0*parlist[1]**2.) # disk likelihood
				lngauss_m33term = scipy.special.logsumexp([lngauss_m33halo, lngauss_m33disk1,lngauss_m33disk2], axis = 0)
				post_prob_d = np.exp(lngauss_m33disk1+lngauss_m33disk2 - lngauss_m33term)
				post_prob_h = np.exp(lngauss_m33halo - lngauss_m33term)
			else:
				lngauss_m33disk1 = np.log(1.-parlist[8]) - np.log(parlist[1]) - np.log(np.sqrt(2.*np.pi)) -(np.array(dat)-np.array(vdisk1))**2./(2.0*parlist[1]**2.) # disk likelihood
				lngauss_m33term = scipy.special.logsumexp([lngauss_m33halo, lngauss_m33disk1], axis = 0)
				post_prob_d = np.exp(lngauss_m33disk1 - lngauss_m33term)
				post_prob_h = np.exp(lngauss_m33halo - lngauss_m33term)
		else:
			if optflags[3]=='T':
				lngauss_m33disk1 = np.log(1.-parlist[4]) - np.log(parlist[1]) - np.log(np.sqrt(2.*np.pi)) -(np.array(dat)-np.array(vdisk1))**2./(2.0*parlist[1]**2.) # disk likelihood
				lngauss_m33term = scipy.special.logsumexp([lngauss_m33disk1, lngauss_m33disk2], axis = 0)
				post_prob_d = np.exp(lngauss_m33disk1+lngauss_m33disk2 - lngauss_m33term)
				post_prob_h = np.zeros(len(vdisk1))
			else:
				post_prob_d = np.ones(len(vdisk1))
				post_prob_h = np.zeros(len(vdisk1))

	return post_prob_d, post_prob_h

set_dict = {1:"LD_SH",2:"NLD_SH",3:"LD_LH",4:"2LD_SH",5:"2LD_LH"}
def fit_model(
    data, diskr, dskycoords, model_vel, key, calcset, optflags,
    minr=0, maxr=100, model=diskmodel, burncut=False, bestevflag=False, abridged=False,
    startype=None
):
    fnam = key + "_" + set_dict[calcset] + "_" + optflags
    if calcset == 1:
        paramlist = params
        priortrans_func = priortrans
        loglike_func = loglike_Ldiskhalo if optflags[1] == 'T' else loglike_Ldisknohalo
    elif calcset == 2:
        paramlist = params
        priortrans_func = priortrans
        if optflags[0] == 'T':
            loglike_func = loglike_Ldiskhalo if optflags[1] == 'T' else loglike_Ldisknohalo
        else:
            loglike_func = loglike_diskhalo if optflags[1] == 'T' else loglike_disknohalo
    elif calcset == 3:
        paramlist = params2
        priortrans_func = priortrans2
        if optflags[1] == 'T':
            loglike_func = loglike_rothalo if optflags[2] == 'T' else loglike_nrothalo
        else:
            loglike_func = loglike_nohalo
    elif calcset == 4:
        paramlist = params3
        priortrans_func = priortrans3
        loglike_func = loglike_all3 if optflags[1] == 'T' else loglike_nohalo3
    elif calcset == 5:
        paramlist = params4
        priortrans_func = priortrans4
        if optflags[0] == 'T':
            if optflags[3] == 'T':
                if optflags[1] == 'T':
                    loglike_func = loglike_all4 if optflags[2] == 'T' else loglike_nrothalo4
                else:
                    loglike_func = loglike_nohalo4
            else:
                if optflags[1] == 'T':
                    loglike_func = loglike_1disk4 if optflags[2] == 'T' else loglike_1disknrothalo4
                else:
                    loglike_func = loglike_1disknohalo4
        else:
            if optflags[3] == 'T':
                if optflags[1] == 'T':
                    loglike_func = loglike_all4noL if optflags[2] == 'T' else loglike_nrothalo4noL
                else:
                    loglike_func = loglike_nohalo4noL
            else:
                if optflags[1] == 'T':
                    loglike_func = loglike_1disk4noL if optflags[2] == 'T' else loglike_1disknrothalo4noL
                else:
                    loglike_func = loglike_1disknohalo4noL

    if calcset in [1, 2, 3]:
        numind = [1, 0, 2, 4, 5]
    elif calcset == 4:
        numind = [[1, 3], [0, 2], 7, 6, np.nan]
    elif calcset == 5:
        numind = [[1, 3], [0, 2], 8, 6, 7]

    maskrad = [(diskr * 60. >= minr) & (diskr * 60. < maxr)][0]
    if 'N' in key:
        maskhalf = [(model_vel < m33_sys)][0]
    elif 'S' in key:
        maskhalf = [(model_vel >= m33_sys)][0]
    else:
        maskhalf = np.full(len(model_vel), True)
    filt = np.logical_and.reduce((maskrad, maskhalf))
    filt_data = data[filt]
    filt_skycoords = dskycoords[filt]
    filt_model_vel = model_vel[filt]

    radius, Rinit, incliter, paiter = m33_tilted_ring_deproj_radius(filt_skycoords)
    theta = m33_tilted_ring_deproj_angle(filt_skycoords, incl=incliter, pa=paiter)
    f_vrot = interp1d(model['Radius_arcmin'], model['Vrot_kms'])
    f_vrot_of_r = (np.array([f_vrot(r * 60.) for r in radius]).reshape(-1))
    extras = [incliter.value * np.pi / 180, theta.value * np.pi / 180, f_vrot_of_r]

    evidence, quantiles, samples, blob = run_nest(
        filt_data['VCORR_STAT'].value, filt_model_vel, extras,
        loglike_func, priortrans_func, paramlist, fnam, abridged
    )

    if abridged != 'T':
        if not burncut:
            chain_topass = samples[int(len(samples[:, 0]) / 5):, :]
            blob_topass = blob[int(len(samples[:, 0]) / 5):, :, :]
        else:
            chain_topass = samples[int(len(samples[:, 0]) / burncut):, :]
            blob_topass = blob[int(len(samples[:, 0]) / burncut):, :, :]

        # Use startype in label (not argmts)
        label = f'Observed {startype} Star Velocities' if startype else 'Observed Star Velocities'

        plot_hist_voffset(
            chain_topass, quantiles, filt_data['VCORR_STAT'].value, filt_model_vel, extras,
            vdatalbl=label, figname=f'{fnam}_voffset.pdf',
            optflags=optflags, ndraws=50, abridged=abridged
        )
        plot_hist_vlos(
            chain_topass, quantiles, filt_data['VCORR_STAT'].value, filt_model_vel, extras,
            vdatalbl=label, figname=f'{fnam}_vlos.pdf',
            optflags=optflags, ndraws=50, abridged=abridged
        )

        pd, ph = calcprobs_slowrotator(quantiles[:, 1], filt_data['VCORR_STAT'].value, filt_model_vel, extras, optflags)
        majax_dist_targs, ma_mod, v_mod = compute_majaxis_model(
            diskmodel, filt_skycoords, extras, quantiles[:, 1], optflags, ndraws=1
        )
        plot_majaxis_model(
            majax_dist_targs, ma_mod, v_mod, filt_data['VCORR_STAT'].value, probs=pd,
            figname=f'{fnam}_pdiskpost.pdf',
            stellarlabel='Observed AGB Velocity', cbarlabel='Disk Probability'
        )

        prob_disk = get_blob_probs(blob_topass)
        plot_majaxis_model(
            majax_dist_targs, ma_mod, v_mod, filt_data['VCORR_STAT'].value, probs=prob_disk[:, 1],
            figname=f'{fnam}_pdiskblob.pdf', stellarlabel='Observed AGB Velocity', cbarlabel='Disk Probability'
        )
        plot_majaxis_model(
            majax_dist_targs, ma_mod, v_mod, filt_data['VCORR_STAT'].value, probs=abs(prob_disk[:, 1] - pd),
            figname=f'{fnam}_deltapdisk.pdf', stellarlabel='Observed AGB Velocity', cbarlabel='Delta Disk Probability'
        )

    # (Rest unchanged)
    if bestevflag:
        # save probs to data array
        if key == kv:
            dprob_50th_all = Column(pd, name='DPROB_50TH_ALL')
        else:
            dprob_50th_radial[filt] = pd

        # assign dictionary values
        if optflags[3] == 'T':
            if optflags[0] == 'T':
                avfrotd = np.average([take_deltas(quantiles[0, :]), take_deltas(quantiles[2, :])], axis=0)
                frotdisk[key] = revert_deltas(avfrotd)
            else:
                deltatd = take_deltas(quantiles[2, :])
                avmed = np.average([deltatd[1], 1.0])
                frotdisk[key] = [avmed - deltatd[0], avmed, avmed + deltatd[2]]
            avsigd = np.average([take_deltas(quantiles[1, :]), take_deltas(quantiles[3, :])], axis=0)
            sigdisk[key] = revert_deltas(avsigd)
        else:
            sigdisk[key] = quantiles[1, :]
            if optflags[0] == 'T':
                frotdisk[key] = quantiles[0, :]
            else:
                frotdisk[key] = [0.0, 0.0, 0.0]
        if optflags[1] == 'T':
            if optflags[2] == 'T':
                frothalo[key] = quantiles[numind[4], :]
            else:
                frothalo[key] = [0.0, 0.0, 0.0]
            sighalo[key] = quantiles[numind[3], :]
            fhalo[key] = quantiles[numind[2], :]
        else:
            fhalo[key] = [0.0, 0.0, 0.0]
            sighalo[key] = [0.0, 0.0, 0.0]

        sigdisk[key], fhalo[key], sighalo[key], frothalo[key], frotdisk[key] = sigdisk_50th, fhalo_50th, sighalo_50th, frot_50th, frotdisk_50th

    return evidence, quantiles, len(filt_data), fnam, pd

# fit_model(np.ones(20), np.ones(20), np.ones(20), np.ones(20), 'allAGB', 1, 'TFFF')

def take_deltas(qrow):
	delta_under = qrow[1]-qrow[0]
	delta_over = qrow[2]-qrow[1]
	return [delta_under,qrow[1],delta_over]
def revert_deltas(avrow):
	return [avrow[1]-avrow[0],avrow[1],avrow[1]+avrow[2]]

def genpdf_voffset(xarray,parlist,extras,modelvels,optflags):
	if len(parlist)==5:
		if optflags[0]=='T':
			veldiskmodel_td = m33_sys + extras[2]*parlist[0]*np.cos(extras[1])*np.sin(extras[0])
			vel_disk = np.zeros(len(xarray))
			for v,td in zip(modelvels, veldiskmodel_td):
				vel_disk = vel_disk + norm.pdf(xarray, v-td, parlist[1])
			diskmodel_pdf = vel_disk/len(modelvels)
		else:
			diskmodel_pdf = norm.pdf(xarray, 0., parlist[1])
		multidiskmodel_pdf = (1-parlist[2])*diskmodel_pdf
		tdiskmodel_pdf = np.zeros(len(xarray))

		if optflags[1]=='T': #halomodel_pdf = norm.pdf(xarray, 0., parlist[4])
			vel_halo = np.zeros(len(xarray))
			for v in modelvels:
				veloffset_halo = v - parlist[3] # velocity offset at a star's location is velocity of HI model - mean halo
				vel_halo = vel_halo + norm.pdf(xarray, veloffset_halo, parlist[4])
			halomodel_pdf = vel_halo/len(modelvels)
		else:
			halomodel_pdf = np.zeros(len(xarray))

		diskplushalo_pdf = (1-parlist[2])*diskmodel_pdf + parlist[2]*halomodel_pdf
		halofrac = parlist[2]

	elif len(parlist)==6:
		veldiskmodel = m33_sys + extras[2]*parlist[0]*np.cos(extras[1])*np.sin(extras[0])
		vel_disk = np.zeros(len(xarray))
		for v,td in zip(modelvels, veldiskmodel):
			vel_disk = vel_disk + norm.pdf(xarray, v-td, parlist[1])
		diskmodel_pdf = vel_disk/len(modelvels)
		multidiskmodel_pdf = (1-parlist[2])*diskmodel_pdf
		tdiskmodel_pdf = np.zeros(len(xarray))

		if optflags[1]=='T':
			if optflags[2]=='T':
				velhalomodel = m33_sys + extras[2]*parlist[5]*np.cos(extras[1])*np.sin(extras[0])
				vel_halo = np.zeros(len(xarray))
				for v,td in zip(modelvels, velhalomodel):
					vel_halo = vel_halo + norm.pdf(xarray, v-td, parlist[4])
				halomodel_pdf = vel_halo/len(modelvels)
			else: #halomodel_pdf = norm.pdf(xarray, 0., parlist[4])
				vel_halo = np.zeros(len(xarray))
				for v in modelvels:
					veloffset_halo = v - parlist[3] # velocity offset at a star's location is velocity of HI model - mean halo
					vel_halo = vel_halo + norm.pdf(xarray, veloffset_halo, parlist[4])
				halomodel_pdf = vel_halo/len(modelvels)
		else:
			halomodel_pdf = np.zeros(len(xarray))

		diskplushalo_pdf = (1-parlist[2])*diskmodel_pdf + parlist[2]*halomodel_pdf
		halofrac = parlist[2]

	elif len(parlist)==8:
		veldiskmodel = m33_sys + extras[2]*parlist[0]*np.cos(extras[1])*np.sin(extras[0])
		vel_disk = np.zeros(len(xarray))
		for v,td in zip(modelvels, veldiskmodel):
			vel_disk = vel_disk + norm.pdf(xarray, v-td, parlist[1])
		diskmodel_pdf = vel_disk/len(modelvels)

		veldiskmodel_td = m33_sys + extras[2]*parlist[2]*np.cos(extras[1])*np.sin(extras[0])
		vel_tdisk = np.zeros(len(xarray))
		for v,td in zip(modelvels, veldiskmodel_td):
			vel_tdisk = vel_tdisk + norm.pdf(xarray, v-td, parlist[3])
		tdiskmodel_pdf = vel_tdisk/len(modelvels)

		if optflags[1]=='T': #halomodel_pdf = norm.pdf(xarray, 0., parlist[6])
			vel_halo = np.zeros(len(xarray))
			for v in modelvels:
				veloffset_halo = v - parlist[5] # velocity offset at a star's location is velocity of HI model - mean halo
				vel_halo = vel_halo + norm.pdf(xarray, veloffset_halo, parlist[6])
			halomodel_pdf = vel_halo/len(modelvels)
		else:
			halomodel_pdf = np.zeros(len(xarray))

		multidiskmodel_pdf = (1-parlist[4]-parlist[7])*diskmodel_pdf + parlist[4]*tdiskmodel_pdf
		diskplushalo_pdf = multidiskmodel_pdf + parlist[7]*halomodel_pdf
		halofrac = parlist[7]

	elif len(parlist)==9:
		if optflags[0]=='T':
			veldiskmodel = m33_sys + extras[2]*parlist[0]*np.cos(extras[1])*np.sin(extras[0])
			vel_disk = np.zeros(len(xarray))
			for v,td in zip(modelvels, veldiskmodel):
				vel_disk = vel_disk + norm.pdf(xarray, v-td, parlist[1])
			diskmodel_pdf = vel_disk/len(modelvels)
		else:
			diskmodel_pdf = norm.pdf(xarray, 0., parlist[1])

		if optflags[3]=='T':
			veldiskmodel_td = m33_sys + extras[2]*parlist[2]*np.cos(extras[1])*np.sin(extras[0])
			vel_tdisk = np.zeros(len(xarray))
			for v,td in zip(modelvels, veldiskmodel_td):
				vel_tdisk = vel_tdisk + norm.pdf(xarray, v-td, parlist[3])
			tdiskmodel_pdf = vel_tdisk/len(modelvels)
			multidiskmodel_pdf = (1-parlist[4]-parlist[8])*diskmodel_pdf + parlist[4]*tdiskmodel_pdf
		else:
			tdiskmodel_pdf = np.zeros(len(xarray))
			multidiskmodel_pdf = (1-parlist[8])*diskmodel_pdf

		if optflags[1]=='T':
			if optflags[2]=='T':
				velhalomodel = m33_sys + extras[2]*parlist[7]*np.cos(extras[1])*np.sin(extras[0])
				vel_halo = np.zeros(len(xarray))
				for v,td in zip(modelvels, velhalomodel):
					vel_halo = vel_halo + norm.pdf(xarray, v-td, parlist[6])
				halomodel_pdf = vel_halo/len(modelvels)
			else: #halomodel_pdf = norm.pdf(xarray, 0., parlist[6])
				vel_halo = np.zeros(len(xarray))
				for v in modelvels:
					veloffset_halo = v - parlist[5] # velocity offset at a star's location is velocity of HI model - mean halo
					vel_halo = vel_halo + norm.pdf(xarray, veloffset_halo, parlist[6])
				halomodel_pdf = vel_halo/len(modelvels)
		else:
			halomodel_pdf = np.zeros(len(xarray))

		diskplushalo_pdf = multidiskmodel_pdf + parlist[8]*halomodel_pdf
		halofrac = parlist[8]

	return diskmodel_pdf, tdiskmodel_pdf, multidiskmodel_pdf, halomodel_pdf, diskplushalo_pdf, halofrac

def genpdf_vlos(xarray,parlist,extras,modelvels,optflags):
	if len(parlist)==5:
		if optflags[0]=='T':
			veldiskmodel = m33_sys + extras[2]*parlist[0]*np.cos(extras[1])*np.sin(extras[0])
			vel_disk = np.zeros(len(xarray))

			for v in veldiskmodel:
				vel_disk = vel_disk + norm.pdf(xarray, v, parlist[1]) #since LOSframe, no need to subtract modelvel here
			diskmodel_pdf = vel_disk/len(veldiskmodel)
		else:
			vel_disk = np.zeros(len(xarray))
			for v in modelvels:
				vel_disk = vel_disk + norm.pdf(xarray, v, parlist[1])
			diskmodel_pdf = vel_disk/len(modelvels)
		multidiskmodel_pdf = (1-parlist[2])*diskmodel_pdf
		tdiskmodel_pdf = np.zeros(len(xarray))

		if optflags[1]=='T':
			halomodel_pdf = norm.pdf(xarray, parlist[3], parlist[4])
		else:
			halomodel_pdf = np.zeros(len(xarray))

		diskplushalo_pdf = (1-parlist[2])*diskmodel_pdf + parlist[2]*halomodel_pdf
		halofrac = parlist[2]

	elif len(parlist)==6:
		veldiskmodel = m33_sys + extras[2]*parlist[0]*np.cos(extras[1])*np.sin(extras[0])
		vel_disk = np.zeros(len(xarray))

		for v in veldiskmodel:
			vel_disk = vel_disk + norm.pdf(xarray, v, parlist[1]) #since LOSframe, no need to subtract modelvel here
		diskmodel_pdf = vel_disk/len(veldiskmodel)
		multidiskmodel_pdf = (1-parlist[2])*diskmodel_pdf
		tdiskmodel_pdf = np.zeros(len(xarray))

		if optflags[1]=='T':
			if optflags[2]=='T':
				velhalomodel = m33_sys + extras[2]*parlist[5]*np.cos(extras[1])*np.sin(extras[0])
				vel_halo = np.zeros(len(xarray))
				for v in velhalomodel:
					vel_halo = vel_halo + norm.pdf(xarray, v, parlist[4])
				halomodel_pdf = vel_halo/len(velhalomodel)
			else:
				halomodel_pdf = norm.pdf(xarray, parlist[3], parlist[4])
		else:
			halomodel_pdf = np.zeros(len(xarray))

		diskplushalo_pdf = (1-parlist[2])*diskmodel_pdf + parlist[2]*halomodel_pdf
		halofrac = parlist[2]

	elif len(parlist)==8:
		veldiskmodel = m33_sys + extras[2]*parlist[0]*np.cos(extras[1])*np.sin(extras[0])
		vel_disk = np.zeros(len(xarray))
		for v in veldiskmodel:
			vel_disk = vel_disk + norm.pdf(xarray, v, parlist[1]) #since LOSframe, no need to subtract modelvel here
		diskmodel_pdf = vel_disk/len(veldiskmodel)

		veldiskmodel_td = m33_sys + extras[2]*parlist[2]*np.cos(extras[1])*np.sin(extras[0])
		vel_tdisk = np.zeros(len(xarray))
		for v in veldiskmodel_td:
			vel_tdisk = vel_tdisk + norm.pdf(xarray, v, parlist[3]) #since LOSframe, no need to subtract modelvel here
		tdiskmodel_pdf = vel_tdisk/len(veldiskmodel_td)

		if optflags[1]=='T':
			halomodel_pdf = norm.pdf(xarray, parlist[5], parlist[6])
		else:
			halomodel_pdf = np.zeros(len(xarray))

		multidiskmodel_pdf = (1-parlist[4]-parlist[7])*diskmodel_pdf + parlist[4]*tdiskmodel_pdf
		diskplushalo_pdf = multidiskmodel_pdf + parlist[7]*halomodel_pdf
		halofrac = parlist[7]

	elif len(parlist)==9:
		if optflags[0]=='T':
			veldiskmodel = m33_sys + extras[2]*parlist[0]*np.cos(extras[1])*np.sin(extras[0])
			vel_disk = np.zeros(len(xarray))
			for v in veldiskmodel:
				vel_disk = vel_disk + norm.pdf(xarray, v, parlist[1]) #since LOSframe, no need to subtract modelvel here
			diskmodel_pdf = vel_disk/len(veldiskmodel)
		else:
			diskmodel_pdf = norm.pdf(xarray, 0., parlist[1])

		if optflags[3]=='T':
			veldiskmodel_td = m33_sys + extras[2]*parlist[2]*np.cos(extras[1])*np.sin(extras[0])
			vel_tdisk = np.zeros(len(xarray))
			for v in veldiskmodel_td:
				vel_tdisk = vel_tdisk + norm.pdf(xarray, v, parlist[3]) #since LOSframe, no need to subtract modelvel here
			tdiskmodel_pdf = vel_tdisk/len(veldiskmodel_td)
			multidiskmodel_pdf = (1-parlist[4]-parlist[8])*diskmodel_pdf + parlist[4]*tdiskmodel_pdf
		else:
			tdiskmodel_pdf = np.zeros(len(xarray))
			multidiskmodel_pdf = (1-parlist[8])*diskmodel_pdf

		if optflags[1]=='T':
			if optflags[2]=='T':
				velhalomodel = m33_sys + extras[2]*parlist[7]*np.cos(extras[1])*np.sin(extras[0])
				vel_halo = np.zeros(len(xarray))
				for v in velhalomodel:
					vel_halo = vel_halo + norm.pdf(xarray, v, parlist[6])
				halomodel_pdf = vel_halo/len(velhalomodel)
			else:
				halomodel_pdf = norm.pdf(xarray, parlist[5], parlist[6])
		else:
			halomodel_pdf = np.zeros(len(xarray))

		diskplushalo_pdf = multidiskmodel_pdf + parlist[8]*halomodel_pdf
		halofrac = parlist[8]

	return diskmodel_pdf,tdiskmodel_pdf,multidiskmodel_pdf,halomodel_pdf,diskplushalo_pdf,halofrac

def get_blob_probs(blob): #feed in after burnin done
	#blob is lchain*nstars array*2
	flat_lng_m33d = blob[:,:,1]
	flat_lng_m33h = blob[:,:,0]
	#check if actually fit a halo: otherwise log-likelihood for halo is not sensible
	if flat_lng_m33h[0,:].sum()==0.0:
		post_prob_d_arr = np.exp(flat_lng_m33d - flat_lng_m33d) #now that's per-star *and* per-step
	else:
		post_prob_d_arr = np.exp(flat_lng_m33d - np.logaddexp(flat_lng_m33d, flat_lng_m33h)) #now that's per-star *and* per-step
	post_prob_d_percentiles = np.percentile(post_prob_d_arr, [16,50,84], axis=0) #per star
	post_prob = post_prob_d_percentiles.T
	return post_prob



def major_minor_transform(coords, pa, centercoords = m33coord, return_xi_eta = False):
    # for a given skycoordinate object/array, and passed position angle with units, compute and return coordinates in receding semi-major axis, minor axis plane

    #convert to xi and eta centered on M33
    c_inm33 = coords.transform_to(centercoords.skyoffset_frame())
    xi, eta = c_inm33.lon.degree, c_inm33.lat.degree

    # coordinates transformation: put in coordinates of (receding semi)major and minor axes
    alpha = eta * np.cos(pa) + xi * np.sin(pa) # major axis
    beta =  - eta * np.sin(pa) + xi * np.cos(pa) # minor axis

    if return_xi_eta:
        return alpha, beta, xi, eta
    else:
        return alpha, beta # in degrees

# def major_minor_transform_reverse(alpha, beta, pa, centercoords = m33coord, return_xi_eta = False):
#     # for a receding major axis, minor axis coordinate, and passed position angle with units, compute and return sky coordinates
#     if return_xi_eta:
#         return skycoords, xi, eta
#     else:
#         return skycoords # in degrees

def m33_tilted_ring_deproj_radius_init(coords):
    # given skycoord object or array, compute (zeroth order) the deprojected radius from M33's center assuming an average (single fixed) PA and inclination for M33, return deprojected radius in arcmin

	alpha, beta = major_minor_transform(coords, m33_pa, centercoords=m33coord)
	ang_dist = np.sqrt(alpha**2 + (beta / np.cos(m33_inclination))**2)
	# print('mmtransform in dprojinit done')
	return ang_dist # # in degrees

def m33_tilted_ring_deproj_converge(skycoord, iconv, paconv, funcinc, funcpa, verbose=False):
    # given:
    # skycoord  sky coordinate object,
    # iconv, paconv    convergence criteria for inclination and pa
    # funcpa, funcinc   function for inclination and pa as function of ang distance in plane of disk

    # iteratively determine angular distance in plane of disk from revised inclination, PA until change in inclination and PA given estimate of angular distance in plane of disk converges.

    # compute initial values using global average m33 incl, pa
    Rinit = m33_tilted_ring_deproj_radius_init(skycoord)
    alpha, beta = major_minor_transform(skycoord, funcpa(Rinit*60.)*u.degree, centercoords=m33coord)
    ang_dist_0 = np.sqrt(alpha**2 + (beta / np.cos(funcinc(Rinit*60.)*u.degree))**2.) # in degrees

    # incl, pa corresponding to Rinit
    incl_init = funcinc(Rinit*60.)*u.degree
    pa_init = funcpa(Rinit*60.)*u.degree
    # values calculated from first computation of angular distance in plane disk based on incl_init and pa_init
    incl = funcinc(ang_dist_0*60.)*u.degree
    pa = funcpa(ang_dist_0*60.)*u.degree

    delta_incl = abs(incl - incl_init)
    delta_pa = abs(pa - pa_init)

    # compute ang_dist computed with incl, pa
    alpha, beta = major_minor_transform(skycoord, pa, centercoords=m33coord)
    ang_dist = np.sqrt(alpha**2 + (beta / np.cos(incl))**2.) # in degrees

    niter = 0
    while delta_incl > iconv or delta_pa > paconv:
        niter += 1
        incl_p, pa_p = incl, pa
        # first update incl and pa of model for current angular distance
        incl = funcinc(ang_dist*60.)*u.degree
        pa = funcpa(ang_dist*60.)*u.degree
        # now recompute angular distance
        alpha, beta = major_minor_transform(skycoord, pa, centercoords=m33coord)
        ang_dist = np.sqrt(alpha**2 + (beta / np.cos(incl))**2.) # in degrees
        # and recompute deltas
        delta_incl = abs(incl - incl_p)
        delta_pa = abs(pa - pa_p)

    if verbose==True:
        print('Convergence reached in ', niter, ' iterations.')

    return ang_dist, incl, pa

def m33_tilted_ring_deproj_radius(coords, model=diskmodel, verbose=False):
	# given skycoord object or array compute a zeroth order deprojected radius, then iterate until converge to deprojected radius taking into account disk model (varying) PA and inclination with radius
	# return deprojected radius in arcmin

	converge_i = 0.01*u.degree # convergence criterion in degree
	# 0.025 is 1/4 the change of inclination per resolution element of the model (which is a steady 0.1 degree per 2 arcmin in radius)
	converge_pa = 0.35*u.degree # convergence criterion in degree
	# over radial range of model, PA changes from a steady 201.3 to nearly 160 degrees. decreases fairly quickly though from 201 to ~175 over ~15 arcmin (by ~3.5 deg per 2arcmin)

    # compute zeroth order deprojected radius
	Rinit = m33_tilted_ring_deproj_radius_init(coords) # in degree
	# print('deproj_radius_init in deprojr done')

	# interpolate between Kam et al. 2017 data points for vrot, major axis PA, and inclination for radius in ARCMIN
	# fill_value tells it what to do if outside bounds of data: currently "extrapolate", but ideally set to first,last value in the model
	#f_pa = interp1d(model['Radius_arcmin'], model['PA_deg'], fill_value=(model['PA_deg'][0], model['PA_deg'][-1]))
	#f_incl = interp1d(model['Radius_arcmin'], model['i_deg'], fill_value=(model['i_deg'][0], model['i_deg'][-1]))
	f_pa = interp1d(model['Radius_arcmin'], model['PA_deg'], fill_value="extrapolate")
	f_incl = interp1d(model['Radius_arcmin'], model['i_deg'], fill_value="extrapolate")

	# Do a single iteration from Rinit, this time using appropriate PA and inclination from disk model for deprojected radius of Rinit
	alpha, beta = major_minor_transform(coords, f_pa(Rinit*60.)*u.degree, centercoords=m33coord)
	# print('mmtransform in deprojr done')

	# deprojection needs to happen for minor axis direction
	ang_dist = np.sqrt(alpha**2 + (beta / np.cos(f_incl(Rinit*60.)*u.degree))**2.) # in degrees
	# print(ang_dist,Rinit)

	# values corresponding to Rinit, calculated assuming global, avg values m33_pa and m33_inclination
	incl_init = f_incl(Rinit*60.)*u.degree
	pa_init = f_pa(Rinit*60.)*u.degree
	# values calculated from first computation of angular distance in plane disk based on incl_init and pa_init
	incl = f_incl(ang_dist*60.)*u.degree
	pa = f_pa(ang_dist*60.)*u.degree

	delta_incl = incl - incl_init
	delta_pa = pa - pa_init

	count=0

	# need to somehow manage single value passes
	if len(np.atleast_1d(ang_dist)) > 1:
		for di,dp,c,r,i in zip(delta_incl, delta_pa, coords, Rinit, range(len(np.atleast_1d(ang_dist)))):
			if abs(di) > converge_i or abs(dp) > converge_pa: # both in degree
				# for this star, iterate until reach convergence
				conv_ang_dist, conv_incl, conv_pa = m33_tilted_ring_deproj_converge(c, converge_i, converge_pa, f_incl, f_pa, verbose=verbose)
				if verbose:
					print('Above a convergence criterion, delta_inclination is: ', di, ' delta_pa is: ', dp)
					print(r, ' -> ', conv_ang_dist)
				# update values for this star using converged values
				ang_dist[i] = conv_ang_dist
				incl[i] = conv_incl
				pa[i] = conv_pa
			elif abs(di) <= converge_i and abs(dp) <= converge_pa:
				count+=1
			else:
				print('Something broke')
		# print('deproj_converge loop done')
	else: # only one coordinate was passed
		if abs(delta_incl) > converge_i or abs(delta_pa) > converge_pa: # both in degree
			# for this star, iterate until reach convergence
			conv_ang_dist, conv_incl, conv_pa = m33_tilted_ring_deproj_converge(coords, converge_i, converge_pa, f_incl, f_pa, verbose=verbose)
			if verbose:
				print('Above a convergence criterion, delta_inclination is: ', delta_incl, ' delta_pa is: ', delta_pa)
				print(r, ' -> ', conv_ang_dist)
			# update values for this star using converged values
			ang_dist = conv_ang_dist
			incl = conv_incl
			pa = conv_pa
		elif abs(delta_incl) <= converge_i and abs(delta_pa) <= converge_pa:
			count+=1
		else:
			print('Something broke')
	if verbose:
		print('Number of stars where first iteration is\n < convergence criterion for both inclination (', converge_i, ') and PA (',
              converge_pa, ') is: ', count)
		print('Total number of stars: ', len(delta_incl))

    # return final ang_dist array
	alpha, beta = major_minor_transform(coords, f_pa(ang_dist*60.)*u.degree, centercoords=m33coord)
	ang_dist2 = np.sqrt(alpha**2 + (beta / np.cos(f_incl(ang_dist*60.)*u.degree))**2.) # in degrees

	return ang_dist2, Rinit, incl, pa # all in degrees, incl and pa have units attached

def m33_tilted_ring_deproj_angle(coords, incl=m33_inclination, pa=m33_pa):
	# given skycoord object or array, compute the azimuthal angle(s) in the plane of the galaxy, relative to the receding semimajor axis. Return in degrees.
	# If pa and incl not set in call, assuming average PA and inclination for M33

	# compute position in coordinate system in plane of sky, rotated so x-axis (alpha) is the receding semi-major axis
	alpha, beta = major_minor_transform(coords, pa, centercoords=m33coord) # in deg
	# print('mmtrans inangle done')
	rdeproj = m33_tilted_ring_deproj_radius_init(coords) # in deg
	# print('deproj_radius_init inangle done')

	# compute the deprojected angle of the star in the plane of the disk from the receding semimajor axis
	# the “y-coordinate” (beta deprojected) is the first function parameter, the “x-coordinate” (alpha) is the second
	theta = np.degrees(np.arctan2(beta/np.cos(incl), alpha))

	# make it [0,360] degrees
	if getattr(theta, 'shape', ()):
		num = len(theta)
	else:
		num = 1

	if num == 1:
		if theta < 0:
			theta += 360.*u.degree
	else:
		for t in range(len(theta)):
			if theta[t] < 0:
				theta[t]+=360.*u.degree

	return theta # in degrees, with units attached

def m33_tilted_ring_contour(xarray, yarray, model=diskmodel, verbose=False):
    # given an ra and dec, compute the observed line of sight disk velocity according to the Kam et al. 2017 HI disk model.
    # returns an array of line of sight disk velocities corresponding to the input x,y arrays

    # interpolate between Kam et al. 2017 data points for vrot, major axis PA, and inclination
    f_vrot = interp1d(model['Radius_arcmin'], model['Vrot_kms'])

    vobs = []
    for y in yarray:
        vobs_row =[]
        print('Iterating through grid at dec = ', y)
        for x in xarray:
            sc = SkyCoord(ra = x, dec = y, unit=(u.deg, u.deg))
            # compute the deprojected radius assuming Kam et al disk model
            r, Rinit, incliter, paiter = m33_tilted_ring_deproj_radius(sc, verbose=verbose) # all in degrees, incl and pa with units
            theta = m33_tilted_ring_deproj_angle(sc, incl=incliter, pa=paiter) # degrees, with units

            if (r*60. > np.min(model['Radius_arcmin'])) and (r*60. < np.max(model['Radius_arcmin'])):
                vobs_r = m33_sys + f_vrot(r*60.)*np.cos(theta)*np.sin(incliter)
                vobs_row.append(vobs_r)
            else:
                #print('Grid outside model at radius ',r,' , coordinates ', sc)
                vobs_row.append(np.nan)
        vobs.append(np.array(vobs_row))

    return np.array(vobs)

def m33_tilted_ring(skycoord, model=diskmodel, frot=1.0):
	# given an array SkyCoord objects, compute the observed disk velocity  according to the Kam et al. 2017 HI disk model at each SkyCoord
	# frot is for setting rotation speed to be some fraction of HI model rotation speed

	# compute the deprojected radius assuming Kam et al disk model and average PA and inclination for M33
	radius, Rinit, incliter, paiter = m33_tilted_ring_deproj_radius(skycoord) # all in degrees, incl and pa with units
	# print(radius, Rinit)
	# print('deproj_Rad_done')
	theta = m33_tilted_ring_deproj_angle(skycoord, incl=incliter, pa=paiter) # degrees, with units
	# print('deproj_angle_done')

	# interpolate between Kam et al. 2017 data points for vrot
	f_vrot = interp1d(model['Radius_arcmin'], model['Vrot_kms'])

	vobs=[]

	if len(np.atleast_1d(radius)) > 1: #array passed
		for r,i,t in zip(radius, incliter, theta):
			if (r*60. > np.min(model['Radius_arcmin'])) and (r*60. < np.max(model['Radius_arcmin'])):
				vobs_r = m33_sys + f_vrot(r*60.)*frot*np.cos(t)*np.sin(i)
				vobs.append(vobs_r)
			else:
				#print('Grid outside model at radius ',r,' , coordinates ', c)
				vobs.append(np.nan)
	else: #single value passed
		if (radius*60. > np.min(model['Radius_arcmin'])) and (radius*60. < np.max(model['Radius_arcmin'])):
			vobs_r = m33_sys + f_vrot(radius*60.)*frot*np.cos(theta)*np.sin(incliter)
			vobs.append(vobs_r)
		else:
			#print('Grid outside model at radius ',r,' , coordinates ', c)
			vobs.append(np.nan)

	return vobs

def m33_tilted_ring_noiteration(skycoord, model=diskmodel):
    # given an array SkyCoord objects, compute the observed disk velocity  according to the Kam et al. 2017 HI disk model at each SkyCoord

    # compute the deprojected radius assuming Kam et al disk model and average PA and inclination for M33
    radius = m33_tilted_ring_deproj_radius_init(skycoord)*60. # convert to arcmin
    # deprojected angle between star and receding semimajor axis in plane of disk
    theta = m33_tilted_ring_deproj_angle(skycoord) # returned in degrees, with units

    # interpolate between Kam et al. 2017 data points for vrot, major axis PA, and inclination
    f_vrot = interp1d(model['Radius_arcmin'], model['Vrot_kms'])
    f_pa = interp1d(model['Radius_arcmin'], model['PA_deg'])
    f_incl = interp1d(model['Radius_arcmin'], model['i_deg'])

    vobs=[]

    for r,t,c in zip(radius,theta, skycoord):
        if (r > np.min(model['Radius_arcmin'])) and (r < np.max(model['Radius_arcmin'])):
            # The position angle of the kinematical major axis is defined as the counterclockwise angle in the plane of the sky from the north to the receding-side semimajor axis. The angle θ is measured relatively to the semimajor axis.
            #theta = m33coord.position_angle(c).degree - f_pa(r) # absolute value doesn't matter

            vobs_r = m33_sys + f_vrot(r)*np.cos(t)*np.sin(f_incl(r)*u.degree)
            vobs.append(vobs_r)
        else:
            #print('Grid outside model at radius ',r,' , coordinates ', c)
            vobs.append(np.nan)

    return vobs



def lighten_color(color, amount=0.5):
	try:
		c = mc.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mc.to_rgb(c))
	return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mc.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmapd = plt.get_cmap('viridis')

def savenice(name,fighandle):
	fighandle.tight_layout()
	fighandle.savefig(name,bbox_inches='tight')
	fig.close()

def distance(dmod):
    dist = 10.**((dmod + 5.)/5.)/10**3
    return dist  #in kpc

# #-----------Defines colours used in Karrie's M33 RGB analysis-------
#
# crgb = 'rosybrown'
# # dark model on light draws
# crgb_model = 'firebrick' #'indianred' # was 'firebrick'
# crgb_model_draws = 'lightcoral'
# crgb_modelmean, crgb_modelmeanunc = 'darkred', 'lightsalmon' #'salmon', 'mistyrose' # for plotting mean velocity (e.g., for non-rotating halo)
# # # light model on dark draws
# # crgb_model = 'lightcoral' #'indianred' # was 'firebrick'
# # crgb_model_draws = 'maroon'
#
# cyoung = 'lightsteelblue'
# # dark model on light draws
# cyoung_model = 'mediumblue' #'royalblue'
# cyoung_model_draws = 'cornflowerblue'
# cyoung_modelmean, cyoung_modelmeanunc = 'cadetblue', 'powderblue'
# # # light model on dark draws
# # cyoung_model = 'blue' #'royalblue'
# # cyoung_model_draws = 'midnightblue'
#
# # for anytime need to mark a fiducial value
# csys = 'dimgrey'
#
# # for HI disk velocities
# ch1disk = 'cornflowerblue'
#
# # for plots with multiple components
# cdisk = cyoung_model # was'steelblue'
# cdiskdraws = cyoung_model_draws
# cdisk2 = 'rebeccapurple' # was 'midnightblue'
# cdisk2draws = 'indigo'
#
# # light model on dark draws
# chalo = crgb_model # was 'firebrick'
# chalodraws = crgb_model_draws
# chalo2 = 'rosybrown'
# chalo2draws = 'lightcoral'
#
# # dark model on light draws
# # ctotal = 'tab:green'
# # ctotaldraws = 'palegreen'
# ctotal = 'sienna'
# ctotaldraws = 'orange' # 'sandybrown'
# # # #light model on dark draws
# # ctotal = 'sandybrown'
# # ctotaldraws = 'saddlebrown'
#
# alpha_sep = 1.0
# alpha_comb = 0.3
# alpha_combscatt = 0.75
#
# allrgb50th = 'burlywood'
# allrgbshade= 'oldlace' #'antiquewhite' #'whitesmoke'

# #-----------Defines colours used in Lara's analysis-------
cmapL = truncate_colormap(plt.get_cmap('plasma'),0.00,0.85)
pt_to_get_cols = np.linspace(0,1,4)
col_hexes = cmapL(pt_to_get_cols)
c4fil = tuple(map(tuple,col_hexes[:,0:3]))

#main RGB colour: bluish (mid)
crgb = 'rebeccapurple' #col_hexes[0]
# crgb_model = col_hexes[1]
# crgb_model_draws = lighten_color(col_hexes[1])
# crgb_modelmean, crgb_modelmeanunc = 'darkred', 'lightsalmon' # for plotting mean velocity (e.g., for non-rotating halo)

#main AGB colour: purpleish (bottom)
cyoung = 'rebeccapurple'
# cyoung_model = col_hexes[0]
# cyoung_model_draws = lighten_color(col_hexes[0],0.4)
# cyoung_modelmean, cyoung_modelmeanunc = 'cadetblue', 'powderblue'

#main carbon colour: greenish (top)
# ccarbon = 'mediumseagreen'
# ccarbon_model = col_hexes[2]
# ccarbon_model_draws = lighten_color(col_hexes[2])

# for anytime need to mark a fiducial value
csys = 'dimgrey'

# for HI disk velocities
ch1disk = 'cornflowerblue'

# for plots with multiple components
cdisk = col_hexes[3]
cdiskdraws = lighten_color(col_hexes[3])

chalo = col_hexes[1]
chalodraws = lighten_color(col_hexes[1])

ctotal = col_hexes[2] #closest to fill colour
ctotaldraws = lighten_color(col_hexes[2])

# chalo2 = 'rosybrown'
# chalo2draws = 'lightcoral'
# cdisk2 = 'rebeccapurple'
# cdisk2draws = 'indigo'

alpha_sep = 1.0
alpha_comb = 0.3
alpha_combscatt = 0.75

allrgb50th = 'burlywood'
allrgbshade= 'oldlace' #'antiquewhite' #'whitesmoke'




def run_population_fits(
    summary,
    age_groups=['young', 'int', 'old'],
    optflags='TTFF',
    modelset=3,
    burncut=2.0,
    abridged=False
):
    """
    Runs velocity mixture model fits for each age group in the provided summary DataFrame,
    and prints the evidence and best-fit parameters (with ±1σ uncertainties).
    """

    # Parameter lists by modelset
    params_dict = {
        1: ['disklag', 'disksig', 'fhalo', 'halocen', 'halosig'],
        2: ['disklag', 'disksig', 'fhalo', 'halocen', 'halosig'],
        3: ['disklag', 'disksig', 'fhalo', 'halocen', 'halosig', 'halolag'],
        4: ['disklag', 'disksig', 'tdisklag', 'tdisksig', 'ftdisk', 'halocen', 'halosig', 'fhalo'],
        5: ['disklag', 'disksig', 'tdisklag', 'tdisksig', 'ftdisk', 'halocen', 'halosig', 'halolag', 'fhalo']
    }
    params_list = params_dict[modelset]

    for population in age_groups:
        print(f"\n=== Fitting {population} population ===")

        # -------- EXTRACT DATA --------
        v_obs = summary[f'{population}_vcorr_stat'].values
        ra = summary[f'{population}_RA'].values
        dec = summary[f'{population}_DEC'].values
        coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        deproj_radius = np.ones(len(summary))  # Replace with actual deproj radius if available

        # -------- CONVERT TO ASTROPY TABLE --------
        table_data = Table.from_pandas(summary)
        table_data['VCORR_STAT'] = v_obs
        model_v = np.array(m33_tilted_ring(coords))

        # -------- FIT THE MODEL --------
        ev, qres, numn, fn, prob = fit_model(
            table_data,
            deproj_radius,
            coords,
            model_v,
            f'{population}_fit',
            modelset,
            optflags,
            burncut=burncut,
            abridged=abridged,
            startype=population   # <<< pass population label to fit_model
        )

        # -------- PRINT RESULTS --------
        print(f"  log(Z): {ev[0]:.2f} ± {ev[1]:.2f}")
        for name, q in zip(params_list, qres):
            print(f"  {name:>10s}: {q[1]:.2f} [+{q[2]-q[1]:.2f}, -{q[1]-q[0]:.2f}]")
