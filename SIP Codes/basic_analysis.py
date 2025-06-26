import numpy as np
import matplotlib.pyplot as plt
# from datetime import datetime

import scipy
from scipy.stats import norm
from astroML.stats import binned_statistic_2d

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column, vstack
from astropy import constants as const
from sklearn.neighbors import KDTree

import warnings
warnings.filterwarnings("ignore", message="'second' was found  to be '60.0'")
warnings.filterwarnings("ignore", message="invalid value encountered in log")
warnings.filterwarnings("ignore", message="Creating an ndarray from ragged nested sequences")
warnings.filterwarnings("ignore", message="Starting from Matplotlib 3.6, colorbar()")
np.set_printoptions(threshold=np.inf)

spec_direct = '/media/sf_M33_postdoc_work/spectra_analysis'
exec(open("/media/sf_Actual_PhD_work_stuff/convenience.py",encoding="utf-8").read()) #includes mpl ticker, cm, colors, truncate, and lighten, + mag to dist (kpc) conversion
exec(open(f"{spec_direct}/color_def.py",encoding="utf-8").read()) #sets color for plots

import argparse
parser = argparse.ArgumentParser(description="", usage="python name.py --field Field_name")

parser.add_argument("-st", "--startype",help="Type of stars:RGB/AGB/young",required=True,type=str)
parser.add_argument("-sn", "--sncut",help="Minimum S/N for data if implementing cut",type=float,default=0.0)
parser.add_argument("-zq", "--zqcut",help="Minimum zquality to use (default=3)",type=int,default=3)
parser.add_argument("-abc", "--abandcut",help="A-band sigma to cut beyond if implemented",type=float)
parser.add_argument("-dmod", "--dmod",help="Distance modulus used to separate RGB/AGB (default=notset=bestfit 24.67)",type=float)
parser.add_argument("-bc", "--burncut",help="Factor to divide blob length by (between 1-2: 2 cuts half the chain)",type=float,default=2.0)
parser.add_argument("-ps", "--psplit",help="Whether to run full position-split set (T/default F)",type=str,default='F')
parser.add_argument("-ms", "--modset",help="Which model set to use:1-5 inclusive",required=True,type=int)
parser.add_argument("-mt", "--modtype",help="Which model variations to run/compare in TTFF,TFFF format",required=True,type=str)
parser.add_argument("-abr", "--abridged",help="Run abridged version (make no plots T / make only 'nice' plots NP / default F)",type=str,)
parser.add_argument("-nst", "--nstest",help="Run N/S split test: T/default F)",type=str,)
parser.add_argument("-rt", "--rtest",help="Run radius split test: T/default F)",type=str,)
parser.add_argument("-nsrt", "--nsrtest",help="Run radius+N/S split test: N/S/default F)",type=str,)
parser.add_argument("-ext", "--extest",help="run N/S test for things<30 only: T/default F)",type=str,)
parser.add_argument("-ss", "--savesamp",help="Save reference sample of things fit for various comparisons",type=str,)
parser.add_argument("-piry", "--phatiry",help="Run on Phatter-IR selected AGB stars: T/F(my equiv sample)/default null",type=str,)
parser.add_argument("-scmd", "--savecmd",help="Save cmd posns and disk prob for a given same for later plotting ()",type=str,)

argmts = parser.parse_args()

if argmts.abridged=='NP':
	plt.rcParams.update({'xtick.major.pad':9, 'ytick.major.pad':9, 'xtick.major.width':2,'ytick.major.width':2, 'xtick.major.size':5, 'ytick.major.size':5, 'xtick.top':True, 'xtick.direction':'in', 'ytick.right':True, 'ytick.direction':'in'})
	plt.rcParams.update({'xtick.minor.visible':True, 'ytick.minor.visible':True, 'xtick.minor.width':1, 'ytick.minor.width':1,'xtick.minor.size':3, 'ytick.minor.size':3, 'ytick.minor.right':True, 'xtick.minor.top':True})
	plt.rcParams.update({'font.size': 25,'figure.dpi': 300,'savefig.bbox':'tight','legend.fontsize': 20})#,'axes.axisbelow':False})
	from matplotlib import rc
	rc('text', usetex=True)
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

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

diskmodel = Table.read(f'{spec_direct}/Kam2017_table4.dat', format='ascii', names=['Radius_arcmin', 'Radius_kpc', 'Vrot_kms', 'Delta_Vrot', 'i_deg', 'PA_deg'])
exec(open(f"{spec_direct}/diskmodel_functions.py",encoding="utf-8").read()) #describes deprojection and tilted ring model

#--------Load data, filter data through quality cuts----------
if argmts.dmod:
	alldata = Table.read(f'{spec_direct}/m33_vels_stars_{argmts.dmod}_donedupes.fits')
	kv = f'all{argmts.startype}{argmts.dmod}'
else:
	kv = f'all{argmts.startype}{argmts.zqcut}{argmts.sncut}'#{argmts.abandcut}
	if argmts.phatiry=='T':
		alldata = Table.read(f'{spec_direct}/phatter_irselsamp.fits')
	elif argmts.phatiry=='F':
		alldata = Table.read(f'{spec_direct}/phatter_irselcompsamp.fits')
	else:
		alldata = Table.read(f'{spec_direct}/m33_vels_stars_donedupes.fits')

alldata['coord'] =  SkyCoord(ra=alldata['RA'], dec=alldata['DEC'], unit=(u.hourangle, u.deg))
# years = np.array([datetime.fromisoformat(k).year for k in alldata['DATE']])
# alldata.add_column(years,name='YEAR')
alldata
alldata['GRATING'][1]

vlos_max = 50.0
vlos_min = -500.0
bad_band = ['pTH1','pTH2','pTH3','pTH4','pTSt1','pTSt2','pTSt3','pTN9','pTN8','pTN7'] #now consistent with those used to make posn plots... extras cut (affects only RGB) are pTN9, pTN8, pTN7, and number differences minimal
# vunc_cut = 15.*u.km/u.s
# abandmax = 80.*u.km/u.s # Raja expects max aband correction to be ~+/-60km/s for 600l/mm grating
# sncut=0
# sel_qual = [(alldata['VCORR_STAT'] >= vlos_min) & (alldata['VCORR_STAT'] <= vlos_max) & (alldata['ZQUALITY'] >= 3) & (alldata['SN'] >= sncut)][0]
sel_qual = [(alldata['VCORR_STAT'] >= vlos_min) & (alldata['VCORR_STAT'] <= vlos_max) & (alldata['ZQUALITY'] >= argmts.zqcut) & (alldata['SN'] >= argmts.sncut)][0]
sel_pos = ~np.array([(alldata['MASKNAME'][k] in bad_band) for k in range(len(alldata))])
# sel_dupe = [(alldata['DUPLICATE']==0)][0]
if argmts.startype=='AGB':
	sel_type = [((alldata['AGB_SEL']==1) | (alldata['CBN_SEL']==1)) & (alldata['WCN_SEL']!=1) & (alldata['FG_SEL']!=1) ][0]
elif argmts.startype=='AGBnoC':
	sel_type = [(alldata['AGB_SEL']==1) & (alldata['WCN_SEL']!=1) & (alldata['CBN_SEL']!=1) & (alldata['FG_SEL']!=1) ][0]
elif argmts.startype=='carbon':
	sel_type = [(alldata['CBN_SEL']==1) & (alldata['WCN_SEL']!=1) & (alldata['FG_SEL']!=1) ][0]
elif argmts.startype=='RGB':
	sel_type = [(alldata['RGB_SEL']==1) & (alldata['WCN_SEL']!=1) & (alldata['FG_SEL']!=1) & (alldata['CBN_SEL']!=1)][0]
elif argmts.startype=='young':
	sel_type = [((alldata['RHB_SEL']==1) | (alldata['WCN_SEL']==1)) & (alldata['FG_SEL']!=1)][0]
elif argmts.startype=='wcn':
	sel_type = [(alldata['WCN_SEL']==1) & (alldata['CBN_SEL']!=1) & (alldata['FG_SEL']!=1)][0]

if argmts.abandcut:
	sel_abc = [((alldata['MASK_X'] >= alldata['FIT_MIN_MASK_X']) & (alldata['MASK_X'] <= alldata['FIT_MAX_MASK_X'])) | (abs(alldata['ABAND_STAT']) <= abs(alldata['ROLL_ABAND_MEAN']) + argmts.abandcut*alldata['ROLL_ABAND_STD'])][0]
	# sel_abc = [((alldata['MASK_X'] >= alldata['FIT_MIN_MASK_X']) & (alldata['MASK_X'] <= alldata['FIT_MAX_MASK_X'])) | (abs(alldata['ABAND_STAT']) <= abs(alldata['ROLL_ABAND_MEAN']) + 3*alldata['ROLL_ABAND_STD'])][0]
	filt = np.logical_and.reduce((sel_qual, sel_type, sel_pos, sel_abc))
else:
	filt = np.logical_and.reduce((sel_qual, sel_type, sel_pos)) #, sel_dupe #can add extra filts
# print(len(sel_qual),sel_pos.shape,len(sel_type),len(sel_abc),len(filt),filt.sum())
good_data = alldata[filt] #actual filtering occurs here
print(f"number of stars fit: {len(good_data)}")
print(good_data['ROLL_ABAND_STD'].mean(),np.median(good_data['ROLL_ABAND_STD']),good_data['ROLL_ABAND_STD'].std())

# if argmts.savesamp=='T':
# 	good_data.write(f'{spec_direct}/refsamp{argmts.startype}{argmts.sncut}{argmts.zqcut}{argmts.abandcut}.fits',overwrite=True)
# 	import sys
# 	sys.exit()

'''
# calculate average vel uncertainty for sample
sysunc = np.array([5.6 if i=='600ZD' else 2.2 for i in good_data['GRATING']])
fullunc = np.sqrt(sysunc**2+good_data['VERR']**2)
print(np.median(fullunc)) #np.mean(fullunc),np.std(fullunc)
#'''

Riter, Rinit, incliter, paiter = m33_tilted_ring_deproj_radius(good_data['coord'], verbose=False)
theta = m33_tilted_ring_deproj_angle(good_data['coord'])
agb_model_vel = np.array(m33_tilted_ring(good_data['coord']))

''' Make diagnostic plots
fig,ax = plt.subplots(1,1,figsize=(8,8)) # Plot losvel as function of on-sky position
colorvals = good_data['VCORR_STAT']
vc = ax.scatter(good_data['coord'].ra.deg, good_data['coord'].dec.deg, c=colorvals, s=10)
cbar = fig.colorbar(vc,ax=ax)
cbar.set_label('Velocity (km s$^{-1}$)')
ax.set_xlabel('RA (deg)')
ax.set_ylabel('Dec (deg)')
# fig.savefig(f'{spec_direct}/AGB_paper_figs/{kv}_spdv.pdf')
plt.close()

fig,ax = plt.subplots(2,1,figsize=(8,8)) # Plot losvel as function of on-sky position
bv = np.arange(-325,-45,10)
bsn = np.arange(0,50,1)
ax[0].hist(good_data['VCORR_STAT'],bins=bv,density=True,ec='k',lw=1)
ax[1].hist(good_data['SN'],bins=bsn,density=True,ec='k',lw=1)
ax[0].set_xlabel('Velocity (km/s)')
ax[1].set_xlabel('S/N')
# fig.savefig(f'{spec_direct}/AGB_paper_figs/{kv}_hist.pdf')
plt.close()

fig,ax = plt.subplots(1,1,figsize=(8,8)) # Plot losvel as function of on-sky position
colorvals = good_data['SN']
vc = ax.scatter(good_data['coord'].ra.deg, good_data['coord'].dec.deg, c=colorvals, s=10,vmin=0,vmax=20)
cbar = fig.colorbar(vc,ax=ax)
cbar.set_label('S/N')
ax.set_xlabel('RA (deg)')
ax.set_ylabel('Dec (deg)')
# fig.savefig(f'{spec_direct}/AGB_paper_figs/{kv}_spdsn.pdf')
plt.close()

def plot_cmd_sn(tbdata, probs, figname,vmax): # plotdisk=False,
	m_cfht = ((~np.isnan(tbdata['G0_CFHT']))) #& (tbdata['ZQUALITY'] >= 3)
	m_475 = ((~np.isnan(tbdata['F475W0_ACS']))) # & (tbdata['ZQUALITY'] >= 3)
	m_606 = ((~np.isnan(tbdata['F606W0_ACS']))) # & (tbdata['ZQUALITY'] >= 3)

	fig,ax = plt.subplots(1,3,figsize=(30,10))
	ax[0].scatter(tbdata['G0_CFHT'][m_cfht] - tbdata['I0_CFHT'][m_cfht], tbdata['I0_CFHT'][m_cfht], c=probs[m_cfht], alpha=0.8,vmin=0,vmax=vmax)
	ax[0].set_xlabel('(g - i)0')
	ax[0].set_ylabel('i0')

	dc = ax[1].scatter(tbdata['F475W0_ACS'][m_475] - tbdata['F814W0_ACS'][m_475], tbdata['F814W0_ACS'][m_475], c=probs[m_475], alpha=0.8,vmin=0,vmax=vmax)
	ax[1].set_xlabel('(F475 - F814)0')
	ax[1].set_ylabel('F8140')

	ax[2].scatter(tbdata['F606W0_ACS'][m_606] - tbdata['F814W0_ACS'][m_606], tbdata['F814W0_ACS'][m_606], c=probs[m_606], alpha=0.8,vmin=0,vmax=vmax)
	ax[2].set_xlabel('(F606 - F814)0')
	ax[2].set_ylabel('F8140')

	for a in ax:
		a.invert_yaxis()
	cbar = plt.colorbar(dc)
	cbar.set_label('S/N')
	cbar.set_alpha(1)
	cbar.draw_all()
	fig.tight_layout()
	fig.savefig(f'{spec_direct}/AGB_paper_figs/{figname}',bbox_inches='tight')
	plt.close(fig)
	return
# plot_cmd_sn(good_data,good_data['SN'],f'{kv}_cmdsn.pdf',20)

halfN = good_data[(agb_model_vel < m33_sys)]
halfS = good_data[(agb_model_vel > m33_sys)]
# print(halfN['SN'].mean(),halfS['SN'].mean())
# print(np.median(halfN['SN']),np.median(halfS['SN']),len(halfN),len(halfS))
#'''

''' Make veldisp plots like Amanda - full samp
X = np.zeros((len(good_data), 2))
X[:,0] = good_data['RA_DEG'] #in degrees
X[:,1] = good_data['DEC_DEG'] #in degrees
kdt = KDTree(X, metric='euclidean')
distances,indices = kdt.query(X, k=101) #default: use the 32 nearest neighbors
distances_arcsec = distances*3600
radinit=50 #arcsec
radmax = 300 #arcsec
nummin=15

rlist,dlist,vlist,plist = [],[],[],[]
for i in range(len(good_data)):
	r_curr=radinit
	numin = np.array([distances_arcsec[i]<r_curr][0]).sum()
	while numin<nummin:
		r_curr+=5
		if r_curr>radmax:
			break
		isin = [distances_arcsec[i]<r_curr][0]
		whichin = indices[i][isin]
		numin = np.array(isin).sum()

	if r_curr<radmax:
		rlist.append(r_curr)
		plist.append(good_data['coord'][i])
		vels = good_data['VCORR_STAT'][whichin]
		if argmts.startype=='RGB':
			grat=good_data['GRATING'][whichin]
			velunc = np.array([np.sqrt(5.6**2+1.65**2) if grat[v]=='600ZD' else np.sqrt(2.2**2+1.85**2) for v in range(len(grat))])
		else:
			velunc = good_data['VERR'][whichin]
		unnw = velunc**(-2)
		normw = unnw/(unnw.sum())
		meanv = np.average(vels,weights=normw)
		vlist.append(meanv)
		disp = np.sqrt((((vels-meanv)**2)*normw).sum())
		dlist.append(disp)

fig,ax = plt.subplots(1,2,figsize=(16,8),sharex=True,sharey=True) # Plot losvel as function of on-sky position
vc = ax[0].scatter([p.ra.deg for p in plist], [p.dec.deg for p in plist], c=vlist, s=10,vmin=-260,vmax=-100)
dc = ax[1].scatter([p.ra.deg for p in plist], [p.dec.deg for p in plist], c=dlist, s=10,vmin=0,vmax=45)
cbar1 = fig.colorbar(vc,ax=ax[0])
cbar1.set_label('Velocity (km s$^{-1}$)')
cbar2 = fig.colorbar(dc,ax=ax[1])
cbar2.set_label('Velocity dispersion (km s$^{-1}$)')
for a in ax:
	a.set_xlabel('RA (deg)')
ax[0].set_ylabel('Dec (deg)')
fig.tight_layout()
fig.savefig(f'{spec_direct}/AGB_paper_figs/{kv}_disp.pdf')
plt.close()
#'''

#---Set up some plotting params-----
y_lim_factor = 0.3 # multiplicative factor to add to the max of the 50th % PDF for the y limit for model/data velocity histogram plots
histylabel = 'Probability Density'
legloc='best'
ylim_pdf_pad = 0.2

radinit=50 #arcsec
radmax = 300 #arcsec
nummin=15

#--------Set up priors for fitting-------
pri_disk_dispmax = 50.
pri_halo_dispmax = 100.
pri_hfrac_max = 0.85
pri_frothalomin = -1.3 # allows halo to counter-rotate. maximum inherently 1: can't have halo rotating faster than disk
pri_frotdiskmax = 1.5 # allows disk to rotate faster than HI. minimum inherently 0: don't allow disk to counter-rotate relative to stars

# initialize empty dictionaries to be used at end for plotting parameter results as function of radius and spatial bins
sigdisk, sighalo, fhalo, frothalo, frotdisk = {}, {}, {}, {}, {}

# prepare to save a bunch of differently calculated disk probabilities (with halo probability = 1 - disk probability)
# dprob_chain_radial = Column([(np.nan,np.nan,np.nan)]*len(good_data), name='DPROB_CHAIN_RADIAL')
dprob_50th_radial = Column([np.nan]*len(good_data), name='DPROB_50TH_RADIAL')
# dprob_chain_spatial = Column([(np.nan,np.nan,np.nan)]*len(good_data), name='DPROB_CHAIN_SPATIAL')
dprob_50th_spatial = Column([np.nan]*len(good_data), name='DPROB_50TH_SPATIAL')

exec(open(f"{spec_direct}/dyn_modfuncs_fraclag.py",encoding="utf-8").read()) #describes deprojection and tilted ring model

#-------------nested model------------
# want the 'ratio' to be positive if 1st model has more evidence than second' negative means other way around, and 0 means each equally good descriptors

mt_list = np.array([item for item in argmts.modtype.split(',')])
elist,rlist,numl =[],[],[]

if argmts.nstest=='T':
	keyvals = [f'{kv}N', f'{kv}S']
	for k in keyvals:
		for mod in mt_list:
			print(f'Running {mod}')
			evn,qresn,numn,fn,prob = fit_model(good_data, Riter, good_data['coord'], agb_model_vel, k, argmts.modset, mod, burncut=argmts.burncut, abridged=argmts.abridged)
			elist.append(evn)
			rlist.append(qresn)
			numl.append(numn)

			restosave = []
			for r in range(qresn.shape[0]):
				resrow = take_deltas(qresn[r,:])
				restosave.append(resrow)
			np.savetxt(f'{spec_direct}/{fn}res.txt',np.array(restosave), '%s')

	for i in range(len(elist)-1):
		print(argmts.startype,argmts.zqcut,argmts.sncut)
		print((elist[i][0]-elist[i+1][0])/np.log(10))

elif argmts.rtest=='T':
	keyvals = [f'{kv}in', f'{kv}mid', f'{kv}midin', f'{kv}midout', f'{kv}out']
	minrs = [0.0,15.0,0.0,15.0,30.0]
	maxrs = [15.0,30.0,30.0,100.0,100.0]

	for k,minrv,maxrv in zip(keyvals,minrs,maxrs):
		for mod in mt_list:
			print(f'Running {mod}')
			evn,qresn,lfd,fn,prob = fit_model(good_data, Riter, good_data['coord'], agb_model_vel, k, argmts.modset, mod, burncut=argmts.burncut, abridged=argmts.abridged, minr=minrv, maxr=maxrv)
			elist.append(evn)
			rlist.append(qresn)
			numl.append(lfd)

	for i in range(len(elist)-1):
		print(argmts.startype,argmts.zqcut,argmts.sncut)
		print(numl[i])
		print((elist[i][0]-elist[i+1][0])/np.log(10))

elif argmts.nsrtest:
	keyvals = [f'{kv}{argmts.nsrtest}in', f'{kv}{argmts.nsrtest}mid', f'{kv}{argmts.nsrtest}out', f'{kv}{argmts.nsrtest}'] #f'{kv}{argmts.nsrtest}midin', f'{kv}{argmts.nsrtest}midout',
	minrs = [0.0,15.0,30.0,0] #0.0,15.0,
	maxrs = [15.0,30.0,100.0,100] #30.0,100.0,

	for k,minr,maxr in zip(keyvals,minrs,maxrs):
		for mod in mt_list:
			print(f'Running {mod}')
			evn,qresn,lfd,fn,prob = fit_model(good_data, Riter, good_data['coord'], agb_model_vel, k, argmts.modset, mod, burncut=argmts.burncut, abridged=argmts.abridged, minr=minr, maxr=maxr)
			elist.append(evn)
			rlist.append(qresn)
			numl.append(lfd)

			restosave = []
			for r in range(qresn.shape[0]):
				resrow = take_deltas(qresn[r,:])
				restosave.append(resrow)
			np.savetxt(f'{spec_direct}/{fn}res.txt',np.array(restosave), '%s')

	for i in range(len(elist)-1):
		print(argmts.startype,argmts.zqcut,argmts.sncut)
		print(numl[i])
		print((elist[i][0]-elist[i+1][0])/np.log(10))

elif argmts.extest:
	keyvals = [f'{kv}Nmidin', f'{kv}Smidin', f'{kv}midin']
	minrs = [0.0,0.0,0.0] #0.0,15.0,
	maxrs = [30.0,30.0,30.0] #30.0,100.0,

	for k,minr,maxr in zip(keyvals,minrs,maxrs):
		for mod in mt_list:
			print(f'Running {mod}')
			evn,qresn,lfd,fn,prob = fit_model(good_data, Riter, good_data['coord'], agb_model_vel, k, argmts.modset, mod, burncut=argmts.burncut, abridged=argmts.abridged, minr=minr, maxr=maxr)
			elist.append(evn)
			rlist.append(qresn)
			numl.append(lfd)

			restosave = []
			for r in range(qresn.shape[0]):
				resrow = take_deltas(qresn[r,:])
				restosave.append(resrow)
			np.savetxt(f'{spec_direct}/{fn}res.txt',np.array(restosave), '%s')

	for i in range(len(elist)-1):
		print(argmts.startype,argmts.zqcut,argmts.sncut)
		print(numl[i])
		print((elist[i][0]-elist[i+1][0])/np.log(10))

else:
	for mod in mt_list:
		print(f'Running {mod}')
		if argmts.psplit=='T':
			ev,qres,numn,fn,prob = fit_model(good_data, Riter, good_data['coord'], agb_model_vel, kv, argmts.modset, mod, burncut=argmts.burncut,bestevflag=True)
		else:
			ev,qres,numn,fn,prob = fit_model(good_data, Riter, good_data['coord'], agb_model_vel, kv, argmts.modset, mod, burncut=argmts.burncut, abridged=argmts.abridged) #,probs
		elist.append(ev)
		rlist.append(qres)
		numl.append(numn)
		print(ev)
		print(qres)
		print(numl)
		#print(prob)

		# restosave = []
		# for r in range(qres.shape[0]):
		# 	resrow = take_deltas(qres[r,:])
		# 	restosave.append(resrow)
		#np.savetxt(f'{spec_direct}/{fn}res.txt',np.array(restosave), '%s')

		# Save probabilities along with final sample
		if argmts.savesamp=='T':
			#print(len(prob),len(good_data))
			probcol = Column(prob, name='P_DISK')
			#print(probcol)
			good_data.add_columns([probcol])
			good_data.write(f'{spec_direct}/probs_{fn}{argmts.abandcut}.fits',overwrite=True)

		''' Make veldisp plots like Amanda - disk only
		disk = good_data[(probs>0.8)]
		X = np.zeros((len(disk), 2))
		X[:,0] = disk['RA_DEG'] #in degrees
		X[:,1] = disk['DEC_DEG'] #in degrees
		kdt = KDTree(X, metric='euclidean')
		distances,indices = kdt.query(X, k=101) #default: use the 32 nearest neighbors
		distances_arcsec = distances*3600

		rlist,dlist,vlist,plist = [],[],[],[]
		for i in range(len(disk)):
			r_curr=radinit
			numin = np.array([distances_arcsec[i]<r_curr][0]).sum()
			while numin<nummin:
				r_curr+=5
				if r_curr>radmax:
					break
				isin = [distances_arcsec[i]<r_curr][0]
				whichin = indices[i][isin]
				numin = np.array(isin).sum()

			if r_curr<radmax:
				rlist.append(r_curr)
				plist.append(disk['coord'][i])
				vels = disk['VCORR_STAT'][whichin]
				if argmts.startype=='RGB':
					grat=disk['GRATING'][whichin]
					velunc = np.array([np.sqrt(5.6**2+1.65**2) if grat[v]=='600ZD' else np.sqrt(2.2**2+1.85**2) for v in range(len(grat))])
				else:
					velunc = disk['VERR'][whichin]
				unnw = velunc**(-2)
				normw = unnw/(unnw.sum())
				meanv = np.average(vels,weights=normw)
				vlist.append(meanv)
				disp = np.sqrt((((vels-meanv)**2)*normw).sum())
				dlist.append(disp)

		fig,ax = plt.subplots(1,2,figsize=(16,8),sharex=True,sharey=True) # Plot losvel as function of on-sky position
		vc = ax[0].scatter([p.ra.deg for p in plist], [p.dec.deg for p in plist], c=vlist, s=10,vmin=-260,vmax=-100)
		dc = ax[1].scatter([p.ra.deg for p in plist], [p.dec.deg for p in plist], c=dlist, s=10,vmin=0,vmax=45)
		cbar1 = fig.colorbar(vc,ax=ax[0])
		cbar1.set_label('Velocity (km s$^{-1}$)')
		cbar2 = fig.colorbar(dc,ax=ax[1])
		cbar2.set_label('Velocity dispersion (km s$^{-1}$)')
		for a in ax:
			a.set_xlabel('RA (deg)')
			a.invert_xaxis()
		ax[0].set_ylabel('Dec (deg)')
		fig.tight_layout()
		fig.savefig(f'{spec_direct}/dyplots/{kv}_{mod}_disp.pdf')
		plt.close()
		#'''

		if argmts.psplit=='T':
			keyvals = [f'{kv}Nin', f'{kv}Nmid', f'{kv}Nout', f'{kv}Sin', f'{kv}Smid', f'{kv}Sout', f'{kv}in', f'{kv}mid', f'{kv}out']
			for k in keyvals:
				if 'in' in k:
					minr=0
					maxr=15
				elif 'mid' in k:
					minr=15
					maxr=30
				elif 'out' in k:
					minr=30
					maxr=100
				evn,qresn = fit_model(good_data, Riter, good_data['coord'], agb_model_vel, k, argmts.modset, mod, burncut=argmts.burncut, minr=minr, maxr=maxr, bestevflag=True)

	# print(elist)
	# print(rlist)
	# print(numl)

	for i in range(mt_list.shape[0]-1):
		if argmts.dmod:
			print(argmts.dmod)
		print(argmts.startype,argmts.zqcut,argmts.sncut)
		print(len(good_data))
		print(f'{mt_list[i]}/{mt_list[i+1]}',(elist[i][0]-elist[i+1][0])/np.log(10),)#(elist[i][0]-elist[i+1][0]))
