from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Load the FITS file
fits_path = r'c:\Github\M33_Analysis\Tables\m33_vels_stars_inc22B_donedupes.fits'
with fits.open(fits_path) as hdul:
    data = hdul[1].data
    df = pd.DataFrame(data.astype(data.dtype.newbyteorder('=')))

print(f"Loaded {len(df)} rows from {fits_path}")
print(df.head())

# 1. Filter out any null positions
mask = df['RA_DEG'].notnull() & df['DEC_DEG'].notnull()
df = df[mask].copy()

# 2. Build SkyCoord objects
center = SkyCoord(ra=23.4621*u.deg, dec=30.6602*u.deg)  # M33 center
stars  = SkyCoord(ra=df['RA_DEG'].values*u.deg,
                  dec=df['DEC_DEG'].values*u.deg)

# 3. Compute sky-plane offsets in arcmin
offs = stars.transform_to(center.skyoffset_frame())
xi_arcmin  = offs.lon.degree * 60.0   
eta_arcmin = offs.lat.degree * 60.0   

# 4. Galaxy geometry (adjust as needed)
PA_deg  = 22    # position angle of M33’s major axis (deg east of north)
inc_deg = 52    # inclination of M33’s disk (deg)
PA  = np.deg2rad(PA_deg)
inc = np.deg2rad(inc_deg)

alpha = eta_arcmin * np.cos(PA) + xi_arcmin * np.sin(PA)   
beta  = -eta_arcmin * np.sin(PA) + xi_arcmin * np.cos(PA)  

# 5. “Un-tilt” the minor-axis coordinate
beta_prime = beta / np.cos(inc)

r_deproj_arcmin = np.sqrt(alpha**2 + beta_prime**2)

scale = (np.pi/180/60) * 794  # arcmin to kpc

df['r_deproj_arcmin'] = r_deproj_arcmin
df['r_deproj_kpc']    = r_deproj_arcmin * scale

# 6. Plot sky positions colored by deprojected radius
xi_deg = xi_arcmin / 60.0
eta_deg = eta_arcmin / 60.0
r_deproj_kpc = df['r_deproj_kpc'].values

fig, ax = plt.subplots(figsize=(6, 8))
sc = ax.scatter(xi_deg, eta_deg, c=r_deproj_kpc, cmap='plasma', s=10, edgecolor='none')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Deprojected Radius [kpc]")
ax.set_xlabel(r'$\xi$ [deg]')
ax.set_ylabel(r'$\eta$ [deg]')
ax.set_title("Sky Positions Colored by Deprojected Radius [kpc]")
ax.set_aspect('equal')
ax.invert_xaxis()  # Invert the x-axis
plt.tight_layout()
plt.show()


# 7. Assign age_flag column
# Default: unclassified

df['age_flag'] = 'unclassified'

# Confirmed Young: WCN_SEL == 1
mask_young_confirmed = (df['age_flag'] == 'unclassified') & (df['WCN_SEL'] == 1)
df.loc[mask_young_confirmed, 'age_flag'] = 'young_confirmed'

# CBN: CBN_SEL == 1
mask_cbn = (df['age_flag'] == 'unclassified') & (df['CBN_SEL'] == 1)
df.loc[mask_cbn, 'age_flag'] = 'CBN'

# Confirmed Young (YMS): YMS_SEL == 1
mask_yms = (df['age_flag'] == 'unclassified') & (df['YMS_SEL'] == 1)
df.loc[mask_yms, 'age_flag'] = 'young_confirmed'

# Young Uncertain (RHB): RHB_SEL == 1
mask_rhb = (df['age_flag'] == 'unclassified') & (df['RHB_SEL'] == 1)
df.loc[mask_rhb, 'age_flag'] = 'young_uncertain'

# Young Uncertain (OHB): OHB_SEL == 1
mask_ohb = (df['age_flag'] == 'unclassified') & (df['OHB_SEL'] == 1)
df.loc[mask_ohb, 'age_flag'] = 'young_uncertain'

# 8. Plot radial distributions for young_confirmed and young_uncertain

# Print number of young_confirmed and young_uncertain stars before RA/DEC cut
n_young_confirmed_before = (df['age_flag'] == 'young_confirmed').sum()
n_young_uncertain_before = (df['age_flag'] == 'young_uncertain').sum()
print(f"Young Confirmed (all): {n_young_confirmed_before}")
print(f"Young Uncertain (all): {n_young_uncertain_before}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Young Confirmed
r_young_confirmed = df.loc[df['age_flag'] == 'young_confirmed', 'r_deproj_kpc']
axes[0].hist(r_young_confirmed, bins=40, color='royalblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Deprojected Radius [kpc]')
axes[0].set_ylabel('Number of Stars')
axes[0].set_title('Young Confirmed')
axes[0].grid(True)

# Young Uncertain
r_young_uncertain = df.loc[df['age_flag'] == 'young_uncertain', 'r_deproj_kpc']
axes[1].hist(r_young_uncertain, bins=40, color='orange', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Deprojected Radius [kpc]')
axes[1].set_title('Young Uncertain')
axes[1].grid(True)

plt.suptitle('Radial Distributions: Young Confirmed vs Young Uncertain (All Stars)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 9. Plot radial distributions for young_confirmed and young_uncertain
# Only include stars with DEC <= 31.25 and RA >= 23.1
mask_region = (df['DEC_DEG'] <= 31.25) & (df['RA_DEG'] >= 23.1)

# Print number of young_confirmed and young_uncertain stars after RA/DEC cut
n_young_confirmed_after = ((df['age_flag'] == 'young_confirmed') & mask_region).sum()
n_young_uncertain_after = ((df['age_flag'] == 'young_uncertain') & mask_region).sum()
print(f"Young Confirmed (RA/DEC cut): {n_young_confirmed_after}")
print(f"Young Uncertain (RA/DEC cut): {n_young_uncertain_after}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Young Confirmed
r_young_confirmed = df.loc[(df['age_flag'] == 'young_confirmed') & mask_region, 'r_deproj_kpc']
axes[0].hist(r_young_confirmed, bins=40, color='royalblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Deprojected Radius [kpc]')
axes[0].set_ylabel('Number of Stars')
axes[0].set_title('Young Confirmed')
axes[0].grid(True)

# Young Uncertain
r_young_uncertain = df.loc[(df['age_flag'] == 'young_uncertain') & mask_region, 'r_deproj_kpc']
axes[1].hist(r_young_uncertain, bins=40, color='orange', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Deprojected Radius [kpc]')
axes[1].set_title('Young Uncertain')
axes[1].grid(True)

plt.suptitle('Radial Distributions: Young Confirmed vs Young Uncertain (RA ≥ 23.1, DEC ≤ 31.25)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
