import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

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

scale = (np.pi/180/60) * 850  # arcmin to kpc

df['r_deproj_arcmin'] = r_deproj_arcmin
df['r_deproj_kpc']    = r_deproj_arcmin * scale

# 6. Plot sky positions colored by deprojected radius
xi_deg = xi_arcmin / 60.0
eta_deg = eta_arcmin / 60.0
r_deproj_kpc = df['r_deproj_kpc'].values

fig, ax = plt.subplots(figsize=(8, 6))
sc = plt.scatter(xi_deg, eta_deg, c=r_deproj_kpc, cmap='gnuplot', s=2, alpha=0.7)
plt.colorbar(sc, label='Deprojected Radius [kpc]')
plt.gca().set_aspect('equal')
plt.xlabel(r'$\xi$ [deg]')
plt.ylabel(r'$\eta$ [deg]')
plt.title('Sky Positions Colored by Deprojected Radius [kpc]')
plt.gca().invert_xaxis()  # RA increases to the left
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()