# Young Stars Velocity Offset Calculation (RHB + OHB)

from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from scipy.interpolate import interp1d
from rich.console import Console
from rich.table import Table as RichTable

# --- 1. Load FITS File ---
fits_path = "c:/Github/M33_Analysis/Tables/m33_vels_stars_inc22B_donedupes.fits"
with fits.open(fits_path) as hdul:
    data = hdul[1].data
    df = pd.DataFrame(data.astype(data.dtype.newbyteorder('=')))

# --- 2. Filter out foreground stars ---
df = df[df['FG_SEL'] != 1]

# --- 3. Select RHB and OHB Stars ---
Young_DF = df[(df['YMS_SEL'] == 1) | (df['WCN_SEL'] == 1)].copy()
#Young_DF = df[(df['RHB_SEL'] == 1) | (df['OHB_SEL'] == 1)].copy()

# --- 4. Compute Deprojected Radius ---
center = SkyCoord(ra=23.4621*u.deg, dec=30.6602*u.deg)  # M33 center
stars  = SkyCoord(ra=Young_DF['RA_DEG'].values*u.deg,
                  dec=Young_DF['DEC_DEG'].values*u.deg)

offs = stars.transform_to(center.skyoffset_frame())
xi_arcmin  = offs.lon.degree * 60.0
eta_arcmin = offs.lat.degree * 60.0

PA_deg = 22
inc_deg = 52
PA  = np.deg2rad(PA_deg)
inc = np.deg2rad(inc_deg)

alpha = eta_arcmin * np.cos(PA) + xi_arcmin * np.sin(PA)
beta  = -eta_arcmin * np.sin(PA) + xi_arcmin * np.cos(PA)
beta_prime = beta / np.cos(inc)
r_deproj_arcmin = np.sqrt(alpha**2 + beta_prime**2)

scale = (np.pi / 180 / 60) * 794  # arcmin to kpc
Young_DF['r_deproj_kpc'] = r_deproj_arcmin * scale

# --- 5. Compute Model Velocity using Kam et al. (2017) ---
diskmodel = Table.read('C:/Github/M33_Analysis/SIP Codes/Kam2017_table4.dat', format='ascii',
                       names=['Radius_arcmin', 'Radius_kpc', 'Vrot_kms', 'Delta_Vrot', 'i_deg', 'PA_deg'])

m33coord = SkyCoord(ra='01h33m50.9s', dec='+30d39m36s', unit=(u.hourangle, u.deg))
v_sys = -180.0

def major_minor_transform(coords, pa, centercoords=m33coord):
    c_offset = coords.transform_to(centercoords.skyoffset_frame())
    xi, eta = c_offset.lon.degree, c_offset.lat.degree
    alpha = eta * np.cos(pa) + xi * np.sin(pa)
    beta =  - eta * np.sin(pa) + xi * np.cos(pa)
    return alpha, beta

def compute_model_los_velocity(coords):
    Rinit = np.sqrt((coords.ra.degree - m33coord.ra.degree)**2 +
                    (coords.dec.degree - m33coord.dec.degree)**2)
    R_arcmin = Rinit * 60.0

    f_pa = interp1d(diskmodel['Radius_arcmin'], diskmodel['PA_deg'], fill_value="extrapolate")
    f_incl = interp1d(diskmodel['Radius_arcmin'], diskmodel['i_deg'], fill_value="extrapolate")
    f_vrot = interp1d(diskmodel['Radius_arcmin'], diskmodel['Vrot_kms'], fill_value="extrapolate")

    pa = f_pa(R_arcmin) * u.deg
    incl = f_incl(R_arcmin) * u.deg
    vrot = f_vrot(R_arcmin)

    alpha, beta = major_minor_transform(coords, pa)
    phi = np.arctan2(beta / np.cos(incl), alpha)

    vlos = v_sys + vrot * np.sin(incl) * np.cos(phi)
    return vlos

# Apply model
coords = SkyCoord(ra=Young_DF['RA_DEG'].values*u.deg,
                  dec=Young_DF['DEC_DEG'].values*u.deg)
Young_DF['V_model'] = compute_model_los_velocity(coords)

# --- 6. Velocity Offset ---
Young_DF['V_offset'] = Young_DF['VCORR_STAT'] - Young_DF['V_model']

# --- 7. Summary with Rich ---
console = Console()
summary_table = RichTable(title="Velocity Offset Summary (RHB + OHB)")

summary_table.add_column("Metric", justify="left")
summary_table.add_column("Value", justify="right")
summary_table.add_row("Total Stars", str(len(Young_DF)))
summary_table.add_row("Mean Offset", f"{Young_DF['V_offset'].mean():.2f} km/s")
summary_table.add_row("Std Dev", f"{Young_DF['V_offset'].std():.2f} km/s")
summary_table.add_row("Min Offset", f"{Young_DF['V_offset'].min():.2f}")
summary_table.add_row("Max Offset", f"{Young_DF['V_offset'].max():.2f}")
console.print(summary_table)

# --- 8. Histogram (clipped) ---
plt.figure(figsize=(8, 6))
plt.hist(Young_DF['V_offset'].clip(-300, 300), bins=40, edgecolor='black', alpha=0.75, density=True)
plt.xlabel('Velocity Offset (VCORR_STAT - V_model) [km/s]')
plt.ylabel('Probability Density')
plt.title('Histogram of Velocity Offsets for YMS & WCN Stars')
plt.tight_layout()
plt.show()

# --- 9. Print Outliers ---
outliers = Young_DF[np.abs(Young_DF['V_offset']) > 1000]
print(f"⚠️ Outliers (|V_offset| > 1000 km/s): {len(outliers)}")
print(outliers[['RA_DEG', 'DEC_DEG', 'VCORR_STAT', 'V_model', 'V_offset']])