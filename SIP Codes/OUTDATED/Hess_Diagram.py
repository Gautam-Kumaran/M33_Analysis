import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.colors import LogNorm
from funcs import *
from matplotlib.patches import Ellipse

# === Load and preprocess FITS data ===
with fits.open(r"C:\Github\M33_Analysis\SIP Codes\pandas_goodstar_bright_sr.fits") as hdul:
    data = hdul[1].data
    panda = pd.DataFrame(data.astype(data.dtype.newbyteorder('=')))

panda = panda.rename(columns={'RA': 'RA_DEG', 'Dec': 'DEC_DEG'})
panda = compute_deprojected_radius(panda)

distance_kpc = 859
center = SkyCoord(ra=23.4621 * u.deg, dec=30.6602 * u.deg)
stars = SkyCoord(ra=panda['RA_DEG'].values * u.deg, dec=panda['DEC_DEG'].values * u.deg)
separation_rad = center.separation(stars).radian
panda['projected_radius_kpc'] = separation_rad * distance_kpc

panda['color'] = panda['g'] - panda['i']
panda['mag'] = panda['i']

# === Define Area Functions ===
def elliptical_annulus_area_analytic(a_in_kpc, a_out_kpc, inc_deg):
    inc_rad = np.deg2rad(inc_deg)
    return np.pi * (a_out_kpc**2 - a_in_kpc**2) * np.cos(inc_rad)

def estimate_area_only(n_points=1_000_000, max_radius_kpc=60,
                       distance_kpc=859, PA_deg=22, inc_deg=52,
                       area_rings=[]):
    def kpc_to_arcmin(kpc):
        return np.rad2deg(kpc / distance_kpc) * 60

    max_radius_rad = max_radius_kpc / distance_kpc
    max_radius_arcmin = np.rad2deg(max_radius_rad) * 60
    xi = np.random.uniform(-max_radius_arcmin, max_radius_arcmin, n_points)
    eta = np.random.uniform(-max_radius_arcmin, max_radius_arcmin, n_points)

    PA_rad = np.deg2rad(PA_deg)
    inc_rad = np.deg2rad(inc_deg)
    xi_rot = xi * np.cos(PA_rad) + eta * np.sin(PA_rad)
    eta_rot = -xi * np.sin(PA_rad) + eta * np.cos(PA_rad)
    eta_deproj = eta_rot / np.cos(inc_rad)
    r_arcmin = np.sqrt(xi_rot**2 + eta_deproj**2)

    areas = {}
    for a_in_kpc, a_out_kpc in area_rings:
        r_in = kpc_to_arcmin(a_in_kpc)
        r_out = kpc_to_arcmin(a_out_kpc)
        mask = (r_arcmin >= r_in) & (r_arcmin < r_out)
        n_inside = np.sum(mask)
        box_area_arcmin2 = (2 * max_radius_arcmin)**2
        arcmin2_to_sr = (np.pi / (180 * 60))**2
        box_area_sr = box_area_arcmin2 * arcmin2_to_sr
        box_area_kpc2 = box_area_sr * (distance_kpc ** 2)
        area_estimate = (n_inside / n_points) * box_area_kpc2
        areas[f"{a_in_kpc}-{a_out_kpc}"] = area_estimate

    return areas

def make_hess(df, bins=(50, 50)):
    hist, xedges, yedges = np.histogram2d(df['color'], df['mag'], bins=bins, range=[[-1,3],[18,25]])
    return hist

# === Define bins ===
radial_bins_kpc = [(3, 10), (10, 13), (13, 15), (15, 16)] + [(r, r + 1) for r in range(16, 32)]
beyond_bin = (40, 50)
all_bins = radial_bins_kpc + [beyond_bin]

# === Estimate Areas Only (no plot)
area_results = estimate_area_only(
    n_points=100_000,
    max_radius_kpc=60,
    distance_kpc=859,
    PA_deg=22,
    inc_deg=52,
    area_rings=all_bins
)

# === Print Area Comparison Table with % Difference ===
print("\n=== Area Comparison for All Bins ===")
print(f"{'Bin (kpc)':<12} {'Analytic Area':>20} {'Monte Carlo Area':>25} {'% Diff':>10}")
print("-" * 75)
for (r_in, r_out) in all_bins:
    key = f"{r_in}-{r_out}"
    area_mc = area_results[key]
    area_an = elliptical_annulus_area_analytic(r_in, r_out, inc_deg=52)
    pct_diff = 100 * (area_mc - area_an) / area_an
    print(f"{key:<12} {area_an:>20.2f} {area_mc:>25.2f} {pct_diff:>10.2f}")

# === Subtract and Plot Hess Diagrams ===
beyond_area = area_results[f"{beyond_bin[0]}-{beyond_bin[1]}"]
beyond_df = panda[(panda['r_deproj_kpc'] >= beyond_bin[0]) & (panda['r_deproj_kpc'] < beyond_bin[1])]
hess_beyond = make_hess(beyond_df)

for r_in, r_out in radial_bins_kpc:
    label = f"{r_in}-{r_out}"
    bin_df = panda[(panda['r_deproj_kpc'] >= r_in) & (panda['r_deproj_kpc'] < r_out)]
    hess_m33 = make_hess(bin_df)

    scale = area_results[label] / beyond_area
    hess_scaled_beyond = scale * hess_beyond
    hess_subtracted = hess_m33 - hess_scaled_beyond

    vlim = np.percentile(np.abs(hess_m33),99)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Raw M33
    im0 = axes[0].imshow(hess_m33.T, origin='upper', aspect='auto',
                         extent=[-1, 3, 18, 25], cmap='viridis',vmin=0, vmax=vlim)
    axes[0].set_title(f"M33 Hess: {label}")
    axes[0].set_xlabel('g - i')
    axes[0].set_ylabel('i magnitude')
    fig.colorbar(im0, ax=axes[0])

    # Scaled Beyond
    im1 = axes[1].imshow(hess_scaled_beyond.T, origin='upper', aspect='auto',
                         extent=[-1, 3, 18, 25], cmap='viridis',vmin=0, vmax=vlim)
    axes[1].set_title("Scaled Beyond Hess")
    axes[1].set_xlabel('g - i')
    fig.colorbar(im1, ax=axes[1])

    # Subtracted Map — Linear scale only
    im2 = axes[2].imshow(hess_subtracted.T, origin='upper', aspect='auto',
                         extent=[-1, 3, 18, 25], cmap='RdBu_r',
                         vmin=-vlim, vmax=vlim)
    axes[2].set_title("Subtracted Hess")
    axes[2].set_xlabel('g - i')
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()
