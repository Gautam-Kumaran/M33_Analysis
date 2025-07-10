import pandas as pd
from astropy.io import fits
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
from tqdm import tqdm
from astropy.table import Table
from scipy.interpolate import interp1d

def load_filtered_fits(filepath):
    """
    Load a FITS file into a pandas DataFrame and remove foreground stars (FG_SEL == 1).
    """
    with fits.open(filepath) as hdul:
        data = hdul[1].data.astype(hdul[1].data.dtype.newbyteorder('='))
        return pd.DataFrame(data).query("FG_SEL != 1").copy()

def report_sel_flag_combinations(df, sel_flags=['RGB_SEL', 'AGB_SEL', 'CBN_SEL', 'YMS_SEL', 'WCN_SEL', 'RHB_SEL', 'OHB_SEL', 'FG_SEL']):
    """
    Analyze and print the number of stars for each unique combination of SEL flags in the DataFrame.
    """
    combo_counts = Counter(
        df.apply(lambda row: tuple(sorted([flag for flag in sel_flags if row.get(flag, 0) == 1])), axis=1)
    )
    print("Number of stars for each SEL flag combination:")
    for combo, count in combo_counts.items():
        label = ', '.join(combo) if combo else 'None'
        print(f"{label}: {count}")

def classify_age_groups(df):
    """
    Assigns an 'age_group' column to the DataFrame based on SEL flags,
    then prints the number of stars in each age group.
    """
    df['age_group'] = 'unclassified'

    df.loc[df['WCN_SEL'] == 1, 'age_group'] = 'young'
    df.loc[df['CBN_SEL'] == 1, 'age_group'] = 'int'

    df.loc[
        (df['age_group'] == 'unclassified') & (df['RGB_SEL'] == 1),
        'age_group'
    ] = 'old'

    df.loc[
        (df['age_group'] == 'unclassified') & (df['AGB_SEL'] == 1),
        'age_group'
    ] = 'int'

    df.loc[
        (df['age_group'] == 'unclassified') & (df['YMS_SEL'] == 1),
        'age_group'
    ] = 'young'

    print("Number of stars in each age group:")
    print(df['age_group'].value_counts())
    return df

def plot_cmd_panels(df):
    """
    Plots three color-magnitude diagrams (CMDs) for different filter combinations,
    colored by stellar age group.
    """
    # Filter data by magnitude availability
    df1 = df[(df['g'].notnull()) & (df['i'].notnull())]
    df2 = df[(df['F475W0_ACS'].notnull()) & (df['F814W0_ACS'].notnull())]
    df3 = df[(df['F606W0_ACS'].notnull()) & (df['F814W0_ACS'].notnull())]

    # Set color and label mappings
    colors = {'young': 'blue', 'int': 'orange', 'old': 'red'}
    labels = {'young': 'young', 'int': 'intermediate', 'old': 'old'}

    # Plot CMDs
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    # Panel 1: g - i vs i
    for group in ['old', 'int', 'young']:
        subset = df1[df1['age_group'] == group]
        axes[0].scatter(subset['g'] - subset['i'], subset['i'],
                        color=colors[group], s=8, alpha=0.4, label=labels[group])
    axes[0].invert_yaxis()
    axes[0].set_xlim(-1, 5)
    axes[0].set_ylim(24, 16)
    axes[0].set_xlabel('g - i')
    axes[0].set_ylabel('i')

    # Panel 2: F475W0 - F814W0
    for group in ['old', 'int', 'young']:
        subset = df2[df2['age_group'] == group]
        axes[1].scatter(subset['F475W0_ACS'] - subset['F814W0_ACS'], subset['F814W0_ACS'],
                        color=colors[group], s=8, alpha=0.4)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-1, 5)
    axes[1].set_ylim(24, 16)
    axes[1].set_xlabel('F475W$_0$ - F814W$_0$')
    axes[1].set_title('CMD age groups')

    # Panel 3: F606W0 - F814W0
    for group in ['old', 'int', 'young']:
        subset = df3[df3['age_group'] == group]
        axes[2].scatter(subset['F606W0_ACS'] - subset['F814W0_ACS'], subset['F814W0_ACS'],
                        color=colors[group], s=8, alpha=0.4)
    axes[2].invert_yaxis()
    axes[2].set_xlim(-1, 3)
    axes[2].set_ylim(24, 16)
    axes[2].set_xlabel('F606W$_0$ - F814W$_0$')

    # Add unified legend
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
def plot_spatial_age_groups(df):
    """
    Plots RA vs. DEC of stars color-coded by age group on a black background.
    """
    # Define colors and labels
    colors = {'young': 'blue', 'int': 'orange', 'old': 'red'}
    labels = {'young': 'young', 'int': 'intermediate', 'old': 'old'}

    # Create figure with black background
    plt.figure(figsize=(8, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    # Scatter plot for each age group
    for group in ['young', 'int', 'old']:
        subset = df[df['age_group'] == group]
        plt.scatter(subset['RA_DEG'], subset['DEC_DEG'],
                    color=colors[group], s=3, label=labels[group], alpha=0.8)

    # Axis formatting
    plt.xlabel("RA (degrees)", color='white')
    plt.ylabel("DEC (degrees)", color='white')
    plt.tick_params(colors='white')
    ax.invert_xaxis()
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.title("Spatial Age Groups (All Stars)", color='white')
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord

def compute_deprojected_radius(df, center_ra=23.4621, center_dec=30.6602, PA_deg=22, inc_deg=52, distance_kpc=850):
    """
    Computes deprojected galactocentric radii for stars in M33.

    Parameters:
        df (DataFrame): Must contain RA_DEG and DEC_DEG.
        center_ra (float): RA of galaxy center in degrees.
        center_dec (float): Dec of galaxy center in degrees.
        PA_deg (float): Position angle in degrees.
        inc_deg (float): Inclination in degrees.
        distance_kpc (float): Distance to galaxy in kpc.

    Returns:
        DataFrame: Copy of input DataFrame with 'r_deproj_arcmin' and 'r_deproj_kpc' columns added.
    """
    df = df[df['RA_DEG'].notnull() & df['DEC_DEG'].notnull()].copy()

    center = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)
    stars = SkyCoord(ra=df['RA_DEG'].values * u.deg, dec=df['DEC_DEG'].values * u.deg)
    offs = stars.transform_to(center.skyoffset_frame())

    xi_arcmin = offs.lon.degree * 60.0
    eta_arcmin = offs.lat.degree * 60.0

    PA = np.deg2rad(PA_deg)
    inc = np.deg2rad(inc_deg)

    alpha = eta_arcmin * np.cos(PA) + xi_arcmin * np.sin(PA)
    beta = -eta_arcmin * np.sin(PA) + xi_arcmin * np.cos(PA)
    beta_prime = beta / np.cos(inc)

    r_deproj_arcmin = np.sqrt(alpha**2 + beta_prime**2)
    arcmin_to_kpc = (np.pi / 180 / 60) * distance_kpc

    df['r_deproj_arcmin'] = r_deproj_arcmin
    df['r_deproj_kpc'] = r_deproj_arcmin * arcmin_to_kpc

    return df

def plot_radial_distribution_by_age(df):
    """
    Plots deprojected radial distribution histograms split by age group.

    Parameters:
        df (DataFrame): Must contain 'r_deproj_kpc' and 'age_group' columns.
    """
    bins = np.arange(0, 25, 0.2)
    age_groups = ['young', 'int', 'old']
    colors = {'young': 'blue', 'int': 'yellow', 'old': 'red'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, group in zip(axes, age_groups):
        subset = df[df['age_group'] == group]
        ax.hist(subset['r_deproj_kpc'], bins=bins, histtype='stepfilled', color=colors[group], alpha=0.7)
        ax.set_title(f'{group.capitalize()} Stars')
        ax.set_xlabel('Deprojected Radius (kpc)')
        if ax is axes[0]:
            ax.set_ylabel('Number of Stars')
        ax.grid(True)

    plt.suptitle('Deprojected Radial Distribution by Age Group')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def match_triplets(df, match_radius_arcmin=1.0):
    """
    Matches young, intermediate, and old stars into spatial triplets within a given radius.

    Parameters:
        df (DataFrame): Must contain 'RA_DEG', 'DEC_DEG', and 'age_group'.
        match_radius_arcmin (float): Matching radius in arcminutes.

    Returns:
        DataFrame: Combined DataFrame of matched stars with a 'group' column.
    """
    young_df = df[df['age_group'] == 'young'].copy()
    int_df   = df[df['age_group'] == 'int'].copy()
    old_df   = df[df['age_group'] == 'old'].copy()

    young_coords = SkyCoord(ra=young_df['RA_DEG'].values * u.deg, dec=young_df['DEC_DEG'].values * u.deg)
    int_coords   = SkyCoord(ra=int_df['RA_DEG'].values * u.deg, dec=int_df['DEC_DEG'].values * u.deg)
    old_coords   = SkyCoord(ra=old_df['RA_DEG'].values * u.deg, dec=old_df['DEC_DEG'].values * u.deg)

    match_radius = match_radius_arcmin * u.arcmin
    used_int = set()
    used_old = set()
    triplets = []

    for i in tqdm(range(len(young_df)), desc="Matching triplets (young base)"):
        y_coord = young_coords[i]
        y_index = young_df.index[i]

        # Match intermediate stars
        _, int_idxs, sep2d_int, _ = search_around_sky(SkyCoord([y_coord]), int_coords, match_radius)
        viable_ints = [(j, s.arcminute) for j, s in zip(int_idxs, sep2d_int) if int_df.iloc[j].name not in used_int]
        if not viable_ints:
            continue
        best_int_idx, _ = min(viable_ints, key=lambda x: x[1])

        # Match old stars
        _, old_idxs, sep2d_old, _ = search_around_sky(SkyCoord([y_coord]), old_coords, match_radius)
        viable_olds = [(j, s.arcminute) for j, s in zip(old_idxs, sep2d_old) if old_df.iloc[j].name not in used_old]
        if not viable_olds:
            continue
        best_old_idx, _ = min(viable_olds, key=lambda x: x[1])

        # Record match
        triplets.append({
            'young_idx': y_index,
            'int_idx': int_df.iloc[best_int_idx].name,
            'old_idx': old_df.iloc[best_old_idx].name
        })

        used_int.add(int_df.iloc[best_int_idx].name)
        used_old.add(old_df.iloc[best_old_idx].name)

    triplets_df = pd.DataFrame(triplets)
    print(f"Matched triplets found: {len(triplets_df)}")

    triplet_full_data = pd.concat([
        df.loc[triplets_df['young_idx']].assign(group='young'),
        df.loc[triplets_df['int_idx']].assign(group='int'),
        df.loc[triplets_df['old_idx']].assign(group='old')
    ]).reset_index(drop=True)

    return triplet_full_data[triplet_full_data['RA_DEG'].notnull() & triplet_full_data['DEC_DEG'].notnull()]

def match_doubles(df, match_radius_arcmin=1.0):
    """
    Matches young and old stars into spatial pairs within a given radius.

    Parameters:
        df (DataFrame): Must contain 'RA_DEG', 'DEC_DEG', and 'age_group'.
        match_radius_arcmin (float): Matching radius in arcminutes.

    Returns:
        DataFrame: Combined DataFrame of matched stars with a 'group' column.
    """
    young_df = df[df['age_group'] == 'young'].copy()
    old_df   = df[df['age_group'] == 'old'].copy()

    young_coords = SkyCoord(ra=young_df['RA_DEG'].values * u.deg, dec=young_df['DEC_DEG'].values * u.deg)
    old_coords   = SkyCoord(ra=old_df['RA_DEG'].values * u.deg, dec=old_df['DEC_DEG'].values * u.deg)

    match_radius = match_radius_arcmin * u.arcmin
    used_old = set()
    pairs = []

    for i in tqdm(range(len(young_df)), desc="Matching doubles (young base)"):
        y_coord = young_coords[i]
        y_index = young_df.index[i]

        # Match old stars
        _, old_idxs, sep2d_old, _ = search_around_sky(SkyCoord([y_coord]), old_coords, match_radius)
        viable_olds = [(j, s.arcminute) for j, s in zip(old_idxs, sep2d_old) if old_df.iloc[j].name not in used_old]
        if not viable_olds:
            continue
        best_old_idx, _ = min(viable_olds, key=lambda x: x[1])

        # Record match
        pairs.append({
            'young_idx': y_index,
            'old_idx': old_df.iloc[best_old_idx].name
        })

        used_old.add(old_df.iloc[best_old_idx].name)

    pairs_df = pd.DataFrame(pairs)
    print(f"Matched young–old pairs found: {len(pairs_df)}")

    pair_full_data = pd.concat([
        df.loc[pairs_df['young_idx']].assign(group='young'),
        df.loc[pairs_df['old_idx']].assign(group='old')
    ]).reset_index(drop=True)

    return pair_full_data[pair_full_data['RA_DEG'].notnull() & pair_full_data['DEC_DEG'].notnull()]

def plot_spatial(df):
    """
    Plots RA vs DEC of matched stars on a black background, grouped by 'group' column.

    Parameters:
        df (DataFrame): Must contain 'RA_DEG', 'DEC_DEG', and 'group' columns.
                        Supports any combination of 'young', 'int', and 'old'.
    """
    colors = {'young': 'blue', 'int': 'orange', 'old': 'red'}
    labels = {'young': 'young', 'int': 'intermediate', 'old': 'old'}

    plt.figure(figsize=(8, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    groups_present = df['group'].unique()

    for group in ['young', 'int', 'old']:
        if group in groups_present:
            subset = df[df['group'] == group]
            plt.scatter(subset['RA_DEG'], subset['DEC_DEG'],
                        color=colors[group], s=5, label=labels[group], alpha=0.8)

    plt.xlabel("RA (degrees)", color='white')
    plt.ylabel("DEC (degrees)", color='white')
    plt.xlim(22.8, 24.2)
    plt.ylim(30, 31.75)
    plt.tick_params(colors='white')
    ax.invert_xaxis()
    plt.legend(loc='lower center', ncol=len(groups_present), bbox_to_anchor=(0.5, -0.05))
    plt.title("Spatial Distribution of Matched Stars", color='white')
    plt.tight_layout()
    plt.show()

M33_COORD = SkyCoord(ra='01h33m50.9s', dec='+30d39m36s', unit=(u.hourangle, u.deg))
V_SYS = -180.0  # Systemic velocity of M33 (km/s)


def load_disk_model(diskmodel_path):
    return Table.read(diskmodel_path, format='ascii',
                      names=['Radius_arcmin', 'Radius_kpc', 'Vrot_kms', 'Delta_Vrot', 'i_deg', 'PA_deg'])


def major_minor_transform(coords, pa):
    """
    Transform coordinates into major/minor axis frame given PA.
    """
    c_offset = coords.transform_to(M33_COORD.skyoffset_frame())
    xi, eta = c_offset.lon.degree, c_offset.lat.degree
    alpha = eta * np.cos(pa) + xi * np.sin(pa)
    beta = -eta * np.sin(pa) + xi * np.cos(pa)
    return alpha, beta


def compute_model_los_velocity(coords, diskmodel):
    """
    Compute model line-of-sight velocities using Kam et al. (2017) rotation curve.
    """
    Rinit = np.sqrt((coords.ra.degree - M33_COORD.ra.degree)**2 +
                    (coords.dec.degree - M33_COORD.dec.degree)**2)
    R_arcmin = Rinit * 60.0

    # Interpolate PA, inclination, and rotation speed
    f_pa = interp1d(diskmodel['Radius_arcmin'], diskmodel['PA_deg'], fill_value="extrapolate")
    f_incl = interp1d(diskmodel['Radius_arcmin'], diskmodel['i_deg'], fill_value="extrapolate")
    f_vrot = interp1d(diskmodel['Radius_arcmin'], diskmodel['Vrot_kms'], fill_value="extrapolate")

    pa = f_pa(R_arcmin) * u.deg
    incl = f_incl(R_arcmin) * u.deg
    vrot = f_vrot(R_arcmin)

    alpha, beta = major_minor_transform(coords, pa)
    phi = np.arctan2(beta / np.cos(incl), alpha)

    vlos = V_SYS + vrot * np.sin(incl) * np.cos(phi)
    return vlos

# Assumes these utility functions already exist
def load_disk_model(diskmodel_path):
    return Table.read(diskmodel_path, format='ascii',
                      names=['Radius_arcmin', 'Radius_kpc', 'Vrot_kms', 'Delta_Vrot', 'i_deg', 'PA_deg'])

def major_minor_transform(coords, pa):
    c_offset = coords.transform_to(M33_COORD.skyoffset_frame())
    xi, eta = c_offset.lon.degree, c_offset.lat.degree
    alpha = eta * np.cos(pa) + xi * np.sin(pa)
    beta = -eta * np.sin(pa) + xi * np.cos(pa)
    return alpha, beta

def compute_model_los_velocity(coords, diskmodel):
    Rinit = np.sqrt((coords.ra.degree - M33_COORD.ra.degree)**2 +
                    (coords.dec.degree - M33_COORD.dec.degree)**2)
    R_arcmin = Rinit * 60.0

    f_pa = interp1d(diskmodel['Radius_arcmin'], diskmodel['PA_deg'], fill_value="extrapolate")
    f_incl = interp1d(diskmodel['Radius_arcmin'], diskmodel['i_deg'], fill_value="extrapolate")
    f_vrot = interp1d(diskmodel['Radius_arcmin'], diskmodel['Vrot_kms'], fill_value="extrapolate")

    pa = f_pa(R_arcmin) * u.deg
    incl = f_incl(R_arcmin) * u.deg
    vrot = f_vrot(R_arcmin)

    alpha, beta = major_minor_transform(coords, pa)
    phi = np.arctan2(beta / np.cos(incl), alpha)

    vlos = V_SYS + vrot * np.sin(incl) * np.cos(phi)
    return vlos

# Global constants
M33_COORD = SkyCoord(ra='01h33m50.9s', dec='+30d39m36s', unit=(u.hourangle, u.deg))
V_SYS = -180.0  # Systemic velocity of M33

def generate_summary_with_model_vlos(df, diskmodel_path='./Kam2017_table4.dat'):
    """
    Computes model-predicted LOS velocities for matched stellar doubles or triplets,
    and creates a summary table including VCORR_STAT and velocity offset.

    Parameters:
        df (DataFrame): Contains 'RA_DEG', 'DEC_DEG', 'group', and 'VCORR_STAT'.
        diskmodel_path (str): Path to Kam et al. (2017) rotation curve data.

    Returns:
        DataFrame: Summary with RA, DEC, model_vlos, VCORR_STAT, and voffset for each group present.
    """
    diskmodel = load_disk_model(diskmodel_path)

    coords = SkyCoord(ra=df['RA_DEG'].values * u.deg,
                      dec=df['DEC_DEG'].values * u.deg)
    
    model_vlos = compute_model_los_velocity(coords, diskmodel)

    df = df.copy()
    df['model_vlos'] = model_vlos
    df['voffset'] = df['VCORR_STAT'] - df['model_vlos']

    groups_present = sorted(df['group'].unique(), key=['young', 'int', 'old'].index)
    summary = pd.DataFrame()

    for group in groups_present:
        group_df = df[df['group'] == group].reset_index(drop=True)
        prefix = f"{group}_"
        summary[f'{prefix}RA'] = group_df['RA_DEG']
        summary[f'{prefix}DEC'] = group_df['DEC_DEG']
        summary[f'{prefix}vlos'] = group_df['model_vlos']
        summary[f'{prefix}vcorr_stat'] = group_df['VCORR_STAT']
        summary[f'{prefix}voffset'] = group_df['voffset']

    return summary

def plot_voffset_hist_young(triplet_summary, bins=np.linspace(-200, 200, 50)):
    """
    Plots histogram of velocity offset (voffset) for young stars.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(triplet_summary['young_voffset'], bins=bins, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Velocity Offset (km/s)")
    plt.ylabel("Number of Stars")
    plt.title("Young Stars: VCORR_STAT - Model VLOS")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_voffset_hist_intermediate(triplet_summary, bins=np.linspace(-200, 200, 50)):
    """
    Plots histogram of velocity offset (voffset) for intermediate-age stars.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(triplet_summary['int_voffset'], bins=bins, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel("Velocity Offset (km/s)")
    plt.ylabel("Number of Stars")
    plt.title("Intermediate Stars: VCORR_STAT - Model VLOS")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_voffset_hist_old(triplet_summary, bins=np.linspace(-200, 200, 50)):
    """
    Plots histogram of velocity offset (voffset) for old stars.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(triplet_summary['old_voffset'], bins=bins, color='red', alpha=0.7, edgecolor='black')
    plt.xlabel("Velocity Offset (km/s)")
    plt.ylabel("Number of Stars")
    plt.title("Old Stars: VCORR_STAT - Model VLOS")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def assign_radial_thirds_equal_count(df, radius_col='r_deproj_kpc'):
    """
    Splits stars into 3 equal-count groups based on deprojected radius.
    - Inner third: closest 1/3
    - Middle third
    - Outer third: farthest 1/3

    Adds a new column 'radial_third' to the dataframe.

    Parameters:
    - df: pandas DataFrame with deprojected radii
    - radius_col: name of the column with radius values (default: 'r_deproj_kpc')

    Returns:
    - Modified DataFrame with 'radial_third' column
    """
    if radius_col not in df.columns:
        raise KeyError(f"Radius column '{radius_col}' not found in DataFrame")

    # Drop rows with NaN in radius column
    valid_df = df[df[radius_col].notnull()].copy()

    # Sort by radius
    sorted_df = valid_df.sort_values(by=radius_col).reset_index()

    # Split indices into 3 roughly equal groups
    splits = np.array_split(sorted_df.index, 3)

    # Create assignment list
    group_labels = ['inner', 'middle', 'outer']
    radial_third = pd.Series(index=sorted_df['index'], dtype='object')
    for group, idxs in zip(group_labels, splits):
        radial_third.loc[sorted_df.loc[idxs, 'index']] = group

    # Assign back to original dataframe
    df['radial_third'] = radial_third

    return df
    
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd
from funcs import load_disk_model, compute_model_los_velocity

def generate_radial_third_velocity_summaries(df,
                                             radius_col='r_deproj_kpc',
                                             diskmodel_path='./Kam2017_table4.dat'):
    """
    For each radial_third ('inner','middle','outer'), splits into age groups
    (young/int/old), computes model_vlos & voffset, and returns a nested dict:
      summaries[radial_third][age_group] = DataFrame with prefixed columns.
    """
    # 1) load disk model once
    diskmodel = load_disk_model(diskmodel_path)

    # 2) compute model_vlos & voffset on *all* stars
    coords = SkyCoord(ra=df['RA_DEG'].values * u.deg,
                      dec=df['DEC_DEG'].values * u.deg)
    work = df.copy()
    work['model_vlos'] = compute_model_los_velocity(coords, diskmodel)
    work['voffset']    = work['VCORR_STAT'] - work['model_vlos']

    summaries = {}
    for thr in ['inner','middle','outer']:
        sub = work[work['radial_third'] == thr]
        if sub.empty:
            print(f"No stars in radial third '{thr}'.")
            continue

        group_dfs = {}
        for gp in ['young','int','old']:
            gp_sub = sub[sub['age_group'] == gp]
            if gp_sub.empty:
                continue

            # pick only the five columns and rename them
            df_gp = gp_sub[['RA_DEG','DEC_DEG','model_vlos','VCORR_STAT','voffset']].copy()
            df_gp = df_gp.rename(columns={
                'RA_DEG':        f'{gp}_RA',
                'DEC_DEG':       f'{gp}_DEC',
                'model_vlos':    f'{gp}_vlos',
                'VCORR_STAT':    f'{gp}_vcorr_stat',
                'voffset':       f'{gp}_voffset',
            })
            group_dfs[gp] = df_gp.reset_index(drop=True)

        summaries[thr] = group_dfs

    return summaries

def plot_all_voffset_histograms(radial_summaries):
    """
    Loop over radial thirds and age groups to plot velocity-offset histograms
    using existing helpers, without adding new titles.

    Parameters:
    - radial_summaries: dict of dicts as returned by
      generate_radial_third_velocity_summaries,
      i.e. radial_summaries['inner']['young'], etc.
    """
    # Mapping age‐group key to plotting function
    plot_map = {
        'young': plot_voffset_hist_young,
        'int':   plot_voffset_hist_intermediate,
        'old':   plot_voffset_hist_old,
    }

    for thr in ['inner', 'middle', 'outer']:
        for age_key, plot_fn in plot_map.items():
            df_gp = radial_summaries.get(thr, {}).get(age_key)
            if df_gp is None or df_gp.empty:
                continue

            plot_fn(df_gp)

