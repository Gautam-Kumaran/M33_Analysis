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
from astropy.coordinates import SkyCoord
from Lfuncs import *

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm



M33_COORD = SkyCoord(ra='01h33m50.9s', dec='+30d39m36s', unit=(u.hourangle, u.deg))
V_SYS = -180.0  # Systemic velocity of M33 (km/s)

import pandas as pd
from astropy.io import fits

def load_filtered_fits(filepath, badindex_path="./BadIndex.csv"):
    """
    Load a FITS file into a pandas DataFrame, remove BadIndex stars
    (defined on the original indexing), then remove foreground stars.
    """

    # Load FITS
    with fits.open(filepath) as hdul:
        data = hdul[1].data.astype(hdul[1].data.dtype.newbyteorder("="))
        df = pd.DataFrame(data)

    # Drop BadIndex first (indices match original FITS rows)
    try:
        badindex = pd.read_csv(badindex_path)["BadIndex"].values
        df = df.drop(index=badindex, errors="ignore")
    except FileNotFoundError:
        print(f"Warning: {badindex_path} not found. Skipping BadIndex removal.")

    # Now remove foreground stars
    df = df.query("FG_SEL != 1").copy()

    return df

        
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

def remove_high_vcorr_stars(df, threshold=1000, verbose=True):
    """
    Identifies and optionally removes stars with VCORR_STAT above a given threshold.
    """
    high_vcorr = df[df['VCORR_STAT'] > threshold]
    
    if verbose and not high_vcorr.empty:
        print(f"Stars with VCORR_STAT > {threshold}:")
        print(high_vcorr[['RA_DEG', 'DEC_DEG', 'VCORR_STAT']])

    cleaned_df = df[df['VCORR_STAT'] <= threshold].copy()
    return cleaned_df


def classify_age_groups(df):
    """
    Assigns an 'age_group' column to the DataFrame based on SEL flags.
    OHB, RHB, and YMS stars are classified as 'young'.
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

    # Combine YMS, OHB, RHB into young
    df.loc[
        (df['age_group'] == 'unclassified') & (
            (df['YMS_SEL'] == 1) | (df['OHB_SEL'] == 1) | (df['RHB_SEL'] == 1)
        ),
        'age_group'
    ] = 'young'

    print("Number of stars in each age group:")
    print(df['age_group'].value_counts())
    return df

def plot_cmd_panels(df):
    """
    Plots three color-magnitude diagrams (CMDs) for different filter combinations,
    colored by stellar age group.
    Each panel has its own legend.
    """
    import matplotlib.pyplot as plt

    # Filter data by magnitude availability
    df1 = df[(df['g'].notnull()) & (df['i'].notnull())]
    df2 = df[(df['F475W0_ACS'].notnull()) & (df['F814W0_ACS'].notnull())]
    df3 = df[(df['F606W0_ACS'].notnull()) & (df['F814W0_ACS'].notnull())]

    # Set color and label mappings
    colors = {'young': 'blue', 'int': 'orange', 'old': 'red'}
    labels = {'young': 'Young', 'int': 'Intermediate', 'old': 'Old'}

    # Plot CMDs
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

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
    axes[0].legend(title='Age Group')

    # Panel 2: F475W0 - F814W0
    for group in ['old', 'int', 'young']:
        subset = df2[df2['age_group'] == group]
        axes[1].scatter(subset['F475W0_ACS'] - subset['F814W0_ACS'], subset['F814W0_ACS'],
                        color=colors[group], s=8, alpha=0.4, label=labels[group])
    axes[1].invert_yaxis()
    axes[1].set_xlim(-1, 5)
    axes[1].set_ylim(24, 16)
    axes[1].set_xlabel('F475W$_0$ - F814W$_0$')
    axes[1].set_title('CMD age groups')
    axes[1].legend(title='Age Group')

    # Panel 3: F606W0 - F814W0
    for group in ['old', 'int', 'young']:
        subset = df3[df3['age_group'] == group]
        axes[2].scatter(subset['F606W0_ACS'] - subset['F814W0_ACS'], subset['F814W0_ACS'],
                        color=colors[group], s=8, alpha=0.4, label=labels[group])
    axes[2].invert_yaxis()
    axes[2].set_xlim(-1, 3)
    axes[2].set_ylim(24, 16)
    axes[2].set_xlabel('F606W$_0$ - F814W$_0$')
    axes[2].legend(title='Age Group')

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
    for group in ['old', 'int', 'young']:
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



def _single_star_deproj(args):
    ra, dec, diskmodel, m33coord = args
    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    ang_dist_deg, _, _, _ = m33_tilted_ring_deproj_radius(sc, model=diskmodel)
    return ang_dist_deg * 60.  # arcmin

def compute_deprojected_radius_tilted_ring(
    df,
    diskmodel = './Kam2017_table4.dat',
    m33coord = None,
    distance_kpc=859
):
    """
    Computes deprojected galactocentric radii for stars in M33 using the full tilted ring model, in parallel,
    with tqdm progress bar.
    """
    from tqdm import tqdm
    from astropy.table import Table
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    #Load diskmodel
    diskmodel = load_disk_model(diskmodel)

    # --- Set default center coordinate if None ---
    if m33coord is None:
        m33coord = SkyCoord(ra=23.4621 * u.deg, dec=30.6602 * u.deg)

    # Remove rows with missing or out-of-bounds RA/DEC

    n = len(df)
    if n == 0:
        df['r_deproj_arcmin'] = []
        df['r_deproj_kpc'] = []
        return df

    # Build argument list for parallelization
    args_list = list(zip(df['RA_DEG'], df['DEC_DEG'],
                         [diskmodel]*n, [m33coord]*n))

    # Parallel computation with progress bar
    with ProcessPoolExecutor() as executor:
        r_deproj_arcmin = list(tqdm(
            executor.map(_single_star_deproj, args_list),
            total=n,
            desc="Deprojecting"
        ))

    # Convert to kpc
    arcmin_to_kpc = (np.pi / 180 / 60) * distance_kpc
    r_deproj_arcmin = np.array(r_deproj_arcmin)
    r_deproj_kpc = r_deproj_arcmin * arcmin_to_kpc

    # Store results
    df['r_deproj_arcmin'] = r_deproj_arcmin
    df['r_deproj_kpc'] = r_deproj_kpc

    return df




def plot_radial_distribution_by_age(df):
    """
    Plots deprojected radial distribution histograms split by age group.

    Parameters:
        df (DataFrame): Must contain 'r_deproj_kpc' and 'age_group' columns.
    """
    bins = np.arange(0, 35, 0.2)
    age_groups = ['young', 'int', 'old']
    colors = {'young': 'blue', 'int': 'orange', 'old': 'red'}

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
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

def plot_spatial(df, outfile="spatial_distribution_matched_stars.png", dpi=300):
    """
    Plots RA vs DEC of matched stars on a black background, grouped by 'group' column,
    and saves the figure to disk.

    Parameters:
        df (DataFrame): Must contain 'RA_DEG', 'DEC_DEG', and 'group' columns.
                        Supports any combination of 'young', 'int', and 'old'.
        outfile (str): Path/filename for the saved image.
        dpi (int): Resolution of the saved figure.
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
            plt.scatter(
                subset['RA_DEG'],
                subset['DEC_DEG'],
                color=colors[group],
                s=5,
                label=labels[group],
                alpha=0.8
            )

    plt.xlabel("RA (degrees)", color='white')
    plt.ylabel("DEC (degrees)", color='white')
    plt.xlim(22.8, 24.2)
    plt.ylim(30, 31.75)
    plt.tick_params(colors='white')
    ax.invert_xaxis()

    plt.legend(
        loc='lower center',
        ncol=len(groups_present),
        bbox_to_anchor=(0.5, -0.05)
    )

    plt.title("Spatial Distribution of Matched Stars", color='white')
    plt.tight_layout()

    # --- SAVE FIGURE ---
    plt.savefig(outfile, dpi=dpi, facecolor='black', bbox_inches='tight')

    plt.show()




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
    Adds a 'radial_third' column to the DataFrame.
    """
    if radius_col not in df.columns:
        raise KeyError(f"Radius column '{radius_col}' not found in DataFrame")
    valid_df = df[df[radius_col].notnull()].copy()
    sorted_df = valid_df.sort_values(by=radius_col).reset_index()
    splits = np.array_split(sorted_df.index, 3)
    group_labels = ['inner', 'middle', 'outer']
    radial_third = pd.Series(index=sorted_df['index'], dtype='object')
    for group, idxs in zip(group_labels, splits):
        radial_third.loc[sorted_df.loc[idxs, 'index']] = group
    df = df.copy()
    df['radial_third'] = radial_third
    return df

    

def generate_radial_third_agegroup_dfs(
    df, 
    radius_col='r_deproj_kpc',
    diskmodel_path='./Kam2017_table4.dat'
):
    diskmodel = load_disk_model(diskmodel_path)
    coords = SkyCoord(ra=df['RA_DEG'].values * u.deg,
                      dec=df['DEC_DEG'].values * u.deg)
    work = df.copy()
    work['model_vlos'] = compute_model_los_velocity(coords, diskmodel)
    work['voffset']    = work['VCORR_STAT'] - work['model_vlos']
    radial_thirds = ['inner', 'middle', 'outer']
    age_groups = ['young', 'int', 'old']
    out = {}
    for thr in radial_thirds:
        for age in age_groups:
            mask = (work['radial_third'] == thr) & (work['age_group'] == age)
            temp = work.loc[mask, ['RA_DEG', 'DEC_DEG', 'model_vlos', 'VCORR_STAT', 'voffset']].copy()
            temp.columns = [f'{age}_RA', f'{age}_DEC', f'{age}_vlos', f'{age}_vcorr_stat', f'{age}_voffset']
            temp = temp.reset_index(drop=True)
            key = f"{thr}_{age}"
            out[key] = temp
    return out


def plot_all_voffset_histograms(summary_dfs):
    """
    Loop over radial thirds and age groups to plot velocity-offset histograms
    using the triplet-style summary DataFrames.
    """
    plot_map = {
        'young': plot_voffset_hist_young,
        'int':   plot_voffset_hist_intermediate,
        'old':   plot_voffset_hist_old,
    }
    for thr, df in summary_dfs.items():
        if df is None or df.empty:
            continue
        for age, plot_fn in plot_map.items():
            voffset_col = f'{age}_voffset'
            if voffset_col in df.columns:
                plot_fn(df[[voffset_col]].dropna())


