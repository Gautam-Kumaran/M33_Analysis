from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load FITS file ===
with fits.open("Tables/m33_vels_stars_inc22B_donedupes.fits") as hdul:
    data = hdul[1].data
    df = pd.DataFrame(data.astype(data.dtype.newbyteorder('=')))

# === Age classification ===
df['age_group'] = "unclassified"
df.loc[(df['YMS_SEL'] == 1) | (df['RHB_SEL'] == 1) | (df['WCN_SEL'] == 1) | (df['OHB_SEL'] == 1), 'age_group'] = 'young'
df.loc[(df['AGB_SEL'] == 1) | (df['CBN_SEL'] == 1), 'age_group'] = 'int'
df.loc[df['RGB_SEL'] == 1, 'age_group'] = 'old'

# === Filter good RA/DEC ===
df = df[df['RA_DEG'].notnull() & df['DEC_DEG'].notnull()]

# === Color map ===
colors = {'young': 'deepskyblue', 'int': 'gold', 'old': 'red'}
labels = {'young': 'young', 'int': 'int', 'old': 'old'}

# === Create spatial plot ===
plt.figure(figsize=(8, 10), facecolor='black')
ax = plt.gca()
ax.set_facecolor('black')

for group in ['young', 'int', 'old']:
    subset = df[df['age_group'] == group]
    if not subset.empty:
        plt.scatter(subset['RA_DEG'], subset['DEC_DEG'], color=colors[group], s=3, label=labels[group], alpha=0.8)

# === Axis formatting ===
plt.xlabel("RA (degrees)", color='white')
plt.ylabel("DEC (degrees)", color='white')
plt.tick_params(colors='white')
plt.gca().invert_xaxis()  # Flip RA axis so it increases to the right
plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
plt.title("Spatial Age Groups (All Stars)", color='white')
plt.tight_layout()
plt.show()
