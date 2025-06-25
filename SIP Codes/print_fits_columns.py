from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load FITS file ===
with fits.open("Tables/m33_vels_stars_inc22B_donedupes.fits") as hdul:
    data = hdul[1].data
    df = pd.DataFrame(data.astype(data.dtype.newbyteorder('=')))

# === Classify stars ===
df['age_group'] = "unclassified"
df.loc[(df['YMS_SEL'] == 1) | (df['RHB_SEL'] == 1) | (df['WCN_SEL'] == 1) | (df['OHB_SEL'] == 1), 'age_group'] = 'young'
df.loc[(df['AGB_SEL'] == 1) | (df['CBN_SEL'] == 1), 'age_group'] = 'int'
df.loc[df['RGB_SEL'] == 1, 'age_group'] = 'old'

# === Filter valid entries for each CMD panel ===
df1 = df[(df['g'].notnull()) & (df['i'].notnull())]
df2 = df[(df['F475W0_ACS'].notnull()) & (df['F814W0_ACS'].notnull())]
df3 = df[(df['F606W0_ACS'].notnull()) & (df['F814W0_ACS'].notnull())]

# === Setup color map ===
colors = {'young': 'blue', 'int': 'orange', 'old': 'red'}
labels = {'young': 'young', 'int': 'int', 'old': 'old'}

# === Plot ===
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

# Panel 1: g - i vs i
for group in ['young', 'int', 'old']:
    subset = df1[df1['age_group'] == group]
    axes[0].scatter(subset['g'] - subset['i'], subset['i'], color=colors[group], s=8, label=labels[group], alpha=0.7)
axes[0].invert_yaxis()
axes[0].set_xlabel('g - i')
axes[0].set_ylabel('i')

# Panel 2: F475W0 - F814W0 vs F814W0
for group in ['young', 'int', 'old']:
    subset = df2[df2['age_group'] == group]
    axes[1].scatter(subset['F475W0_ACS'] - subset['F814W0_ACS'], subset['F814W0_ACS'], color=colors[group], s=8, alpha=0.7)
axes[1].invert_yaxis()
axes[1].set_xlabel('F475W$_0$ - F814W$_0$')
axes[1].set_title('CMD age groups')

# Panel 3: F606W0 - F814W0 vs F814W0
for group in ['young', 'int', 'old']:
    subset = df3[df3['age_group'] == group]
    axes[2].scatter(subset['F606W0_ACS'] - subset['F814W0_ACS'], subset['F814W0_ACS'], color=colors[group], s=8, alpha=0.7)
axes[2].invert_yaxis()
axes[2].set_xlabel('F606W$_0$ - F814W$_0$')

# Add unified legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()