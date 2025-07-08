import os
import glob
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich import box

# Initialize rich console
console = Console()

# Set the center of the galaxy (replace with your values)
center_ra = 23.4621   # example: in degrees
center_dec = 30.6602  # example: in degrees
center_coord = SkyCoord(ra=center_ra*u.deg, dec=center_dec*u.deg, frame='icrs')

# Path to search for FITS files
search_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Tables'))

# Find all FITS files with 'donedupes' in the name
fits_files = glob.glob(os.path.join(search_path, '*donedupes*.fits'))

if not fits_files:
    console.print("[red]No matching FITS files found.[/red]")
    exit()

console.print(f"[cyan]Found {len(fits_files)} FITS files to process.[/cyan]")

# Prepare lists to hold distances by population
pop_distances = {'young': [], 'int': [], 'old': []}

for fits_file in track(fits_files, description="Processing FITS files..."):
    with fits.open(fits_file) as hdul:
        for hdu in hdul:
            if not hasattr(hdu, 'data') or hdu.data is None:
                continue
            data = hdu.data
            columns = [col.name.lower() for col in hdu.columns]
            ra_col = next((col for col in columns if 'ra' in col), None)
            dec_col = next((col for col in columns if 'dec' in col), None)
            pop_col = next((col for col in columns if col in ['pop', 'population', 'age', 'type']), None)
            if ra_col and dec_col and pop_col:
                ra = data[ra_col]
                dec = data[dec_col]
                pop = data[pop_col]
                coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
                sep = coords.separation(center_coord)
                for d, p in zip(sep.arcmin, pop):
                    p_str = str(p).strip().lower()
                    if 'young' in p_str:
                        pop_distances['young'].append(d)
                    elif 'int' in p_str or 'inter' in p_str:
                        pop_distances['int'].append(d)
                    elif 'old' in p_str:
                        pop_distances['old'].append(d)
                break  # Only process the first table HDU

# Print summary table using Rich
summary_table = Table(title="Stellar Population Summary", box=box.SIMPLE)
summary_table.add_column("Population", justify="left", style="bold")
summary_table.add_column("Count", justify="right", style="green")

for pop in ['young', 'int', 'old']:
    summary_table.add_row(pop.capitalize(), str(len(pop_distances[pop])))

console.print(summary_table)

# Plotting
plt.figure(figsize=(10, 6))
bins = 30
for pop, color in zip(['young', 'int', 'old'], ['blue', 'green', 'red']):
    if pop_distances[pop]:
        plt.hist(pop_distances[pop], bins=bins, alpha=0.5, label=pop.capitalize(), color=color)
plt.xlabel('Distance from Center (arcmin)')
plt.ylabel('Number of Stars')
plt.title('Distribution of Stellar Populations by Distance from Center')
plt.legend()
plt.tight_layout()
plt.show()
