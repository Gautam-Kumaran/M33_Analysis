from astropy.io import fits

# Open the FITS file
with fits.open("Tables/m33_vels_stars_inc22B_donedupes.fits") as hdul:
    data = hdul[1].data
    columns = data.columns.names
    print("Columns in the FITS file:")
    for col in columns:
        print(col)
