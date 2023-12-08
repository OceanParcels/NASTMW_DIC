import numpy as np
import xarray as xr
import sys
from glob import glob
import argparse
import tqdm


input_dir_phys = "/nethome/4302001/local_data/FREEGLORYS2V4/"
input_dir_bio = "/nethome/4302001/local_data/FREEBIORYS2V4/"
input_dir_edw = "/nethome/4302001/local_data/FREEGLORYS2V4_EDW/"

mesh_z = xr.open_dataset("/nethome/4302001/local_data/mesh/mesh_hgr_PSY4V2_deg_NA_GoM_subset.nc")
mesh_h = xr.open_dataset("/nethome/4302001/local_data/mesh/mesh_zgr_PSY4V2_deg_NA_GoM_subset.nc")
mesh = xr.merge([mesh_z, mesh_h])

vars_phys = ["2D", "T", "S", "U", "V", "W", "KZLN10"]
vars_bio = ["alk", "chl", "po4", "dic", "no3", "nppv", "o2", "si"]


def create_NA_climatology(month: int, var='T', filename=None, year_extent=(1995, 2017)):
    print(f"Climatology computation has started for {var}.")

    if var in vars_phys:
        fieldtype = "freeglorys2v4"
        input_dir = input_dir_phys
    elif var in vars_bio:
        fieldtype = "freebiorys2v4"
        input_dir = input_dir_bio
    elif var == "EDW":
        fieldtype = "freeglorys2v4"
        input_dir = input_dir_edw
    else:
        print(f"Variable {var} is not a valid variable.")
        sys.exit(1)

    if var == "EDW":
        append = "_s0.01-t17.0_20.5"
        date_indices = (-30, -26)
    else:
        append = ""
        date_indices = (-13, -9)

    month_files = np.array(sorted(glob(input_dir + f"{fieldtype}-NorthAtlanticGoM-daily_{var}_????-{month:02d}-??{append}.nc")))
    file_keep = [] # used to circumvent nasty glob pattern for years spanning 2 centuries
    for index, file in enumerate(month_files):
        if int(file[date_indices[0]:date_indices[1]]) >= year_extent[0] and int(file[date_indices[0]:date_indices[1]]) <= year_extent[1]:
            file_keep.append(index)
    month_file_select = month_files[file_keep]

    MOi_climatology = xr.open_dataset(month_file_select[0]).drop(['time_counter', 'time_counter_bounds', 'time_centered_bounds', 'time_centered']).copy()
    # the variables in MOi_climatology that are bools should be converted to floats
    if var == "EDW":
        for var in MOi_climatology.variables:
            if MOi_climatology[var].dtype == 'bool' or MOi_climatology[var].dtype == 'int64':
                MOi_climatology[var] = MOi_climatology[var].astype('float64')

    for file in tqdm.tqdm(month_file_select[1:], desc="Climatology computation"):
        # print(f"Trying to open {file}")
        new_file = xr.open_dataset(file).drop(['time_counter', 'time_counter_bounds', 'time_centered_bounds', 'time_centered'])
        if var == "EDW":
            for var in new_file.variables:
                if new_file[var].dtype == 'bool' or new_file[var].dtype == 'int64':
                    new_file[var] = new_file[var].astype('float64')
        MOi_climatology = MOi_climatology + new_file
        new_file.close()

    if not filename:
        filename = f"/nethome/4302001/local_data/climatology/{fieldtype}/{fieldtype}-NorthAtlanticGoM-climatology_{var}_{month:02d}_years{year_extent[0]}-{year_extent[1]}.nc"

    MOi_climatology = MOi_climatology/len(month_file_select)

    MOi_climatology.to_netcdf(filename)

    print("Climatology computation has finished.")


parser = argparse.ArgumentParser(description='Compute climatology for MOi.')
parser.add_argument('month', metavar='M', type=int, help='Month for which the climatology should be computed')
parser.add_argument('variable', metavar='V', type=str, help='Variable for which the climatology should be computed')
parser.add_argument('--filename', type=str, default=None, help='=Filename for the climatology')
parser.add_argument('--year_extent', type=int, nargs=2, default=[1995, 2017], help='Start and end years to compute the climatology for')

args = parser.parse_args()

create_NA_climatology(month=args.month, var=args.variable, filename=args.filename, year_extent=args.year_extent)
