import numpy as np
import xarray as xr

import pandas as pd

from glob import glob
from pathlib import Path
from importlib import reload
import tqdm

import xgcm
import xnemogcm as xn

data_dir_bio = "/nethome/4302001/local_data/FREEBIORYS2V4/"
data_dir_phy = "/nethome/4302001/local_data/FREEGLORYS2V4/"
data_dir_mesh = "/nethome/4302001/local_data/mesh/"

output_dir_fluxes = "/nethome/4302001/local_data/FREEBIORYS2V4_fluxes/"

ds_mask_hgr = xr.open_dataset(data_dir_mesh + "mask_PSY4V2_deg_NA_GoM_subset.nc")
ds_mesh_hgr = xr.open_dataset(data_dir_mesh + "mesh_hgr_PSY4V2_deg_NA_GoM_subset.nc")
ds_mesh_zgr = xr.open_dataset(data_dir_mesh + "mesh_zgr_PSY4V2_deg_NA_GoM_subset.nc")

aux_files = list(sorted(glob(data_dir_mesh + "*NA_GoM_subset.nc")))

domcfg = xn.open_domain_cfg(files=aux_files)

date_first = pd.Timestamp("1992-01-01")
date_last = pd.Timestamp("2019-12-16")
interval_days = (date_last - date_first).components.days
dates = [date_first + pd.Timedelta(i, 'D') for i in range(interval_days)]

def comp_diffv(grid, ds, tracer, unit='d'):
    if unit == 's':
        multip = 1
    elif unit == 'd':
        multip = 24 * 60 * 60
    return (grid.diff(10**(ds.vokzln10) * grid.diff(ds[tracer], axis='Z') / ds.e3w_1d, 'Z') / ds.e3t_1d) * multip


for i, date in tqdm.tqdm(enumerate(dates), total=len(dates)):
    ds_template = xr.open_dataset(f'{data_dir_bio}freebiorys2v4-NorthAtlanticGoM-daily_alk_{str(date)[:10]}.nc')

    ds_inst = xn.process_nemo(domcfg=domcfg, positions=[
    (xr.open_dataset(f'{data_dir_bio}freebiorys2v4-NorthAtlanticGoM-daily_alk_{str(date)[:10]}.nc'), "T"),
    (xr.open_dataset(f'{data_dir_bio}freebiorys2v4-NorthAtlanticGoM-daily_dic_{str(date)[:10]}.nc'), "T"),
    (xr.open_dataset(f'{data_dir_bio}freebiorys2v4-NorthAtlanticGoM-daily_no3_{str(date)[:10]}.nc'), "T"),
    (xr.open_dataset(f'{data_dir_bio}freebiorys2v4-NorthAtlanticGoM-daily_chl_{str(date)[:10]}.nc'), "T"),
    (xr.open_dataset(f'{data_dir_bio}freebiorys2v4-NorthAtlanticGoM-daily_o2_{str(date)[:10]}.nc'), "T"),
    (xr.open_dataset(f'{data_dir_bio}freebiorys2v4-NorthAtlanticGoM-daily_po4_{str(date)[:10]}.nc'), "T"),
    (xr.open_dataset(f'{data_dir_phy}freeglorys2v4-NorthAtlanticGoM-daily_KZLN10_{str(date)[:10]}.nc').drop(['x', 'y']), "W"),    
        ]#, suppress_t_bounds=True
        )
    ds_inst = xn._merge_nemo_and_domain_cfg(nemo_ds=ds_inst, domcfg=domcfg.drop(['x', 'y'])).expand_dims(dim='time').drop('time_centered')

    if i == 0:
        grid = xgcm.Grid(ds_inst, metrics=xn.get_metrics(ds_inst), periodic=False)

    for tracer in [
                    'alk',
                    'chl', 
                    'dic', 
                    'no3', 
                    'o2', 
                    'po4'
                    ]:
        diff_flux = comp_diffv(grid, ds_inst, tracer=tracer, unit='d').rename(f"flux_difv_{tracer}")
        # diff_flux
        ds_export = ds_template.copy()
        ds_export[f"{tracer}_difv_flux"] = xr.DataArray(diff_flux.drop('t').isel(time=0).data.compute(), 
                                                    dims = ["deptht", "y", "x"]).astype('float32')
        ds_export[f"{tracer}_difv_flux"].attrs = {'standard_name': f'{tracer}_flux_from_vertical_diffusion',
                                                'long_name': f'{tracer} flux from vertical diffusion',
                                                'units': 'mmol m-3 day-1'}
        ds_export = ds_export.drop('alk')
        diff_flux.close()
        ds_export.to_netcdf(f"{output_dir_fluxes}freebiorys2v4-NorthAtlanticGoM-daily_flux_difv_{tracer}_{str(date)[:10]}.nc")
        ds_export.close()

    ds_template.close()

