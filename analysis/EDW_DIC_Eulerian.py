import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy as cart
import os
from glob import glob
import tqdm
import gsw

data_dir = "/nethome/4302001/local_data/"
output_dir = "/nethome/4302001/NASTMW_DIC/analysis/output/"


hgrid = xr.open_dataset(data_dir + "mesh/mesh_hgr_PSY4V2_deg_NA_subset.nc")
zgrid = xr.open_dataset(data_dir + "mesh/mesh_zgr_PSY4V2_deg_NA_subset.nc")

t0 = np.datetime64('1992-01-01')
tend = np.datetime64('2019-12-01')

t_length = tend-t0
timestamps = np.arange(t0, tend, np.timedelta64(1, 'D'), dtype='datetime64[D]')
ndays = t_length.astype('timedelta64[D]').astype(int)

nlevels = 75

EDW_files = sorted(glob(data_dir + "FREEGLORYS2V4_EDW/*EDW.nc"))
DIC_files = sorted(glob(data_dir + "FREEBIORYS2V4/*dic*.nc"))
T_files = sorted(glob(data_dir + "FREEGLORYS2V4/*_T_*.nc"))
S_files = sorted(glob(data_dir + "FREEGLORYS2V4/*_S_*.nc"))

dummy_dic = xr.open_dataset(data_dir + f"FREEBIORYS2V4/freebiorys2v4-NorthAtlanticGoM-daily_dic_{str(t0 + np.timedelta64(0, 'D'))}.nc")
dummy_edw = xr.open_dataset(
    data_dir + f"FREEGLORYS2V4_EDW/freeglorys2v4-NorthAtlanticGoM-daily_EDW_{str(t0 + np.timedelta64(0, 'D'))}_s0.01-t17.0_20.5.nc")
dummy_temp = xr.open_dataset(
    data_dir + f"FREEGLORYS2V4/freeglorys2v4-NorthAtlanticGoM-daily_T_{str(t0 + np.timedelta64(0, 'D'))}.nc").isel(x=slice(70, 290), y=slice(50, 250))
dummy_salt = xr.open_dataset(
    data_dir + f"FREEGLORYS2V4/freeglorys2v4-NorthAtlanticGoM-daily_S_{str(t0 + np.timedelta64(0, 'D'))}.nc").isel(x=slice(70, 290), y=slice(50, 250))

volume = (hgrid.e1t * hgrid.e2t * zgrid.e3t_1d).rename({"z": "deptht"}).assign_coords(deptht=dummy_dic.deptht)
areas = (hgrid.e1t * hgrid.e2t)

def compute_DIC_record():
    DIC_record = {
        "gravimetric": np.zeros((ndays, nlevels)) * np.nan,
        "volumetric": np.zeros((ndays, nlevels)) * np.nan,
        "carbon_mass": np.zeros(ndays) * np.nan,
        "gravimetric_incl_thickness": np.zeros((ndays, nlevels)) * np.nan,
        "volumetric_incl_thickness": np.zeros((ndays, nlevels)) * np.nan,
        "carbon_mass_incl_thickness": np.zeros(ndays) * np.nan,
        "gravimetric_incl_thickness_blobs": np.zeros((ndays, nlevels)) * np.nan,
        "volumetric_incl_thickness_blobs": np.zeros((ndays, nlevels)) * np.nan,
        "carbon_mass_incl_thickness_blobs": np.zeros(ndays) * np.nan,
        "volumetric_avg": np.zeros(ndays) * np.nan,
        "volumetric_avg_incl_thickness": np.zeros(ndays) * np.nan,
        "volumetric_avg_incl_thickness_blobs": np.zeros(ndays) * np.nan,
        "EDW_volume": np.zeros(ndays) * np.nan,
        "EDW_volume_incl_thickness": np.zeros(ndays) * np.nan,
        "EDW_volume_incl_thickness_blobs": np.zeros(ndays) * np.nan,
        "DIC_station_1" : np.zeros((ndays, nlevels)) * np.nan,
        "DIC_station_2" : np.zeros((ndays, nlevels)) * np.nan,
        "S_station_1" : np.zeros((ndays, nlevels)) * np.nan,
        "S_station_2" : np.zeros((ndays, nlevels)) * np.nan,
        "n36_volumetric": np.zeros((ndays, nlevels)) * np.nan,
        "n36_volumetric_incl_thickness": np.zeros((ndays, nlevels)) * np.nan,
        "n36_volumetric_incl_thickness_blobs": np.zeros((ndays, nlevels)) * np.nan,
        "n36_volumetric_avg": np.zeros(ndays) * np.nan,
        "n36_volumetric_avg_incl_thickness": np.zeros(ndays) * np.nan,
        "n36_volumetric_avg_incl_thickness_blobs": np.zeros(ndays) * np.nan,
    }

    for day in tqdm.tqdm(range(ndays)):
        ds_dic = xr.open_dataset(data_dir + f"FREEBIORYS2V4/freebiorys2v4-NorthAtlanticGoM-daily_dic_{str(t0 + np.timedelta64(day, 'D'))}.nc")
        ds_edw = xr.open_dataset(
            data_dir + f"FREEGLORYS2V4_EDW/freeglorys2v4-NorthAtlanticGoM-daily_EDW_{str(t0 + np.timedelta64(day, 'D'))}_s0.01-t17.0_20.5.nc")
        ds_temp = xr.open_dataset(
            data_dir + f"FREEGLORYS2V4/freeglorys2v4-NorthAtlanticGoM-daily_T_{str(t0 + np.timedelta64(day, 'D'))}.nc").isel(x=slice(70, 290), y=slice(50, 250))
        ds_salt = xr.open_dataset(
            data_dir + f"FREEGLORYS2V4/freeglorys2v4-NorthAtlanticGoM-daily_S_{str(t0 + np.timedelta64(day, 'D'))}.nc").isel(x=slice(70, 290), y=slice(50, 250))

        criteria = {
            "" : ds_edw.EDW_criterion,
            "_incl_thickness" : ds_edw.EDW_criterion.where(ds_edw.EDW_total_thickness >= 50, False),
            "_incl_thickness_blobs" : ds_edw.EDW_criterion.where(
                        (ds_edw.EDW_part_of_biggest_blob + ds_edw.EDW_part_of_smaller_blob) * (ds_edw.EDW_total_thickness > 50), False)
        }

        pressure = gsw.p_from_z(-ds_temp.deptht, ds_temp.nav_lat)
        CT = gsw.CT_from_pt(ds_salt.vosaline, ds_temp.votemper)
        density = gsw.rho(ds_salt.vosaline, CT, pressure)

        for crit_type in ["", "_incl_thickness", "_incl_thickness_blobs"]:
            DIC_record[f'volumetric{crit_type}'][day, :] = (ds_dic.dic.isel(x=slice(70, 290), y=slice(50, 250)).where(
                criteria[crit_type] > 0)).weighted(areas).mean(dim=['x', 'y'], skipna=True).values
            
            DIC_record[f'volumetric_avg{crit_type}'][day] = float((ds_dic.dic.isel(x=slice(70, 290), y=slice(50, 250)).where(
                criteria[crit_type] > 0)).weighted(volume).mean(skipna=True))
            
            DIC_record[f'n36_volumetric{crit_type}'][day, :] = ((ds_dic.dic.isel(x=slice(70, 290), y=slice(50, 250)) / ds_salt.vosaline * 36).where(
                criteria[crit_type] > 0)).weighted(areas).mean(dim=['x', 'y'], skipna=True).values
            
            DIC_record[f'n36_volumetric_avg{crit_type}'][day] = float(((ds_dic.dic.isel(x=slice(70, 290), y=slice(50, 250)) / ds_salt.vosaline * 36).where(
                criteria[crit_type] > 0)).weighted(volume).mean(skipna=True))
            
            DIC_record[f'gravimetric{crit_type}'][day, :] = (ds_dic.dic.isel(x=slice(70, 290), y=slice(50, 250)).where(
                criteria[crit_type] > 0)/density*1000).weighted(areas).mean(dim=['x', 'y'], skipna=True).values

            DIC_record[f'carbon_mass{crit_type}'][day] = float((ds_dic.dic.isel(x=slice(70, 290), y=slice(50, 250)).where(
                criteria[crit_type] > 0) * volume).sum(skipna=True)/1000 * 12.011 / 10**12)
            
            DIC_record[f'EDW_volume{crit_type}'][day] = (ds_edw.EDW_criterion.where(criteria[crit_type]).astype(np.float64) * volume).sum(dim=["x", "y", "deptht"])

        DIC_record[f'DIC_station_1'][day] = ds_dic.dic.isel(x=140, y=137).values
        DIC_record[f'S_station_1'][day] = ds_salt.vosaline.isel(x=140 - 70, y=137 - 50).values
        DIC_record[f'DIC_station_2'][day] = ds_dic.dic.isel(x=199, y=136).values
        DIC_record[f'S_station_2'][day] = ds_salt.vosaline.isel(x=199 - 70, y=136 - 50).values
        # Salinities are already subsetted on the NASTMW region, so no need to subtract the offsets
        # x=slice(70, 290), y=slice(50, 250)

        ds_dic.close()
        ds_edw.close()
        ds_temp.close()
        ds_salt.close()
    return DIC_record



DIC_record = compute_DIC_record()

da_dict = dict()
for key in DIC_record.keys():
    if 'volumetric_avg' in key:
        da_dict["DIC_EDW_" + key] = xr.DataArray(DIC_record[key], dims=["time"], coords={"time": timestamps})
        da_dict["DIC_EDW_" + key].attrs = {"units": "µmol/L"}
    elif 'gravimetric' in key:
        da_dict["DIC_EDW_" + key] = xr.DataArray(DIC_record[key], dims=["time", "deptht"], coords={"time": timestamps, "deptht": dummy_dic.deptht})
        da_dict["DIC_EDW_" + key].attrs = {"units": "µmol/kg"}
    elif 'volumetric' in key:
        da_dict["DIC_EDW_" + key] = xr.DataArray(DIC_record[key], dims=["time", "deptht"], coords={"time": timestamps, "deptht": dummy_dic.deptht})
        da_dict["DIC_EDW_" + key].attrs = {"units": "µmol/L"}
    elif 'mass' in key:
        da_dict["DIC_EDW_" + key] = xr.DataArray(DIC_record[key], dims=["time"], coords={"time": timestamps})
        da_dict["DIC_EDW_" + key].attrs = {"units": "Tg C"}
    elif 'EDW_volume' in key:
        da_dict["DIC_EDW_" + key] = xr.DataArray(DIC_record[key], dims=["time"], coords={"time": timestamps})
        da_dict["DIC_EDW_" + key].attrs = {"units": "m³"}
    elif 'DIC_station' in key:
        da_dict[key] = xr.DataArray(DIC_record[key], dims=["time", "deptht"], coords={"time": timestamps, "deptht": dummy_dic.deptht})
        da_dict[key].attrs = {"units": "µmol/L"}
    elif 'S_station' in key:
        da_dict[key] = xr.DataArray(DIC_record[key], dims=["time", "deptht"], coords={"time": timestamps, "deptht": dummy_dic.deptht})
        da_dict[key].attrs = {"units": "psu"}

ds = xr.Dataset(da_dict)
cell_thickness = zgrid.e3t_1d.rename({'z':'deptht'})
cell_thickness['deptht'] = ds.deptht
cell_thickness.attrs = {"units": "m"}
ds["cell_thickness"] = cell_thickness

ds.to_netcdf(output_dir + "DIC_EDW_record.nc")

