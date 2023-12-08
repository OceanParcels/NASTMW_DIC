from importlib import reload
import numpy as np
import xarray as xr
import pandas as pd

import json
import sys
import pickle

import tqdm


sys.path.append('/nethome/4302001/tracer_backtracking/tools')


data_trajs = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectories_postprocessed/"
data_masks = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectory_masks/"

dummy_ds = xr.open_dataset(data_trajs + "EDW_wfluxes_B_2004-09-01_1095d_dt90_odt24.nc")
mask_ds = xr.open_dataset(data_masks + "EDW_wfluxes_B_2004-09-01_1095d_dt90_odt24_masks.nc")

years = np.arange(1995, 2016)
# years = np.arange(1995, 1996)

flux_types = ["DDIC", "DDIC_bio_soft", "DDIC_bio_carbonate", "DDIC_diff", "DDIC_residual"]

mathsign = {"": 1, "-": -1}

results = {}
for year in years:
    results[year] = {}


for year in tqdm.tqdm(years, desc="years"):
    for sign in ["", "-"]:
        ds_trajectory = xr.open_dataset(data_trajs + f"EDW_wfluxes_B_{year}-09-01_1095d_dt{sign}90_odt24.nc")
        mask_ds = xr.open_dataset(data_masks + f"EDW_wfluxes_B_{year}-09-01_1095d_dt{sign}90_odt24_masks.nc")

        t0 = ds_trajectory.time[0].values

        assert (ds_trajectory.trajectory == mask_ds.trajectory).all()

        start_in_edw = mask_ds.start_in_edw

        if sign == "":
            mask_dict = {
                "return_to_edw_1y": mask_ds.forw_persistent_mask * start_in_edw,
                "stay_in_edw_1y": mask_ds.forw_persistent_mask_full * start_in_edw,
                "densification_005_1y_after_2y": mask_ds.forw_densification_mask_005_2y_yearpersist * start_in_edw,
                "densification_005_1y_after_3y": mask_ds.forw_densification_mask_005_3y_yearpersist * start_in_edw,
                "densification_001_1y_after_2y": mask_ds.forw_densification_mask_001_2y_yearpersist * start_in_edw,
                "densification_001_1y_after_3y": mask_ds.forw_densification_mask_001_3y_yearpersist * start_in_edw,
                "densification_0_1y_after_2y": mask_ds.forw_densification_mask_0_2y_yearpersist * start_in_edw,
                "densification_0_1y_after_3y": mask_ds.forw_densification_mask_0_3y_yearpersist * start_in_edw,
                "reached_mixed": mask_ds.forw_been_in_mixed_layer * start_in_edw,
                "reached_mixed_back_in_edw": mask_ds.forw_been_in_mixed_layer * mask_ds.forw_persistent_mask * start_in_edw,
                "reached_mixed_not_in_edw": mask_ds.forw_been_in_mixed_layer * ~mask_ds.forw_persistent_mask * start_in_edw,
                "reached_mixing": mask_ds.forw_been_in_mixing_layer * start_in_edw,
                "reached_mixing_back_in_edw": mask_ds.forw_been_in_mixing_layer * mask_ds.forw_persistent_mask * start_in_edw,
                "reached_mixing_not_in_edw": mask_ds.forw_been_in_mixing_layer * ~mask_ds.forw_persistent_mask * start_in_edw,
            }
        elif sign == "-":
            mask_dict = {
                "subduction_after_1y": mask_ds.backw_subduction_mask_1y * start_in_edw,
                "subduction_after_2y": mask_ds.backw_subduction_mask_2y * start_in_edw,
                "subduction_after_3y": mask_ds.backw_subduction_mask_3y * start_in_edw,
            }

        sub_1y = ds_trajectory.isel(obs=slice(None, 366))
        sub_2y = ds_trajectory.isel(obs=slice(None, 731))
        sub_3y = ds_trajectory

        for experiment, mask in mask_dict.items():
            if "2y" in experiment:
                nmonths = 24
                ds_traj_sub = sub_2y
            elif "3y" in experiment:
                nmonths = 36
                ds_traj_sub = sub_3y
            else:
                nmonths = 12
                ds_traj_sub = sub_1y
            
            month_boundaries = [t0 + pd.DateOffset(months=month) for month in np.arange(nmonths + 1)]
            month_boundary_days = np.array([(month_boundary - t0).days for month_boundary in month_boundaries])
            month_boundary_days[-1] = -1
            month_lengths = [(month_boundaries[i+1] - month_boundaries[i]).days for i in range(nmonths)]
            # month_offsets = np.arange(nmonths)

            print(f"👉 Checkpoint: Experiment {experiment}, year {year}")
            if experiment not in results[year]:
                results[year][experiment] = {}
            for flux_type in flux_types:
                if sign == "":
                    results[year][experiment][flux_type] = {
                        "mean": [ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[month]:month_boundary_days[month+1]].sum(axis=1).mean() for month in np.arange(nmonths)],
                        "std": [ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[month]:month_boundary_days[month+1]].sum(axis=1).std() for month in np.arange(nmonths)],
                        "min": [ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[month]:month_boundary_days[month+1]].sum(axis=1).min() for month in np.arange(nmonths)],
                        "max": [ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[month]:month_boundary_days[month+1]].sum(axis=1).max() for month in np.arange(nmonths)],
                        "p01": [np.percentile(ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[month]:month_boundary_days[month+1]].sum(axis=1), 1) for month in np.arange(nmonths)],
                        "p99": [np.percentile(ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[month]:month_boundary_days[month+1]].sum(axis=1), 99) for month in np.arange(nmonths)],
                        "p05": [np.percentile(ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[month]:month_boundary_days[month+1]].sum(axis=1), 5) for month in np.arange(nmonths)],
                        "p95": [np.percentile(ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[month]:month_boundary_days[month+1]].sum(axis=1), 95) for month in np.arange(nmonths)],
                    }
                elif sign == "-":
                    results[year][experiment][flux_type] = {
                        "mean": [ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[::-1][month+1]:month_boundary_days[::-1][month]].sum(axis=1).mean() for month in np.arange(nmonths)],
                        "std": [ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[::-1][month+1]:month_boundary_days[::-1][month]].sum(axis=1).std() for month in np.arange(nmonths)],
                        "min": [ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[::-1][month+1]:month_boundary_days[::-1][month]].sum(axis=1).min() for month in np.arange(nmonths)],
                        "max": [ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[::-1][month+1]:month_boundary_days[::-1][month]].sum(axis=1).max() for month in np.arange(nmonths)],
                        "p01": [np.percentile(ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[::-1][month+1]:month_boundary_days[::-1][month]].sum(axis=1), 1) for month in np.arange(nmonths)],
                        "p99": [np.percentile(ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[::-1][month+1]:month_boundary_days[::-1][month]].sum(axis=1), 99) for month in np.arange(nmonths)],
                        "p05": [np.percentile(ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[::-1][month+1]:month_boundary_days[::-1][month]].sum(axis=1), 5) for month in np.arange(nmonths)],
                        "p95": [np.percentile(ds_traj_sub[flux_type].values[mask.values][:, month_boundary_days[::-1][month+1]:month_boundary_days[::-1][month]].sum(axis=1), 95) for month in np.arange(nmonths)],
                    }
                pass
            print(f"✅")
with open(f"output/EDW_DIC_monthly_change_{years[0]}-{years[-1]}.pickle", "wb") as f:
    pickle.dump(results, f)
