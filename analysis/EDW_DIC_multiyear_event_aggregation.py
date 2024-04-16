
import event_identification
import preprocess_timeseries
from importlib import reload
import numpy as np
import xarray as xr

import sys
import pickle
from collections import deque
import tqdm


sys.path.append('/nethome/4302001/NASTMW_DIC/tools')

traj_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectories/"
# event_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_events_timescales_yearly/"
event_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_events_timescales_yearly/" #_sensitivity/"
mask_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectory_masks/"
# timescale_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_multiyear_timescales/"
aggregated_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_events_aggregated/"

experiments = [
    "start_in_edw_any",
    "backw_start_in_edw_any"
    "stay_in_edw_1y",
    "reached_mixing_back_in_edw",
    "densification_001_1y_after_2y",
    "subduction_after_1y",
    # above represent the most important ones
    "reached_mixing",
    "densification_005_1y_after_2y",
    "densification_005_1y_after_3y",
    "densification_001_1y_after_3y",
    "densification_0_1y_after_2y",
    "densification_0_1y_after_3y",
    "reached_mixed",
    "reached_mixed_back_in_edw",
    "reached_mixed_not_in_edw",
    "return_to_edw_1y",
    "reached_mixing_not_in_edw",
    "subduction_after_2y",
    "subduction_after_3y"
]


for experiment in experiments:
    for rollwindow in [10]: #[1, 6, 10, 20]:
        print(f"🕵️  Now working on experiment {experiment}, rollwindow = {rollwindow}")
        aggregated_events_per_year = {}
        for year in range(1995, 2016):
            print(f"🗓️ Year: {year}")

            if "subduction" in experiment:
                sign = "-"
            else:
                sign = ""

            events = event_identification.open_from_pickle_and_link(f"EDW_wfluxes_B_{year}-09-01_1095d_dt{sign}90_odt24",
                                                                    preprocess={"fluxes": False,
                                                                                "sequestration": False,
                                                                                "mask": False,
                                                                                "in_edw": False
                                                                                },
                                                                    pickle_suffix=f"_events.pkl", #"_rw{rollwindow}.pkl",
                                                                    pickle_dir=event_dir) # + f"filter_{rollwindow}day/")
            mask_ds = xr.open_dataset(mask_dir + f"EDW_wfluxes_B_{year}-09-01_1095d_dt{sign}90_odt24_masks.nc")
            event_identification.check_mask_ds_attrs(events.ds, mask_ds)
            event_identification.check_events_mask_trajs(events, mask_ds)

            start_in_edw = mask_ds.start_in_edw

            if experiment == "start_in_edw_any":
                total_mask = start_in_edw
            elif experiment == "return_to_edw_1y":
                total_mask = mask_ds.forw_persistent_mask * start_in_edw
            elif experiment == "stay_in_edw_1y":
                total_mask = mask_ds.forw_persistent_mask_full * start_in_edw
            elif experiment == "densification_005_1y_after_2y":
                total_mask = mask_ds.forw_densification_mask_005_2y_yearpersist * start_in_edw
            elif experiment == "densification_005_1y_after_3y":
                total_mask = mask_ds.forw_densification_mask_005_3y_yearpersist * start_in_edw
            elif experiment == "densification_001_1y_after_2y":
                total_mask = mask_ds.forw_densification_mask_001_2y_yearpersist * start_in_edw
            elif experiment == "densification_001_1y_after_3y":
                total_mask = mask_ds.forw_densification_mask_001_3y_yearpersist * start_in_edw
            elif experiment == "densification_0_1y_after_2y":
                total_mask = mask_ds.forw_densification_mask_0_2y_yearpersist * start_in_edw
            elif experiment == "densification_0_1y_after_3y":
                total_mask = mask_ds.forw_densification_mask_0_3y_yearpersist * start_in_edw
            elif experiment == "reached_mixed":
                total_mask = mask_ds.forw_been_in_mixed_layer * start_in_edw
            elif experiment == "reached_mixed_back_in_edw":
                total_mask = mask_ds.forw_been_in_mixed_layer * mask_ds.forw_persistent_mask * start_in_edw
            elif experiment == "reached_mixed_not_in_edw":
                total_mask = mask_ds.forw_been_in_mixed_layer * ~mask_ds.forw_persistent_mask * start_in_edw
            elif experiment == "reached_mixing":
                total_mask = mask_ds.forw_been_in_mixing_layer * start_in_edw
            elif experiment == "reached_mixing_back_in_edw":
                total_mask = mask_ds.forw_been_in_mixing_layer * mask_ds.forw_persistent_mask * start_in_edw
            elif experiment == "reached_mixing_not_in_edw":
                total_mask = mask_ds.forw_been_in_mixing_layer * ~mask_ds.forw_persistent_mask * start_in_edw
            elif experiment == "subduction_after_1y":
                total_mask = mask_ds.backw_subduction_mask_1y * start_in_edw
            elif experiment == "subduction_after_2y":
                total_mask = mask_ds.backw_subduction_mask_2y * start_in_edw
            elif experiment == "subduction_after_3y":
                total_mask = mask_ds.backw_subduction_mask_3y * start_in_edw
            else:
                raise ValueError("Experiment not recognized")

            last_index = 1095 if "after_3y" in experiment else 730 if "after_2y" in experiment else 365

            trajlist = total_mask.trajectory.where(total_mask).dropna("trajectory").astype(int).values

            print(f"Experiment: {experiment}. Year: {year}. The number of particles considered is:")
            print(f"{int(np.sum(total_mask))}/{int(np.sum(mask_ds.start_in_edw))} ({np.sum(total_mask)/np.sum(mask_ds.start_in_edw)*100:.2f}%)")
            aggregated_events_per_year[year] = event_identification.aggregate_events(events.event_dict, 
                                                                                     trajfilter=trajlist, 
                                                                                     provenance=True, 
                                                                                     indices=True, 
                                                                                     progress=False,
                                                                                     up_to_index=last_index)

        with open(aggregated_dir + f"EDW_wfluxes_B_1095d_dt{sign}90_odt24_{experiment}_aggregated_events_per_year_rw{rollwindow}.pkl", "wb") as f:
            pickle.dump(aggregated_events_per_year, f)

        print(f"💾  Done with experiment {experiment}. Pickle has been saved.")
