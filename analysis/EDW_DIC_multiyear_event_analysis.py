
import event_identification
import preprocess_timeseries
from importlib import reload
import numpy as np
import xarray as xr

import sys
import pickle
from collections import deque
import tqdm


sys.path.append('/nethome/4302001/tracer_backtracking/tools')

traj_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectories/"
# event_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_events_timescales_yearly/"
event_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_events_timescales_yearly_sensitivity/"
mask_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectory_masks/"
# timescale_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_multiyear_timescales/"
timescale_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_multiyear_timescales_sensitivity/"


experiments = [
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
    for rollwindow in [1, 6, 10, 20]:
        print(f"🕵️  Now working on experiment {experiment}, rollwindow = {rollwindow}")
        binned_events_per_year = {}
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
                                                                    pickle_suffix=f"_events_rw{rollwindow}.pkl",
                                                                    pickle_dir=event_dir + f"filter_{rollwindow}day/")
            mask_ds = xr.open_dataset(mask_dir + f"EDW_wfluxes_B_{year}-09-01_1095d_dt{sign}90_odt24_masks.nc")
            event_identification.check_mask_ds_attrs(events.ds, mask_ds)
            event_identification.check_events_mask_trajs(events, mask_ds)

            true_template = (events.ds.DIC*0+1).fillna(1).astype(bool)
            start_in_edw = true_template * mask_ds.start_in_edw

            if experiment == "return_to_edw_1y":
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

            # Note that we strictly don't need the 'mode' argument here anymore, as the total masks are already true for the
            # entire trajectory, given the trajectory meets the right requirement. Instead, we select the last index based on the experiment duration.
            filter = event_identification.multi_event_filter_from_ds(events.event_dict, total_mask,
                                                                    last_index=last_index, last_index_inclusion_mode="any", mode="all")

            print(f"Experiment: {experiment}. Year: {year}. The number of particles considered is:")
            print(f"{int(np.sum(total_mask.isel(obs=0)))}/{int(np.sum(mask_ds.start_in_edw))} ({np.sum(total_mask.isel(obs=0))/np.sum(mask_ds.start_in_edw)*100:.2f}%)")
            binned_events_per_year[year] = event_identification.bin_aggregate_events(events.aggregate_event_dict,
                                                                                    filter,
                                                                                    vars_to_analyze=[
                                                                                        "cs_DIC_total", "cs_DIC_bio_soft", "cs_DIC_bio_carbonate", "cs_DIC_diff"],
                                                                                    normalize=True)

        with open(timescale_dir + f"EDW_wfluxes_B_1095d_dt{sign}90_odt24_{experiment}_binned_events_per_year_rw{rollwindow}.pkl", "wb") as f:
            pickle.dump(binned_events_per_year, f)

        print(f"💾  Done with experiment {experiment}. Pickle has been saved.")
