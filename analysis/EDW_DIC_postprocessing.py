
import event_identification
import preprocess_timeseries
from importlib import reload
import numpy as np
import xarray as xr
import copy
import sys
import pickle
from collections import deque
import tqdm
import logging
import argparse
import os


# ———————–—— Set logging ————————————–——
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
# ———————————————————————————————————————

traj_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectories/"
out_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectories_postprocessed/"

def merge_dicts(d1, d2):
    """
    Merges two nested dictionaries d1 and d2 without modifying them.
    Returns a new dictionary.
    For overlapping keys, sub-dictionaries are properly merged.
    """
    merged = copy.deepcopy(d1)
    for key, value in d2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


encoding_dtypes = {
    'goes_out_of_bounds': {"dtype" : "bool"},  
    "temp_range_crossing" : {"dtype" : "bool"},
    "temp_range_entry_from_below" : {"dtype" : "bool"},
    "temp_range_exit_to_below" : {"dtype" : "bool"},
    "temp_range_entry_from_above" : {"dtype" : "bool"},
    "temp_range_exit_to_above" : {"dtype" : "bool"},
    "strat_range_entry" : {"dtype" : "bool"},
    "strat_range_exit" : {"dtype" : "bool"},
    "mixing_layer_entry" : {"dtype" : "bool"},
    "mixing_layer_exit" : {"dtype" : "bool"},
    "mixed_layer_entry" : {"dtype" : "bool"},
    "mixed_layer_exit" : {"dtype" : "bool"},
    "thickness_regime_entry" : {"dtype" : "bool"},
    "thickness_regime_exit" : {"dtype" : "bool"},
    "edw_entry" : {"dtype" : "bool"},
    "edw_exit" : {"dtype" : "bool"},
    "goes_out_of_bounds" : {"dtype" : "bool"},
    "EDW_part_of_outcropping_blob" : {"dtype" : "bool"},
    "EDW_part_of_biggest_blob" : {"dtype" : "bool"},
    "EDW_part_of_smaller_blob" : {"dtype" : "bool"},
    "EDW_outcropping_column_mask" : {"dtype" : "bool"},
    "EDW_Lagrangian" : {"dtype" : "bool"},
    "EDW_Eulerian" : {"dtype" : "bool"},
    "lat" : {"dtype" : "float32"},
    "lon" : {"dtype" : "float32"},
    "z" : {"dtype" : "float32"},
    "DDIC" : {"dtype" : "float32"},
    "rho" : {"dtype" : "float32"},
    "sigma0" : {"dtype" : "float32"},
    "last_sigma0" : {"dtype" : "float32"},
    "last_rho" : {"dtype" : "float32"},
    }

def main():
    parser = argparse.ArgumentParser(description="Postprocess (preprocess) a .nc or .zarr particleset to add the necessary variables for event identification.")
    parser.add_argument("name", help="name of the file in EDW_trajectory_dir.")
    parser.add_argument("--output_dir", "-o", help="Output directory", default=out_dir)
    parser.add_argument("--input_dir", "-d", help="Input directory", default=traj_dir)
    parser.add_argument("--compression", "-c", action="store_true", help="Compress the output file.")
    args = parser.parse_args()

    # check file extension of input file
    if args.name.endswith(".nc"):
        filetype = "netcdf"
    elif args.name.endswith(".zarr"):
        filetype = "zarr"
    else:
        raise ValueError("The provided path does not have a .zarr or .nc extension.")
    
    # Parse and check the input and output paths
    if args.input_dir.endswith("/"):
        input_dir = args.input_dir
    else:
        input_dir = args.input_dir + "/"

    if args.output_dir.endswith("/"):
        output_dir = args.output_dir
    else:
        output_dir = args.output_dir + "/"

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The provided input directory {input_dir} does not exist.")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"The provided output directory {output_dir} does not exist.")

    if not os.path.exists(input_dir + args.name):
        raise FileNotFoundError(f"The provided input file {input_dir + args.name} does not exist.")
    
    output_name = args.name.split(".")[0] + ".nc"
    output_path = output_dir + output_name

    if filetype == 'zarr':
        ds = xr.open_dataset(input_dir + args.name, engine="zarr")
    elif filetype == 'netcdf':
        ds = xr.open_dataset(input_dir + args.name)
    
    logger.info(f"Loaded dataset: {args.name}")

    ds = preprocess_timeseries.preprocess(ds, in_edw=True)
    logger.info("Preproccesed the dataset")

    ds = preprocess_timeseries.create_EDW_criteria_masks(ds, sequestration=True)
    logger.info("Created EDW criteria masks")

    if args.compression:
        logger.info("Turning on compression of output variables.")
        encoding_compression = {key: {"zlib": True, "complevel": 3} for key in ds.data_vars}
        encoding = merge_dicts(encoding_compression, encoding_dtypes)
    else:
        encoding = encoding_dtypes

    ds.to_netcdf(output_path, encoding=encoding)
    logger.info(f"Saved dataset to {output_path}")

if __name__ == "__main__":
    main()