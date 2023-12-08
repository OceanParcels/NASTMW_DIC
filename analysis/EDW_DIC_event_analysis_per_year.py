import event_identification
import preprocess_timeseries
import xarray as xr
import logging
import sys
import pickle
import os
import argparse



sys.path.append('/nethome/4302001/tracer_backtracking/tools')

# ———————–—— Set logging ————————————–——
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
# ————————————————————————————————————————–——

data_in = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectories_postprocessed/"
# data_in = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectories/"

# data_out = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_batch_timescales/"
data_out = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_events_timescales_yearly_sensitivity/"


def event_detection(dataset, rollwindow=10, labeling=True):
    filename = dataset + ".nc"
    path = data_in + filename

    ds = xr.open_dataset(path)

    ds.load()

    logger.info(f"Loaded dataset: {dataset}")

    # ds = preprocess_timeseries.preprocess(ds, in_edw=True)
    # logger.info("Preproccesed the dataset")

    # ds = preprocess_timeseries.create_EDW_criteria_masks(ds, sequestration=True)
    # logger.info("Created EDW criteria masks")

    events = event_identification.batch_analysis(
        ds, ["cs_DIC_total", "cs_DIC_bio", "cs_DIC_bio_carbonate", "cs_DIC_bio_soft", "cs_DIC_diff", "cs_DIC_residual"],
        rollwindow=rollwindow if rollwindow != 1 else None)
    
    if not labeling:
        logger.info("Skipping labeling of events")

    events.categorize_events(label_events=labeling)
    logger.info("Categorized events")
  

    events.aggregate_events(provenance=True, labels=labeling)
    logger.info("Aggregated events")

    pickname = data_out + dataset + f"_events_rw{rollwindow}.pkl"
    with open(pickname, "wb") as f:
        pickle.dump(events, f)
    logger.info(f"Saved events. {os.stat(pickname).st_size / 1e9} GB")

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="Do an event analysis on a dataset.")
    parser.add_argument('-n', type=str, help="Run id, like `EDW_wfluxes_B_2010-03-01_1095d_dt90_odt24`. Ommit the .zarr extension.")
    parser.add_argument('-rw', type=int, default=10, const=None, nargs='?', help="Rolling window size in days. Default: 10. Pass without argument to use `None`.")
    parser.add_argument('--skip_labeling', action='store_true', help="Skip the labeling of the events. ")
    args = parser.parse_args()

    print("Starting event detection of dataset: ", args.n)
    event_detection(args.n, rollwindow=args.rw, labeling=not args.skip_labeling)
    print("Finished event detection of dataset: ", args.n)







