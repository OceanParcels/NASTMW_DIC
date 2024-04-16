import numpy as np
import xarray as xr
import pickle

import sys
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import copy

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.append('/nethome/4302001/NASTMW_DIC/tools')
import preprocess_timeseries


def find_peaks_and_valleys(x, include_edges=True):
    """
    Find peaks and valleys in a 1D array or xarray 

    Parameters
    ----------
    x : array-like
        1D array or xarray.DataArray to find peaks and valleys in
    include_edges : bool, optional
        Whether to include the first and last points as peaks/valleys

    Returns
    -------
    peaks : array-like
        Boolean array of peaks
    valleys : array-like
        Boolean array of valleys
    """
    def is_xarray(data):
        return isinstance(data, xr.DataArray)

    def zeros_like(data):
        return xr.zeros_like(data) if is_xarray(data) else np.zeros_like(data)

    # Convert to numpy array for peak and valley detection
    x_np = x.values if is_xarray(x) else x

    peaks_np = np.zeros_like(x_np, dtype=bool)
    valleys_np = np.zeros_like(x_np, dtype=bool)

    peaks_np[1:-1] = (x_np[1:-1] > x_np[:-2]) & (x_np[1:-1] > x_np[2:])
    valleys_np[1:-1] = (x_np[1:-1] < x_np[:-2]) & (x_np[1:-1] < x_np[2:])

    if include_edges:
        peaks_np[0] = x_np[0] > x_np[1]
        peaks_np[-1] = x_np[-1] > x_np[-2]
        valleys_np[0] = x_np[0] < x_np[1]
        valleys_np[-1] = x_np[-1] < x_np[-2]

    # Convert back to xarray DataArray if necessary
    if is_xarray(x):
        coords = x.coords
        peaks = xr.DataArray(peaks_np, coords=coords)
        valleys = xr.DataArray(valleys_np, coords=coords)
    else:
        peaks = peaks_np
        valleys = valleys_np

    return peaks, valleys


def adapt_peaks_and_valleys_for_rolling(da: xr.DataArray, rollwindow=10):
    """
    Find peaks and valleys in a 1D array or xarray. Employ a rolling window

    Parameters
    ----------
    da : array-like
        1D array or xarray.DataArray to find peaks and valleys in
    rollwindow : int, optional
        If not None, smooth the data with a rolling mean of this window size before finding peaks and valleys

    Returns
    -------
    peaks : array-like
        Boolean array of peaks
    valleys : array-like
        Boolean array of valleys
    """
    if type(rollwindow) is int:
        assert rollwindow % 2 == 0, "rollwindow must be even"
        da_rolled = da.rolling(obs=rollwindow).mean().shift(obs=-rollwindow//2)
        peaks, valleys = find_peaks_and_valleys(da_rolled, include_edges=True)
    else:
        peaks, valleys = find_peaks_and_valleys(da, include_edges=True)

    edges_determined = (False, False)
    if np.argmax(peaks.values) < np.argmax(valleys.values):
        start_with = 'increase'
        if rollwindow is not None:
            valleys[0] = True
    elif np.argmax(peaks.values) > np.argmax(valleys.values):
        start_with = 'decrease'
        if rollwindow is not None:
            peaks[0] = True
    if np.argmax(peaks.values[::-1]) < np.argmax(valleys.values[::-1]):
        valleys[-1] = True
    elif np.argmax(peaks.values[::-1]) > np.argmax(valleys.values[::-1]):
        peaks[-1] = True
    if np.argmax(peaks.values[::-1]) == np.argmax(valleys.values[::-1]):
        # This case handles a monotonic array
        if da[0] > da[-1]:
            start_with = 'decrease'
            if rollwindow is not None:
                peaks[0] = True
                valleys[0] = False
                peaks[-1] = False
                valleys[-1] = True
        else:
            start_with = 'increase'
            if rollwindow is not None:
                peaks[0] = False
                valleys[0] = True
                peaks[-1] = True
                valleys[-1] = False

    if type(rollwindow) == int:
        return peaks, valleys, da_rolled
    else:
        return peaks, valleys, None


def find_and_plot_pv(da: xr.DataArray, rollwindow=10, return_fig=False, style='default'):
    """
    Find peaks and valleys in a 1D array or xarray and plot them

    Parameters
    ----------
    da : array-like
        1D array or xarray.DataArray to find peaks and valleys in
    rollwindow : int, optional
        If not None, smooth the data with a rolling mean of this window size before finding peaks and valleys
    return_fig : bool, optional
        If True, return the figure
    style : str, optional
        Matplotlib style to use, by default 'default'
        
    Returns
    -------
    peaks : array-like
        Boolean array of peaks
    valleys : array-like
        Boolean array of valleys
    fig : matplotlib.figure.Figure (optional)
        Figure
    """
    if type(rollwindow) is int:
        peaks, valleys, da_rolled = adapt_peaks_and_valleys_for_rolling(da, rollwindow=rollwindow)
    else:
        peaks, valleys = find_peaks_and_valleys(da, include_edges=True)

    with plt.style.context([style]):
        fig, ax = plt.subplots(figsize=(6, 4))
        da.plot(ax=ax, color='#1f77b4')
        if rollwindow is not None:
            da_rolled.plot(linewidth=0.5, linestyle='--', color='k')
        da.isel(obs=peaks).plot.scatter(ax=ax, color='#2ca02c', zorder=100, alpha=0.75, label='peaks')
        da.isel(obs=valleys).plot.scatter(ax=ax, color='#d62728', zorder=100, alpha=0.75, label='valleys')

        ax.set_xlabel("Time [days]")
        ax.set_ylabel("DIC [µmol/L]")
        ax.set_xlim(0, da.obs.size)

        ax.set_title(f"{da.name}, Traj. {int(da.trajectory.values)}")
        ax.legend()

    what_returns = [peaks, valleys]
    if return_fig:
        what_returns += [fig]
    return what_returns


def categorize_events(da_arr, constituents=None, peaks=None, valleys=None, rollwindow=10):
    """
    Categorize peaks and valleys as either increasing or decreasing.

    If peaks and valleys are not provided, they will be found using `find_peaks_and_valleys`.

    Parameters
    ----------
    da_arr : array-like
        1D array or xarray.DataArray to find peaks and valleys in
    constituents : dictionary of array-like, optional
        dictionary of arrays with constituents that contribute to the total array
    peaks : array-like, optional
        Boolean array of peaks
    valleys : array-like, optional
        Boolean array of valleys
    rollwindow : int, optional
        If not None, smooth the data with a rolling mean of this window size before finding peaks and valleys

    Returns
    -------
    event_indices : list of tuples
        List of tuples of (start, end) indices for each event
    event_durations : list of ints
        List of Regime durations (in observations)
    event_magnitudes : np.ndarray of floats
        List of event magnitudes (in units of da_arr)
    constituent_magnitudes : dict of np.ndarray of floats (optional)
        Dictionary of np.ndarray of event magnitudes for each constituent.
        Only returned if constituents is not None
    """

    if peaks is None or valleys is None:
        peaks, valleys, da_rolled = adapt_peaks_and_valleys_for_rolling(da_arr, rollwindow=rollwindow)
    all_extremes = peaks.values + valleys.values


    peak_indices = np.where(peaks)[0]
    valley_indices = np.where(valleys)[0]
    extreme_indices = np.where(all_extremes)[0]

    n_extremes = len(extreme_indices)
    n_events = n_extremes - 1

    event_indices = np.zeros((n_events, 2), dtype=int)

    event_indices[:, 0] = extreme_indices[:-1]
    event_indices[:, 1] = extreme_indices[1:]

    event_durations = event_indices[:, 1] - event_indices[:, 0]
    # event_magnitudes = da_arr.isel(obs=event_indices[:, 1]).values - da_arr.isel(obs=event_indices[:, 0]).values # MINOR PERFORMANCE IMPROVEMENT
    event_magnitudes = da_arr[event_indices[:, 1]].values - da_arr[event_indices[:, 0]].values

    if constituents is not None:
        constituent_magnitudes = {}
        for constit_key, constit_arr in constituents.items():
            constituent_magnitudes[constit_key] = constit_arr[event_indices[:, 1]].values - constit_arr[event_indices[:, 0]].values

        return event_indices, event_durations, event_magnitudes, constituent_magnitudes

    else:
        return event_indices, event_durations, event_magnitudes


bit_labels = {
    "hasnan": 2**0,
    "edw_full": 2**1,
    "edw_part": 2**2,
    "mxl_full": 2**3,
    "mxl_part": 2**4,
    "temp_below_full": 2**5,
    "temp_below_part": 2**6,
    "temp_above_full": 2**7,
    "temp_above_part": 2**8,
    "thickness_regime_full": 2**9,
    "thickness_regime_part": 2**10,
    "in_strat_range_full": 2**11,
    "in_strat_range_part": 2**12,
    "edw_start_and_ends_inside": 2**13,
    "edw_start_and_ends_outside": 2**14,
    "edw_starts_out_ends_in": 2**15,
    "edw_starts_in_ends_out": 2**16,
    "sequestered_full": 2**17,
    "sequestered_part": 2**18,
}


bit_labels_inv = {v: k for k, v in bit_labels.items()}


def get_bit_labels(labels, bit_labels=bit_labels):
    """
    Write bit labels

    Parameters
    ----------
    labels : list of str or str
        List of labels
    bit_labels : dict, optional
        Dictionary of bit labels, by default bit_labels

    Returns
    -------
    bitcode : int
        Bit code label
    """
    bitcode = 0
    if type(labels) is str:
        bitcode = bit_labels[labels]
    else:
        for label in labels:
            bitcode += bit_labels[label]
    return bitcode


def read_bit_labels(bit_labels):
    """
    Read bit labels

    Parameters
    ----------
    bit_labels : int
        Bit labels

    Returns
    -------
    labels : list of str
        List of labels
    """
    labels = []
    for k, v in bit_labels_inv.items():
        if bit_labels & k:
            labels.append(v)
    return labels


def check_array_for_bit_labels(arr, bit_labels):
    """
    For each array item, check for each bit label whether it is set. 
    Return an array of bools, with shape (arr.shape), where each item is True 
    if the corresponding bit labels are all found

    Parameters
    ----------
    arr : array-like
        Array to check
    bit_labels : list or array-like
        List of bit labels to check

    Returns
    -------
    array-like
        Array of bools, with shape (arr.shape)
    """

    check = np.ones_like(arr, dtype=bool)

    for label in bit_labels:
        check = check & (arr & label).astype(bool)

    return check


def categorize_events_in_ds(ds, var, label_events=True, constituents=None, peaks=None, valleys=None, rollwindow=10):
    """
    Bin events, based on a var in ds, and label the type of trajectory that they occur in.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing one single trajectory
    var : str
        Variable to analyze
    label_events : bool, optional
        Whether to label the events, by default True
    constituents : list of str, optional
        List of constituents to analyze, by default None
        Constituents should be variables in ds
    peaks : array-like, optional
        Boolean array of peaks
    valleys : array-like, optional
        Boolean array of valleys
    rollwindow : int, optional
        If not None, smooth the data with a rolling mean of this window size before finding peaks and valleys

    Returns
    -------
    event_indices : list of tuples
        List of tuples of (start, end) indices for each event
    event_durations : list of ints
        List of Regime durations (in observations)
    event_magnitudes : np.ndarray of floats
        List of event magnitudes (in units of da_arr)
    event_labels : np.ndarray of ints
        List of event labels. None if label_events is False
    constituent_magnitudes : dict of np.ndarray of floats (optional)
        Dictionary of np.ndarray of event magnitudes for each constituent.
        Only returned if constituents is not None
    """

    assert ds.trajectory.values.size == 1, "Dataset must contain only one trajectory"

    da = ds[var]

    if constituents is not None:
        event_indices, event_durations, event_magnitudes, constituent_magnitudes = categorize_events(da,
                                                                                                     constituents=dict(
                                                                                                         zip(constituents, [ds[constit] for constit in constituents])),
                                                                                                     peaks=peaks,
                                                                                                     valleys=valleys,
                                                                                                     rollwindow=rollwindow)
    else:
        event_indices, event_durations, event_magnitudes = categorize_events(da, peaks=peaks, valleys=valleys, rollwindow=rollwindow)

    if label_events:
        event_labels = np.zeros_like(event_durations, dtype=int)

        ds.load()

        for event_idx, indices in enumerate(event_indices):
            slice_var = ds[var].values[indices[0]:indices[1]]
            slice_in_edw_strict = ds.in_edw_strict.values[indices[0]:indices[1]]
            slice_in_mixing_layer = ds.in_mixing_layer.values[indices[0]:indices[1]]
            slice_below_temp_range = ds.below_temp_range.values[indices[0]:indices[1]]
            slice_above_temp_range = ds.above_temp_range.values[indices[0]:indices[1]]
            slice_thick_enough = ds.thick_enough.values[indices[0]:indices[1]]
            slice_in_strat_range = ds.in_strat_range.values[indices[0]:indices[1]]
            slice_sequestered = ds.sequestered.values[indices[0]:indices[1]]

            if np.any(np.isnan(slice_var)):
                event_labels[event_idx] += bit_labels["hasnan"]

            if np.all(slice_in_edw_strict):
                event_labels[event_idx] += bit_labels["edw_full"]
            elif np.any(slice_in_edw_strict):
                event_labels[event_idx] += bit_labels["edw_part"]

            if np.all(slice_in_mixing_layer):
                event_labels[event_idx] += bit_labels["mxl_full"]
            elif np.any(slice_in_mixing_layer):
                event_labels[event_idx] += bit_labels["mxl_part"]

            if np.all(slice_below_temp_range):
                event_labels[event_idx] += bit_labels["temp_below_full"]
            elif np.any(slice_below_temp_range):
                event_labels[event_idx] += bit_labels["temp_below_part"]

            if np.all(slice_above_temp_range):
                event_labels[event_idx] += bit_labels["temp_above_full"]
            elif np.any(slice_above_temp_range):
                event_labels[event_idx] += bit_labels["temp_above_part"]

            if np.all(slice_thick_enough):
                event_labels[event_idx] += bit_labels["thickness_regime_full"]
            elif np.any(slice_thick_enough):
                event_labels[event_idx] += bit_labels["thickness_regime_part"]

            if np.all(slice_in_strat_range):
                event_labels[event_idx] += bit_labels["in_strat_range_full"]
            elif np.any(slice_in_strat_range):
                event_labels[event_idx] += bit_labels["in_strat_range_part"]

            if slice_in_edw_strict[0] and slice_in_edw_strict[-1]:
                event_labels[event_idx] += bit_labels["edw_start_and_ends_inside"]
            elif not slice_in_edw_strict[0] and not slice_in_edw_strict[-1]:
                event_labels[event_idx] += bit_labels["edw_start_and_ends_outside"]
            elif slice_in_edw_strict[0] and not slice_in_edw_strict[-1]:
                event_labels[event_idx] += bit_labels["edw_starts_in_ends_out"]
            elif not slice_in_edw_strict[0] and slice_in_edw_strict[-1]:
                event_labels[event_idx] += bit_labels["edw_starts_out_ends_in"]

            if np.all(slice_sequestered):
                event_labels[event_idx] += bit_labels["sequestered_full"]
            elif np.any(slice_sequestered):
                event_labels[event_idx] += bit_labels["sequestered_part"]

    else:
        event_labels = None

    if constituents is not None:
        return event_indices, event_durations, event_magnitudes, event_labels, constituent_magnitudes
    else:
        return event_indices, event_durations, event_magnitudes, event_labels


def filter_events(event_labels, filter_labels):
    """
    Filter events by bit labels. 

    Parameters
    ----------
    event_labels : list of ints
        List of event labels
    filter_labels : list of ints
        List of bit labels to filter by in an OR-fashion (since ANDs are already encoded in the labels)

    Returns
    -------
    filter : np.ndarray
        Boolean array of events that pass the filter
    """

    filter = np.zeros_like(event_labels, dtype=bool)
    if type(filter_labels) is int:
        label = filter_labels
        filter = filter | ((event_labels.astype(int) & label) == label).astype(bool)
    else:
        for label in filter_labels:
            filter = filter | ((event_labels.astype(int) & label) == label).astype(bool)

    return filter


def filter_from_dataset(event_dict=None, event_arr=None, bool_da=None, mode='all', last_index=None, last_index_inclusion_mode='any', returnmode='flatten'):
    """
    Given event_indices (dict with trajs and arrays of indices, or only one array of indices),
    create a filter to check whether the events in the dataset are fully or partially true.

    Parameters
    ----------
    event_dict : dict 
        Event dictionary, as returned by categorize_events_in_ds (be sure to select one variable)
    event_arr : array-like
        Array of event indices, as returned by categorize_events_in_ds
    bool_da : xarray.Dataset
        Dataset of boolean variables
    mode : str, optional
        Whether the event should have 'any' or 'all' True values in the dataset, by default 'all'
    returnmode : str, optional
        Whether to return a flattened filter (array) or a dictionary of filters, by default 'flatten'
    last_index : int, optional
        If not None, only consider events up to this index
    last_index_inclusion_mode : str, optional
       Whether the last event should have 'any' or 'all' True values in the dataset, by default 'any'

    Returns
    -------
    filter : array-like or dict
        Filter to check whether the events in the dataset are fully or partially true
    """
    last_index_idx = None 
    if type(event_dict) == dict:
        filter = {}
        assert np.all(np.array(list(event_dict.keys())) == bool_da.trajectory), "Trajectories in event_dict and bool_da must match"
        for trajdx, traj in tqdm(enumerate(event_dict.keys()), desc="looping over trajectories", position=0):
            filter[traj] = np.zeros_like(event_dict[traj]['durations'], dtype=bool)
            if last_index is not None:
                # find the index of the last event that should be considered
                last_index_idx =  np.searchsorted(event_dict[traj]['indices'][:, 0], last_index)
                indices = event_dict[traj]['indices'][:last_index_idx]
            else:
                indices = event_dict[traj]['indices']
            for event_idx, index_pair in enumerate(indices):
                if last_index is not None and event_idx == last_index_idx - 1:
                    if last_index_inclusion_mode == 'any':
                        filter[traj][event_idx] = np.any(bool_da.values[trajdx, index_pair[0]:index_pair[1]])
                    elif last_index_inclusion_mode == 'all':
                        filter[traj][event_idx] = np.all(bool_da.values[trajdx, index_pair[0]:index_pair[1]])
                    else:
                        raise ValueError("last_index_inclusion_mode must be 'any' or 'all'")
                else:
                    if mode == 'all':
                        filter[traj][event_idx] = np.all(bool_da.values[trajdx, index_pair[0]:index_pair[1]])
                    elif mode == 'any':
                        filter[traj][event_idx] = np.any(bool_da.values[trajdx, index_pair[0]:index_pair[1]])

        if returnmode == 'flatten':
            filter = np.concatenate(list(filter.values()))
        elif returnmode == 'dictionary':
            pass
        else:
            raise ValueError("returnmode must be 'flatten' or 'dictionary'")

    elif type(event_arr) == np.ndarray:
        assert len(bool_da.values.shape) == 1, "bool_da must be 1D"
        filter = np.zeros_like(event_arr, dtype=bool)
        for event_idx, indices in enumerate(event_arr):
            if mode == 'all':
                filter[event_idx] = np.all(bool_da.values[indices[0]:indices[1]])
            elif mode == 'any':
                filter[event_idx] = np.any(bool_da.values[indices[0]:indices[1]])

    return filter


def multi_event_filter_from_ds(event_dict, bool_da, mode='any', last_index=None, last_index_inclusion_mode='any', returnmode='flatten'):
    """
    Returns a dictionary with masks for each variable in the event_dict

    Parameters
    ----------
    event_dict : dict
        Dictionary with events (event_dict[var])
    bool_da : xarray.DataArray
        Boolean DataArray with the same dimensions as the ds that is used to create the event_dict
        mode : str, optional
        Whether the event should have 'any' or 'all' True values in the dataset, by default 'all'
    returnmode : str, optional
        Whether to return a flattened filter (array) or a dictionary of filters, by default 'flatten'
    last_index : int, optional
        If not None, only consider events up to this index
    last_index_inclusion_mode : str, optional
       Whether the last event should have 'any' or 'all' True values in the dataset, by default 'any'

    Returns
    -------
    mask_dict : dict
        Dictionary with masks for each event in the event_dict
    """

    mask_dict = {}
    for var in ["cs_DIC_total", "cs_DIC_bio_soft", "cs_DIC_bio_carbonate", "cs_DIC_diff"]:
        mask_dict[var] = filter_from_dataset(
            event_dict=event_dict[var], bool_da=bool_da, mode=mode, last_index=last_index, last_index_inclusion_mode=last_index_inclusion_mode, returnmode=returnmode)

    return mask_dict


def bin_events(event_durations, event_magnitudes, constituent_magnitudes=None, bin_width=10, bin_edges=None, return_bins=False, normalize=False, skip_nan=True):
    """
    Bin events by duration and magnitude.

    Parameters
    ----------
    event_durations : array-like
        List of Regime durations (in observations)
    event_magnitudes : array-like
        List of event magnitudes (in units of da_arr)
    constituent_magnitudes : dict of array-like, optional
        Dictionary of np.ndarray of event magnitudes for each constituent.
    bin_width : int, optional
        Width of bins in observations
    bin_edges : array-like, optional
        Edges of bins. If specified, bin_width is ignored
    return_bins : bool, optional
        If True, return the bin edges
    normalize : bool, optional
        If True, normalize the event magnitudes by their duration


    Returns
    -------
    binned_events : dict
        Dictionary of binned events, with keys corresponding to bin indices
    binned_events_normed (optional) : dict
        Dictionary of binned events, with keys corresponding to bin indices
    binned_constits (optional): dict of dict
        Dictionary of binned constituents, with keys corresponding to constituent keys and bin indices
        Only returned if constituent_magnitudes is not None
    tot_pos_diff : array-like (n_bins,)
        Total positive difference in each bin
    tot_neg_diff : array-like (n_bins,)
        Total negative difference in each bin
    total_diff : array-like (n_bins,)
        Total difference in each bin (positive + negative)
    bins_edges : array-like (n_bins+1,)
        Edges of bins. Only returned if return_bins is True
    bins_edges_bounds : array-like (n_bins, 2)
        Bounds of bins. Only returned if return_bins is True
    """
    if skip_nan == True:
        nan_filter = ~np.isnan(event_magnitudes)
        # if type(constituent_magnitudes) is not type(None):
        #     for constit_key, constit_arr in constituent_magnitudes.items():
        #         assert np.all(nan_filter == ~np.isnan(constit_arr)), "nans should be consistent between arrays"

        event_durations = event_durations[nan_filter]
        event_magnitudes = event_magnitudes[nan_filter]
        if type(constituent_magnitudes) is not type(None):
            for constit_key, constit_arr in constituent_magnitudes.items():
                constituent_magnitudes[constit_key] = constit_arr[nan_filter]

    if type(bin_edges) == type(None) and type(bin_width) == int:
        bins_edges = np.arange(0, 101, bin_width)  # last bin corresponds to exceeding the upper bound
        bins_edges = np.concatenate((bins_edges, [np.inf]))
    else:
        bins_edges = bin_edges
        if bins_edges[-1] != np.inf:
            print("Warning: last bin should correspond to exceeding the upper bound.")
            print("Appending np.inf to bin edges.")
            bins_edges = np.concatenate((bins_edges, [np.inf]))
    bins_edges_bounds = np.vstack((bins_edges[:-1], bins_edges[1:])).T
    bindices = np.arange(len(bins_edges)-1)
    binned_events = dict(zip(bindices, [[] for _ in bindices]))
    if normalize:
        binned_events_normed = dict(zip(bindices, [[] for _ in bindices]))

    tot_pos_diff = np.zeros_like(bins_edges[:-1], np.float64)
    tot_neg_diff = np.zeros_like(bins_edges[:-1], np.float64)

    if constituent_magnitudes is not None:
        binned_constits = dict(zip(constituent_magnitudes.keys(), [dict(
            zip(bindices, [[] for _ in bindices])) for _ in constituent_magnitudes.keys()]))

    for event_id, duration in enumerate(event_durations):
        bindex = np.searchsorted(bins_edges, duration) - 1 # bins are inclusive to the right boundary
        binned_events[bindex].append(event_magnitudes[event_id])
        if normalize:
            binned_events_normed[bindex].append(event_magnitudes[event_id] / duration)
        if constituent_magnitudes is not None:
            for constit_key, constit_arr in constituent_magnitudes.items():
                binned_constits[constit_key][bindex].append(constit_arr[event_id])

        if event_magnitudes[event_id] > 0:
            tot_pos_diff[bindex] += event_magnitudes[event_id]
        else:
            tot_neg_diff[bindex] += event_magnitudes[event_id]

    total_diff = tot_pos_diff + tot_neg_diff

    what_returns = [binned_events]
    if constituent_magnitudes is not None:
        what_returns += [binned_constits]
    if normalize:
        what_returns += [binned_events_normed]
    what_returns += [tot_pos_diff, tot_neg_diff, total_diff]
    if return_bins:
        what_returns += [bins_edges, bins_edges_bounds]

    return what_returns


def process_binned_constits(binned_constits):
    """
    Compute total positive, total negative, and net contributions for each processes for each bin

    Parameters
    ----------
    binned_constits : dict of dict
        Dictionary of binned constituents, with keys corresponding to constituent keys and bin indices, 
        containing arrays of constituent magnitudes

    Returns
    -------
    processed_bins_constits : dict of dict of dict
        Dictionary of binned constituents, with keys corresponding to positive/negative/net constituent keys and bin indices
    """
    processed_bins_constits = {"positive": {}, "negative": {}, "net": {}}
    for constit_key, constit_bins in binned_constits.items():
        processed_bins_constits["positive"][constit_key] = np.zeros(len(binned_constits[constit_key].keys()), dtype=np.float64)
        processed_bins_constits["negative"][constit_key] = np.zeros(len(binned_constits[constit_key].keys()), dtype=np.float64)
        processed_bins_constits["net"][constit_key] = np.zeros(len(binned_constits[constit_key].keys()), dtype=np.float64)
        for bin, arr in constit_bins.items():
            processed_bins_constits["positive"][constit_key][bin] = np.sum(np.array(arr)[np.array(arr) > 0])
            processed_bins_constits["negative"][constit_key][bin] = np.sum(np.array(arr)[np.array(arr) < 0])
            processed_bins_constits["net"][constit_key][bin] = np.sum(np.array(arr))

    return processed_bins_constits


pretty_DIC_variables = {
    "cs_DIC_bio": "Biology",
    "cs_DIC_bio_soft": "Soft tissue",
    "cs_DIC_bio_carbonate": "Carbonate",
    "cs_DIC_diff": "Mixing",
    "cs_DIC_residual": "Residual",
}


constituent_colors = {
    "cs_DIC_total": ("#1f77b4", "#a6cee3"), # blue
    "cs_DIC_bio_soft": ("#ff7f00", "#fdbf6f"), # orange
    "cs_DIC_bio_carbonate": ("#33a02c", "#b2df8a"), # green
    "cs_DIC_diff": ("#e31a1c", "#fb9a99"), # red
    "cs_DIC_residual": ("#6a3d9a", "#cab2d6") # purple
}


def plot_event_distribution(binned_events, tot_pos_diff, tot_neg_diff, total_diff, bins_edges_bounds,
                            binned_constits=None, types=["bar", "box"], da=None, return_fig=False,
                            hide_outliers=True, normalize_bars=False, normed_events=None,
                            title=None, color_as=None, style='default'):
    """
    Plot bar and box plot of event distribution

    Parameters
    ----------
    binned_events : dict
        Dictionary of binned events, with keys corresponding to bin indices
    tot_pos_diff : array-like (n_bins,)
        Total positive difference in each bin
    tot_neg_diff : array-like (n_bins,)
        Total negative difference in each bin
    total_diff : array-like (n_bins,)
        Total difference in each bin (positive + negative)
    bins_edges_bounds : array-like (n_bins, 2)
        Bounds of bins
    binned_constits : dict, optional
        Dictionary of binned constituents, with keys corresponding to bin indices
    types : list of str, optional
        Types of plots to include. Options are "bar", "box", and "box_no_extremes"
    da : xarray.DataArray, optional
        DataArray of data, used for labelling
    return_fig : bool, optional
        Whether to return the figure
    hide_outliers : bool, optional
        Whether to hide outliers in the box plots  
    normalize_bars : bool, optional
        Whether to normalize the bar plots by the total positive and negative differences  
    normed_events : dict
        Dictionary of binned events, normalized to their duration, with keys corresponding to bin indices
    title : str, optional
        Title of plot
    color_as : str, optional
        Color bars according to this constituent
    style : str, optional
        Style of plot
    """

    total_events_per_bin = np.array([len(binned_events[i]) for i in range(len(binned_events))])

    if normalize_bars:
        tot_pos_diff_sum = tot_pos_diff.sum()
        tot_neg_diff_sum = tot_neg_diff.sum()
        # normalizer = np.abs(tot_pos_diff_sum) + np.abs(tot_neg_diff_sum)
        # normalizer = np.abs(tot_pos_diff_sum - tot_neg_diff_sum)
        normalizer = np.abs(total_diff.sum())
    else:
        normalizer = 1

    nbars = bins_edges_bounds.shape[0]
    nrows = 1
    with plt.style.context([style]):
    # with plt.style.context(['ggplot']):
        fig = plt.figure(constrained_layout=True, figsize=(8,3.5))
        gs = fig.add_gridspec(ncols=4, nrows=2, figure=fig, height_ratios=[0.01,4], width_ratios=[5, 0.7, 5, 0.7])

        ax_bar = fig.add_subplot(gs[:, 0])
        ax_bar_other = fig.add_subplot(gs[:, 1])
        bin_widths = bins_edges_bounds[:-1, 1] - bins_edges_bounds[:-1, 0]

        if binned_constits is not None:
            processed_bins_constits = process_binned_constits(binned_constits)
            process_binned_constits_hatched = copy.deepcopy(processed_bins_constits)
            process_binned_constits_last = copy.deepcopy(processed_bins_constits)
            for bar_type in process_binned_constits_hatched.keys():
                for var in process_binned_constits_hatched[bar_type].keys():
                    process_binned_constits_hatched[bar_type][var] = processed_bins_constits[bar_type][var][:-1]
                    process_binned_constits_last[bar_type][var] = processed_bins_constits[bar_type][var][-1]

            bar_keys = ["cs_DIC_bio_soft", "cs_DIC_bio_carbonate", "cs_DIC_diff", "cs_DIC_residual"]

            cmap = plt.get_cmap('Paired')

            pos_bottom = 0
            neg_top = 0
            for i, bar_key in enumerate(bar_keys):
                ax_bar.bar(bins_edges_bounds[:-1, 0] + bin_widths/2, process_binned_constits_hatched["positive"][bar_key]/normalizer, width=bin_widths,
                           bottom=pos_bottom, color=constituent_colors[bar_key][0], label=pretty_DIC_variables[bar_key], zorder=100)
                ax_bar.bar(bins_edges_bounds[:-1, 0] + bin_widths/2, process_binned_constits_hatched["negative"][bar_key]/normalizer, width=bin_widths,
                           bottom=neg_top, color=constituent_colors[bar_key][1], zorder=100)
                pos_bottom += process_binned_constits_hatched["positive"][bar_key]/normalizer
                neg_top += process_binned_constits_hatched["negative"][bar_key]/normalizer
        else:
            if color_as is not None:
                ax_bar.bar(bins_edges_bounds[:-1, 0] + bin_widths/2, tot_pos_diff[:-1]/normalizer, width=bin_widths, zorder=100, color=constituent_colors[color_as][0])
                ax_bar.bar(bins_edges_bounds[:-1, 0] + bin_widths/2, tot_neg_diff[:-1]/normalizer, width=bin_widths, zorder=100, color=constituent_colors[color_as][1])
            else:
                ax_bar.bar(bins_edges_bounds[:-1, 0] + bin_widths/2, tot_pos_diff[:-1]/normalizer, width=bin_widths, zorder=100)
                ax_bar.bar(bins_edges_bounds[:-1, 0] + bin_widths/2, tot_neg_diff[:-1]/normalizer, width=bin_widths, zorder=100)
        ax_bar.bar(bins_edges_bounds[:-1, 0] + bin_widths/2, total_diff[:-1]/normalizer, width=bin_widths,
                   label='Net', color='none', edgecolor='k', hatch='//', linewidth=0.5, zorder=100)

        ax_bar.axhline(0, color='k', linestyle='--')
        ax_bar.set_xlim(0, 100)
        ax_bar.minorticks_on()
        ax_bar.xaxis.set_minor_locator(MultipleLocator(10))
        if normalize_bars:
            ax_bar.set_ylabel("Normalized $\Delta$DIC", fontsize=10)
        else:
            ax_bar.set_ylabel("Total $\Delta$DIC [µmol/L$]", fontsize=10)
        ax_bar.legend(loc='best', fontsize=8)
        # ax_bar.set_title("Sum of each event's $\Delta$DIC")
        ax_bar.set_xlabel("Regime duration [days]", fontsize=10)

        if binned_constits is not None:
            pos_bottom = 0
            neg_top = 0
            for i, bar_key in enumerate(bar_keys):
                # print(bar_key, process_binned_constits_last["positive"][bar_key]/normalizer, process_binned_constits_last["negative"][bar_key]/normalizer)
                ax_bar_other.bar(1, process_binned_constits_last["positive"][bar_key]/normalizer, width=10,
                                 bottom=pos_bottom, color=constituent_colors[bar_key][0], label=pretty_DIC_variables[bar_key], zorder=100)
                ax_bar_other.bar(1, process_binned_constits_last["negative"][bar_key]/normalizer, width=10,
                                 bottom=neg_top, color=constituent_colors[bar_key][0], zorder=100)
                pos_bottom += process_binned_constits_last["positive"][bar_key]/normalizer
                neg_top += process_binned_constits_last["negative"][bar_key]/normalizer
        else:
            if color_as is not None:
                ax_bar_other.bar(1, tot_pos_diff[-1]/normalizer, width=10, zorder=100, color=constituent_colors[color_as][0])
                ax_bar_other.bar(1, tot_neg_diff[-1]/normalizer, width=10, zorder=100, color=constituent_colors[color_as][1])
            else:
                ax_bar_other.bar(1, tot_pos_diff[-1]/normalizer, width=10, zorder=100)
                ax_bar_other.bar(1, tot_neg_diff[-1]/normalizer, width=10, zorder=100)
            ax_bar_other.bar(1, total_diff[-1]/normalizer, width=10,
                             color='none', edgecolor='k', hatch='//', linewidth=0.5, zorder=100)
        ax_bar_other.bar(1, total_diff[-1]/normalizer, width=10,
                   label='Net', color='none', edgecolor='k', hatch='//', linewidth=0.5, zorder=100)

        ax_bar_other.axhline(0, color='k', linestyle='--')

        lims = ax_bar.get_ylim(), ax_bar_other.get_ylim()
        ax_bar.set_ylim(min(lims[0][0], lims[1][0]), max(lims[0][1], lims[1][1]))
        ax_bar_other.set_ylim(min(lims[0][0], lims[1][0]), max(lims[0][1], lims[1][1]))
        ax_bar.text(-0.22, 0.97, "a)", fontweight='bold', ha='right', va='top', transform=ax_bar.transAxes)


        if type(normed_events) is dict:
            boxplot_data = list(normed_events.values())

        else:
            boxplot_data = list(binned_events.values())

        ax_box = fig.add_subplot(gs[1, 2], sharex=ax_bar)
        ax_box_other = fig.add_subplot(gs[1, 3], sharex=ax_bar_other)
        ax_box.set_yscale('symlog')
        ax_box_other.set_yscale('symlog')
        boxplot = ax_box.boxplot(boxplot_data[:-1],
                                 positions=bins_edges_bounds[:-1, 0] + bin_widths/2,
                                 widths=bin_widths * 0.75, 
                                 patch_artist=True)
        boxplot_other = ax_box_other.boxplot(boxplot_data[-1], positions=[1], widths=7.5,
                                             patch_artist=True)

        for bplot in (boxplot, boxplot_other):
            for patch in bplot['boxes']:
                patch.set_facecolor('white')

        ax_box.text(-0.1, 1.3, "b)", fontweight='bold', ha='right', va='top', transform=ax_box.transAxes)

        # Computing stats for boxplot
        # Initialize arrays
        Q1 = np.zeros(nbars)
        Q3 = np.zeros(nbars)
        IQR = np.zeros(nbars)
        outliers_above_percents = np.zeros(nbars)
        outliers_below_percents = np.zeros(nbars)

        high_adjust = 0
        low_adjust = 0

        # Cleaning up outliers in the boxplots
        if hide_outliers:
            for boxidx in np.arange(nbars):

                data = np.array(boxplot_data[boxidx])

                if len(data) > 0:
                    Q1[boxidx] = np.quantile(data, 0.25)
                    Q3[boxidx] = np.quantile(data, 0.75)
                    IQR[boxidx] = Q3[boxidx] - Q1[boxidx]

                    lower_bound = Q1[boxidx] - 1.5 * IQR[boxidx]
                    upper_bound = Q3[boxidx] + 1.5 * IQR[boxidx]

                    outliers_above = data[data > upper_bound]
                    outliers_below = data[data < lower_bound]

                    outliers_above_percents[boxidx] = len(outliers_above) / len(data) * 100
                    outliers_below_percents[boxidx] = len(outliers_below) / len(data) * 100

                    # plot outlier percentages above and below
                    if outliers_above.size > 0:
                        max_outlier = outliers_above.max()
                        if max_outlier > 0:
                            flier_label_offset_top = max_outlier*2
                        else:
                            flier_label_offset_top = max_outlier*0.5
                        if boxidx == nbars-1:
                            ax_box_other.scatter(1,
                                                 max_outlier, color='none', marker='v', edgecolors='black')
                            ax_box_other.text(1, flier_label_offset_top,
                                              f'{outliers_above_percents[boxidx]:.1f}%', ha='center', va='center', fontsize=7,
                                              bbox=dict(facecolor='white', edgecolor='none', pad=0), color='gray')
                        else:
                            ax_box.scatter(bins_edges_bounds[boxidx, 0] + bin_widths[boxidx]/2,
                                           max_outlier, color='none', marker='v', edgecolors='black')
                            ax_box.text(bins_edges_bounds[boxidx, 0] + bin_widths[boxidx]/2, flier_label_offset_top,
                                        f'{outliers_above_percents[boxidx]:.1f}%', ha='center', va='center', fontsize=7,
                                        bbox=dict(facecolor='white', edgecolor='none', pad=0), color='gray')
                        if flier_label_offset_top > high_adjust:
                            high_adjust = flier_label_offset_top*2
                    if outliers_below.size > 0:
                        min_outlier = outliers_below.min()
                        if min_outlier < 0:
                            flier_label_offset_bottom = min_outlier*2
                        else:
                            flier_label_offset_bottom = min_outlier*0.5
                        if boxidx == nbars-1:
                            ax_box_other.scatter(1,
                                                 min_outlier, color='none', marker='v', edgecolors='black')
                            ax_box_other.text(1, flier_label_offset_bottom,
                                              f'{outliers_below_percents[boxidx]:.1f}%', ha='center', va='center', fontsize=7,
                                              bbox=dict(facecolor='white', edgecolor='none', pad=0), color='gray')
                        else:
                            ax_box.scatter(bins_edges_bounds[boxidx, 0] + bin_widths[boxidx]/2,
                                           min_outlier, color='none', marker='^', edgecolors='black')
                            ax_box.text(bins_edges_bounds[boxidx, 0] + bin_widths[boxidx]/2, flier_label_offset_bottom,
                                        f'{outliers_below_percents[boxidx]:.1f}%', ha='center', va='center', fontsize=7,
                                        bbox=dict(facecolor='white', edgecolor='none', pad=0), color='gray')
                        if flier_label_offset_bottom < low_adjust:
                            low_adjust = flier_label_offset_bottom*2

            for outlier in boxplot['fliers']:
                outlier.set_visible(False)
            for outlier in boxplot_other['fliers']:
                outlier.set_visible(False)

        # Adjust ticks for the other boxplot
        ax_bar_other.set_yticklabels([])
        ax_bar_other.set_xticks([1])
        ax_bar_other.set_xticklabels([f"> {int(bins_edges_bounds[-1, 0])}"])

        # Adjusting axes
        ax_box.set_xticks(np.arange(0, 110, 20))
        ax_box.set_xticklabels([0, 20, 40, 60, 80, 100])
        ax_box.set_xlabel("Regime duration [days]", fontsize=10)
        if type(normed_events) is dict:
            ax_box.set_ylabel("$\Delta$DIC [µmol/L/day]", fontsize=10)
        else:
            ax_box.set_ylabel("$\Delta$DIC [µmol/L]", fontsize=10)

        lims = ax_box.get_ylim(), ax_box_other.get_ylim()

        ax_box.set_ylim(min(lims[0][0], lims[1][0], low_adjust, -1e-1), max(lims[0][1], lims[1][1], high_adjust, 1e-1))
        ax_box_other.set_ylim(min(lims[0][0], lims[1][0], low_adjust, -1e-1), max(lims[0][1], lims[1][1], high_adjust, 1e-1))
        ax_box_other.set_yticklabels([])

        ax_bar_other.set_yticklabels([])
        ax_bar_other.set_xticks([1])
        ax_bar_other.set_xticklabels([f"> {int(bins_edges_bounds[-1, 0])}"])

        num_samples = [len(data) for data in boxplot_data[:-1]]
        top_y_limit = ax_box.get_ylim()[1]
        label_offset = top_y_limit * 0.7

        for i, num in enumerate(num_samples):
            if num > 0:
                label_y_pos = top_y_limit + label_offset
                ax_box.text(bins_edges_bounds[i, 0] + bin_widths[i]/2, label_y_pos,
                            f'n={num}', ha='center', va='bottom', fontsize=7, rotation=90, color='gray')

        if len(boxplot_data[-1]) > 0:
            label_y_pos = top_y_limit + label_offset
            ax_box_other.text(1, label_y_pos, f'n={len(boxplot_data[-1])}',
                              ha='center', va='bottom', fontsize=7, rotation=90, color='gray')

        title_base = "Strength of DIC enrichment and depletion events per duration"
        if type(da) == xr.core.dataarray.DataArray:
            total_diff_sum = total_diff.sum()
            title_default = f"{da.name} — Traj:{da.trajectory.values}" + "\n" \
                + title_base + "\n" \
                + rf"(total $\Delta$DIC: {total_diff_sum:.2f} mmol/m$^3$)"
        else:
            total_diff_sum = None
            title_default = ""

        ax_bar.grid(axis='y', linestyle='--', zorder=-100)
        ax_bar_other.grid(axis='y', linestyle='--', zorder=-100)

        ax_bar.spines['top'].set_visible(False)
        ax_bar_other.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar_other.spines['right'].set_visible(False)
        ax_bar_other.spines['left'].set_visible(False)
        ax_bar.spines['left'].set_color('gray')
        ax_bar.spines['bottom'].set_color('gray')
        ax_bar_other.spines['bottom'].set_color('gray')

        ax_bar.tick_params(axis='both', which='major', colors='gray')
        ax_bar_other.tick_params(axis='both', which='major', colors='gray')

        ax_box.grid(axis='y', linestyle='--', zorder=-100)
        ax_box_other.grid(axis='y', linestyle='--', zorder=-100)

        ax_box.spines['top'].set_visible(False)
        ax_box_other.spines['top'].set_visible(False)
        ax_box.spines['right'].set_visible(False)
        ax_box_other.spines['right'].set_visible(False)
        ax_box_other.spines['left'].set_visible(False)
        ax_box.spines['left'].set_color('gray')
        ax_box.spines['bottom'].set_color('gray')
        ax_box_other.spines['bottom'].set_color('gray')

        ax_box.tick_params(axis='both', which='major', colors='gray')
        ax_box_other.tick_params(axis='both', which='major', colors='gray')

        fig.suptitle(title if title is not None else title_default)

        if return_fig:
            return fig
        else:
            plt.show()


def plot_aggregate_event_distribution(binned_events, bins_edges_bounds):
    """

    """
    with plt.style.context(['ggplot']):
        fig = plt.figure(constrained_layout=True, figsize=(5, 4))
        gs = fig.add_gridspec(ncols=2, nrows=1, figure=fig, width_ratios=[5, 0.7])

        boxplot_data = list(binned_events.values())

        bin_widths = bins_edges_bounds[:-1, 1] - bins_edges_bounds[:-1, 0]

        ax_box = fig.add_subplot(gs[0, 0])
        ax_box_other = fig.add_subplot(gs[0, 1])

        bplot = ax_box.boxplot(boxplot_data[:-1],
                               positions=bins_edges_bounds[:-1, 0] + bin_widths/2,
                               widths=bin_widths * 0.75, 
                               boxprops=dict(facecolor='white'),
                               zorder=100)

        bplot_other = ax_box_other.boxplot(boxplot_data[-1], positions=[1], widths=10 * 0.75)

        ax_box.set_xlim(0, 100)
        ax_box.minorticks_on()
        ax_box.xaxis.set_minor_locator(MultipleLocator(10))

        ax_box_other.set_yticklabels([])
        ax_box_other.set_xticks([1])
        ax_box_other.set_xticklabels(["> 100"])

        ax_box.set_xticks(np.arange(0, 110, 20))
        ax_box.set_xticklabels([0, 20, 40, 60, 80, 100])
        ax_box.set_xlabel("Regime duration [days]")
        ax_box.set_ylabel(r"$\Delta$DIC [mmol/m^3]")

        lims = ax_box_other.get_ylim(), ax_box_other.get_ylim()
        ax_box.set_ylim(min(lims[0][0], lims[1][0]), max(lims[0][1], lims[1][1]))
        ax_box_other.set_ylim(min(lims[0][0], lims[1][0]), max(lims[0][1], lims[1][1]))
        ax_box_other.set_yticklabels([])

        num_samples = [len(data) for data in boxplot_data[:-1]]
        top_y_limit = ax_box.get_ylim()[1]
        label_offset = top_y_limit * 0.1  # 5% of the y range

        for i, num in enumerate(num_samples):
            if num > 0:
                label_y_pos = top_y_limit + label_offset
                ax_box.text(bins_edges_bounds[i, 0] + bin_widths[i]/2, label_y_pos,
                            f'n={num}', ha='center', va='bottom', fontsize=5, rotation=90, color='gray')

        if len(boxplot_data[-1]) > 0:
            label_y_pos = top_y_limit + label_offset
            ax_box_other.text(1, label_y_pos, f'n={len(boxplot_data[-1])}',
                              ha='center', va='bottom', fontsize=5, rotation=90, color='gray')

        title = None #"Strength of DIC enriching and depletion events per associated duration"

        fig.suptitle(title)
        return fig


def aggregate_events(event_dict, trajfilter=None, provenance=True, labels=True, indices=False, progress=True, up_to_index=None):
    """
    Aggregate events for each variable in varlist. If constituent magnitudes are available, aggregate those too.

    Aggregate_event_dict    
    └─ variable
        ├─ durations
        ├─ magnitudes
        ├─ labels
        ├─ provenance (optional)
        └─ constituents
            └─ constituent
                └─ magnitudes

    """
    varlist = list(event_dict.keys())

    if trajfilter is not None:
        trajlist = trajfilter
    else:
        trajlist = np.array(list(event_dict['cs_DIC_total'].keys()))
    
    

    aggregate_event_dict = dict(zip(varlist, [{"durations": [], "magnitudes": [], "labels": [], "constituents": {}} for var in varlist]))

    if provenance:
        for var in event_dict.keys():
            aggregate_event_dict[var]["provenance"] = []
    if indices:
        for var in event_dict.keys():
            aggregate_event_dict[var]["index_start"] = []
            aggregate_event_dict[var]["index_end"] = []

    n_total_events = 0

    for var in tqdm(varlist, desc="Looping over variables", position=0, disable=not progress):
        for trajectory in tqdm(trajlist, desc="Looping over trajectories", position=1, leave=False, disable=not progress):
            if up_to_index is not None:
                index_mask = event_dict[var][trajectory]["indices"][:, 0] < up_to_index
            else:
                index_mask = np.ones_like(event_dict[var][trajectory]["indices"][:, 0], dtype=bool)
            aggregate_event_dict[var]["durations"].append(event_dict[var][trajectory]["durations"][index_mask])
            aggregate_event_dict[var]["magnitudes"].append(event_dict[var][trajectory]["magnitudes"][index_mask])
            if labels:
                aggregate_event_dict[var]["labels"].append(event_dict[var][trajectory]["labels"][index_mask])

            n_total_events += event_dict[var][trajectory]["durations"].size

            if provenance:
                aggregate_event_dict[var]["provenance"].append(trajectory * np.ones_like(event_dict[var][trajectory]["durations"][index_mask]))

            if indices:
                aggregate_event_dict[var]["index_start"].append(event_dict[var][trajectory]["indices"][:, 0][index_mask])
                aggregate_event_dict[var]["index_end"].append(event_dict[var][trajectory]["indices"][:, 1][index_mask])

            if event_dict[var][trajectory]["constituents"] is not None:
                for constit_key, constit_arr in event_dict[var][trajectory]["constituents"].items():
                    if constit_key not in aggregate_event_dict[var]["constituents"].keys():
                        aggregate_event_dict[var]["constituents"][constit_key] = []
                    aggregate_event_dict[var]["constituents"][constit_key].append(constit_arr[index_mask])

        aggregate_event_dict[var]["durations"] = np.concatenate(aggregate_event_dict[var]["durations"])
        aggregate_event_dict[var]["magnitudes"] = np.concatenate(aggregate_event_dict[var]["magnitudes"])
        if labels:
            aggregate_event_dict[var]["labels"] = np.concatenate(aggregate_event_dict[var]["labels"])

        if len(aggregate_event_dict[var]["constituents"]) > 0:
            for constit_key, constit_arr in aggregate_event_dict[var]["constituents"].items():
                aggregate_event_dict[var]["constituents"][constit_key] = np.concatenate(
                    aggregate_event_dict[var]["constituents"][constit_key])
        
        if provenance:
            aggregate_event_dict[var]["provenance"] = np.concatenate(aggregate_event_dict[var]["provenance"])

        if indices:
            aggregate_event_dict[var]["index_start"] = np.concatenate(aggregate_event_dict[var]["index_start"])
            aggregate_event_dict[var]["index_end"] = np.concatenate(aggregate_event_dict[var]["index_end"])

    return aggregate_event_dict


class batch_analysis:
    """
    Class for detecting events in a dataset and binning them by duration and magnitude.
    """

    def __init__(self, ds, varlist, rollwindow=10, deepcopy=False):
        """
        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing variables in varlist
        varlist : list of str
            List of variables to analyze
        rollwindow : int, optional
            Window size for rolling mean
        """
        self.ds = ds
        if deepcopy:
            self.ds_attrs = copy.deepcopy(self.ds.attrs)
            self.ds_DIC_dims = copy.deepcopy(self.ds.DIC.dims)
        else:
            self.ds_attrs = self.ds.attrs
            self.ds_DIC_dims = self.ds.DIC.dims
        self.varlist = varlist
        self.rollwindow = rollwindow
        self.trajlist = ds.trajectory.values

    def __getstate__(self):
        # Remove the ds attribute from the object's dictionary for pickling
        state = self.__dict__.copy()
        del state['ds']
        return state

    def __setstate__(self, state):
        # Update the object's dictionary to None when unpickling
        self.__dict__.update(state)
        self.ds = None

    def hook_up_ds(self, ds, verify=True):
        self.ds = ds
        if verify:
            if ds.attrs != self.ds_attrs:
                print("Warning: dataset attributes have changed since initialization")
            if ds.DIC.dims != self.ds_DIC_dims:
                print("Warning: DIC dimensions have changed since initialization")

    def categorize_events(self, label_events=True):
        """
        Categorize events for each variable in varlist. If constituent magnitudes are available, categorize those too
        (in a nested fashion). Note that labeling in a large dataset can take a long time.

         Event_dict
         └─ variable
            └─ trajectory
               ├─ indices
               ├─ durations
               ├─ magnitudes
               ├─ labels
               └─ constituents
        """

        self.event_dict = dict(zip(self.varlist, [dict(
            zip(self.trajlist, [{"indices": None,
                                 "durations": None,
                                 "magnitudes": None,
                                 "labels": None,
                                 "constituents": None
                                 } for traj in self.trajlist])) for var in self.varlist]))

        
        for var in tqdm(self.varlist, desc="Looping over variables", position=0):
            for trajdx, trajectory in tqdm(enumerate(self.trajlist), desc="Looping over trajectories", position=1, leave=False):
                if var == "cs_DIC_total":
                    event_indices, event_durations, event_magnitudes, event_labels, constituent_magnitudes = categorize_events_in_ds(
                        self.ds.isel(trajectory=trajdx),
                        var=var,
                        constituents=["cs_DIC_bio_soft", "cs_DIC_bio_carbonate", "cs_DIC_diff", "cs_DIC_residual"],
                        rollwindow=self.rollwindow,
                        label_events=label_events)
                
                    self.event_dict[var][trajectory]["indices"] = event_indices
                    self.event_dict[var][trajectory]["durations"] = event_durations
                    self.event_dict[var][trajectory]["magnitudes"] = event_magnitudes
                    if label_events:
                        self.event_dict[var][trajectory]["labels"] = event_labels
                    self.event_dict[var][trajectory]["constituents"] = constituent_magnitudes
                else:
                    # event_indices, event_durations, event_magnitudes = categorize_events(
                    #     self.ds[var].sel(trajectory=trajectory),
                    #     rollwindow=self.rollwindow)
                    # self.event_dict[var][trajectory]["indices"] = event_indices
                    # self.event_dict[var][trajectory]["durations"] = event_durations
                    # self.event_dict[var][trajectory]["magnitudes"] = event_magnitudes
                    event_indices, event_durations, event_magnitudes, event_labels = categorize_events_in_ds(
                        self.ds.isel(trajectory=trajdx),
                        var=var,
                        rollwindow=self.rollwindow,
                        label_events=label_events
                    )
                    self.event_dict[var][trajectory]["indices"] = event_indices
                    self.event_dict[var][trajectory]["durations"] = event_durations
                    self.event_dict[var][trajectory]["magnitudes"] = event_magnitudes
                    if label_events:
                        self.event_dict[var][trajectory]["labels"] = event_labels

    def aggregate_events(self, provenance=True, labels=True):
        """
        Aggregate events for each variable in varlist. If constituent magnitudes are available, aggregate those too.

        Aggregate_event_dict    
        └─ variable
            ├─ durations
            ├─ magnitudes
            ├─ labels
            ├─ provenance (optional)
            └─ constituents
                └─ constituent
                    └─ magnitudes

        """
        assert hasattr(self, "event_dict"), "Run categorize_events first"

        self.aggregate_event_dict = aggregate_events(self.event_dict, provenance=provenance, labels=labels)


    def check_integrity(self):
        for var in self.varlist:
            assert self.aggregate_event_dict[var]["durations"].shape == self.aggregate_event_dict[var]["magnitudes"].shape, "Durations and magnitudes do not have the same shape"
            assert self.aggregate_event_dict[var]["durations"].shape == self.aggregate_event_dict[var]["labels"].shape, "Durations and labels do not have the same shape"

            if len(self.aggregate_event_dict[var]["constituents"]) > 0:
                for constit_key, constit_arr in self.aggregate_event_dict[var]["constituents"].items():
                    assert self.aggregate_event_dict[var]["durations"].shape == constit_arr.shape, "Durations and constituents do not have the same shape"

            if "provenance" in self.aggregate_event_dict.keys():
                assert self.aggregate_event_dict[var]["durations"].shape == self.aggregate_event_dict[var]["provenance"].shape, "Durations and provenance do not have the same shape"


def bin_aggregate_events(aggregate_event_dict, filter_dict=None, vars_to_analyze=None,
                         normalize=False, bin_width=None, bin_edges=np.arange(0, 101, 10),
                         skip_nan=True):
    """
    Bin the aggregated events for each variable in varlist. If constituent magnitudes are available, bin those too.
    Labels are not used, since the events are already detached from their trajectories. Rather, labels can be used to 
    Filter the events before binning.

    Parameters
    ----------
    aggregate_event_dict : dict
        Dictionary of aggregated events, with keys corresponding to variables in varlist
        Usually comes from the batch_analysis.aggregate_events method
    filter_dict : dict (optional)
        Dictionary with boolean arrays of same length as durations, magnitudes, and labels, for each variable. If provided, only events with True values will be binned
        This array can be constructed with the 'filter_events' function and the 'aggregate_event_dict' 'labels' key. 
    vars_to_analyze : list of str, optional
        List of variables to analyze. If None, all variables in varlist will be analyzed
    normalize : bool, optional
        If True, the binned events will be normalized by their durations
    bin_width : float, optional
        Width of bins. If None, bin_edges must be specified
    bin_edges : array-like, optional
        Edges of bins. If None, bin_width must be specified
    skip_nan : bool, optional
        If True, events with NaN magnitudes will be skipped

    Returns
    -------
    binned_events_dict : dict
        Dictionary of binned events, with keys corresponding to variables in varlist
        └─ variable
            ├─ binned_events: array of binned event magnitudes
            ├─ binned_constits (optional; if constituent magnitudes are available)
            │   └─ constituent
            │       └─ array of binned constituent magnitudes 
            ├─ tot_pos_diff: array of total positive difference in each bin
            ├─ tot_neg_diff: array of total negative difference in each bin
            ├─ total_diff: array of total difference in each bin (positive + negative)
            ├─ bins_edges: array of bin edges
            └─ bins_edges_bounds: array of bin edges bounds

    """
    if type(vars_to_analyze) is list:
        varlist = vars_to_analyze
    else:
        varlist = aggregate_event_dict.keys()
    binned_events_dict = dict(zip(varlist, [{"binned_events": None,
                                             "binned_constits": None,
                                             "tot_pos_diff": None,
                                             "tot_neg_diff": None,
                                             "total_diff": None,
                                             "bins_edges": None,
                                             "bins_edges_bounds": None
                                             } for var in varlist]))
    for var in varlist:
        if var == "cs_DIC_total":

            if var in filter_dict.keys():
                durations_view = aggregate_event_dict[var]["durations"][filter_dict[var]]
                magnitudes_view = aggregate_event_dict[var]["magnitudes"][filter_dict[var]]
                constituent_magnitudes_view = {}
                for k, v in aggregate_event_dict[var]["constituents"].items():
                    constituent_magnitudes_view[k] = v[filter_dict[var]]

            else:
                durations_view = aggregate_event_dict[var]["durations"]
                magnitudes_view = aggregate_event_dict[var]["magnitudes"]
                constituent_magnitudes_view = aggregate_event_dict[var]["constituents"]

            if normalize == True:
                binned_events_dict[var]["binned_events"], \
                    binned_events_dict[var]["binned_constits"], \
                    binned_events_dict[var]["binned_events_normed"], \
                    binned_events_dict[var]["tot_pos_diff"], \
                    binned_events_dict[var]["tot_neg_diff"], \
                    binned_events_dict[var]["total_diff"], \
                    binned_events_dict[var]["bins_edges"], \
                    binned_events_dict[var]["bins_edges_bounds"] = bin_events(event_durations=durations_view,
                                                                              event_magnitudes=magnitudes_view,
                                                                              constituent_magnitudes=constituent_magnitudes_view,
                                                                              return_bins=True,
                                                                              bin_width=bin_width,
                                                                              bin_edges=bin_edges,
                                                                              normalize=True,
                                                                              skip_nan = skip_nan)

            else:
                binned_events_dict[var]["binned_events"], \
                    binned_events_dict[var]["binned_constits"], \
                    binned_events_dict[var]["tot_pos_diff"], \
                    binned_events_dict[var]["tot_neg_diff"], \
                    binned_events_dict[var]["total_diff"], \
                    binned_events_dict[var]["bins_edges"], \
                    binned_events_dict[var]["bins_edges_bounds"] = bin_events(event_durations=durations_view,
                                                                              event_magnitudes=magnitudes_view,
                                                                              constituent_magnitudes=constituent_magnitudes_view,
                                                                              return_bins=True,
                                                                              bin_width=bin_width,
                                                                              bin_edges=bin_edges,
                                                                              normalize=False,
                                                                              skip_nan = skip_nan)
        else:
            if var in filter_dict.keys():
                durations_view = aggregate_event_dict[var]["durations"][filter_dict[var]]
                magnitudes_view = aggregate_event_dict[var]["magnitudes"][filter_dict[var]]
            else:
                durations_view = aggregate_event_dict[var]["durations"]
                magnitudes_view = aggregate_event_dict[var]["magnitudes"]
            if normalize == True:
                binned_events_dict[var]["binned_events"], \
                    binned_events_dict[var]["binned_events_normed"], \
                    binned_events_dict[var]["tot_pos_diff"], \
                    binned_events_dict[var]["tot_neg_diff"], \
                    binned_events_dict[var]["total_diff"], \
                    binned_events_dict[var]["bins_edges"], \
                    binned_events_dict[var]["bins_edges_bounds"] = bin_events(event_durations=durations_view,
                                                                              event_magnitudes=magnitudes_view,
                                                                              return_bins=True,
                                                                              bin_width=bin_width,
                                                                              bin_edges=bin_edges,
                                                                              normalize=True, 
                                                                              skip_nan = skip_nan)
            else:
                binned_events_dict[var]["binned_events"], \
                    binned_events_dict[var]["tot_pos_diff"], \
                    binned_events_dict[var]["tot_neg_diff"], \
                    binned_events_dict[var]["total_diff"], \
                    binned_events_dict[var]["bins_edges"], \
                    binned_events_dict[var]["bins_edges_bounds"] = bin_events(event_durations=durations_view,
                                                                              event_magnitudes=magnitudes_view,
                                                                              return_bins=True,
                                                                              bin_width=bin_width,
                                                                              bin_edges=bin_edges,
                                                                              normalize=False, 
                                                                              skip_nan = skip_nan)

    return binned_events_dict


def binned_events_multiyear_aggregate(binned_events_per_year):
    n_years = len(binned_events_per_year.keys())
    n_bins = len(binned_events_per_year[1995]["cs_DIC_total"]["binned_events"])

    binned_events = {year: {} for year in binned_events_per_year[1995]}
    for key in binned_events.keys():
        print(key)
        binned_events[key]["binned_events"] = {bin: [] for bin in binned_events_per_year[1995][key]["binned_events"]}
        binned_events[key]["binned_events_normed"] = {bin: [] for bin in binned_events_per_year[1995][key]["binned_events_normed"]}
        binned_events[key]["tot_pos_diff_arr"] = np.zeros((n_years, n_bins))
        binned_events[key]["tot_pos_diff_agg"] = np.zeros(n_bins)
        binned_events[key]["tot_neg_diff_arr"] = np.zeros((n_years, n_bins))
        binned_events[key]["tot_neg_diff_agg"] = np.zeros(n_bins)
        binned_events[key]["total_diff_arr"] = np.zeros((n_years, n_bins))
        binned_events[key]["total_diff_agg"] = np.zeros(n_bins)
        binned_events[key]["bins_edges"] = binned_events_per_year[1995][key]["bins_edges"]
        binned_events[key]["bins_edges_bounds"] = binned_events_per_year[1995][key]["bins_edges_bounds"]

        if binned_events_per_year[1995][key]["binned_constits"] is not None:
            binned_events[key]["binned_constits"] = {constit_key: {bin : [] for bin in np.arange(n_bins)} for constit_key in binned_events_per_year[1995][key]["binned_constits"]}

        for yearidx, year in enumerate(binned_events_per_year.keys()):
            for bin in binned_events_per_year[year][key]["binned_events"]:
                binned_events[key]["binned_events"][bin].append(np.array(binned_events_per_year[year][key]["binned_events"][bin]))
                binned_events[key]["binned_events_normed"][bin].append(np.array(binned_events_per_year[year][key]["binned_events_normed"][bin]))
                binned_events[key]["tot_pos_diff_arr"][yearidx, bin] = binned_events_per_year[year][key]["tot_pos_diff"][bin]
                binned_events[key]["tot_neg_diff_arr"][yearidx, bin] = binned_events_per_year[year][key]["tot_neg_diff"][bin]
                binned_events[key]["total_diff_arr"][yearidx, bin] = binned_events_per_year[year][key]["total_diff"][bin]
            if binned_events_per_year[year][key]["binned_constits"] is not None:
                for constit_key in binned_events_per_year[year][key]["binned_constits"]:
                    for bin in np.arange(n_bins):
                        binned_events[key]["binned_constits"][constit_key][bin].append(binned_events_per_year[year][key]["binned_constits"][constit_key][bin])

        
        for bin in binned_events[key]["binned_events"]:
            binned_events[key]["binned_events"][bin] = np.concatenate(binned_events[key]["binned_events"][bin])
            binned_events[key]["binned_events_normed"][bin] = np.concatenate(binned_events[key]["binned_events_normed"][bin])
            binned_events[key]["tot_pos_diff_agg"][bin] = np.sum(binned_events[key]["tot_pos_diff_arr"][:, bin])
            binned_events[key]["tot_neg_diff_agg"][bin] = np.sum(binned_events[key]["tot_neg_diff_arr"][:, bin])
            binned_events[key]["total_diff_agg"][bin] = np.sum(binned_events[key]["total_diff_arr"][:, bin])

        if binned_events_per_year[1995][key]["binned_constits"] is not None:
            for constit_key in binned_events_per_year[1995][key]["binned_constits"]:
                for inner_bin in np.arange(n_bins):
                    binned_events[key]["binned_constits"][constit_key][inner_bin] = np.concatenate(binned_events[key]["binned_constits"][constit_key][inner_bin])
            
    return binned_events


def open_from_pickle(run_id, 
                     pickle_dir="/nethome/4302001/data/output_data/data_Daan/EDW_batch_analyses/",
                     pickle_suffix="_events.pkl"):
    """
    Open a pickled batch_analysis object and link it to the corresponding dataset

    Parameters
    ----------
    run_id : str
        Run ID of the batch_analysis object
    pickle_dir : str, optional
        Directory of pickled batch_analysis object
    pickle_suffix : str, optional
        Suffix of pickled batch_analysis object
    """
    with open(pickle_dir + run_id + pickle_suffix, "rb") as f:
        ba = pickle.load(f)
    return ba

def open_from_pickle_and_link(run_id,
                              pickle_dir="/nethome/4302001/data/output_data/data_Daan/EDW_batch_analyses/",
                              pickle_suffix="_events.pkl",
                              ds_dir="/nethome/4302001/data/output_data/data_Daan/EDW_trajectories/",
                              preprocess={
                                  "fluxes": True,
                                  "in_edw": True,
                                  "mask": True,
                                  "sequestration": True,
                              },
                              verify=True):
    """
    Open a pickled batch_analysis object and link it to the corresponding dataset

    Parameters
    ----------
    run_id : str
        Run ID of the batch_analysis object
    pickle_dir : str, optional
        Directory of pickled batch_analysis object
    pickle_suffix : str, optional
        Suffix of pickled batch_analysis object
    ds_dir : str, optional
        Directory of dataset
    verify : bool, optional
        If True, verify that the dataset and batch_analysis object come from the same run

    Returns
    -------
    event_identification.batch_analysis
        Batch analysis object with associated dataset
    """

    ba = open_from_pickle(run_id, pickle_dir=pickle_dir, pickle_suffix=pickle_suffix)
    print("Loaded batch analysis object")

    ds = xr.open_dataset(ds_dir + run_id + ".zarr", engine="zarr")
    print("Loaded dataset")

    if preprocess["fluxes"]:
        if preprocess["in_edw"]:
            ds = preprocess_timeseries.preprocess(ds, in_edw=True)
        else:
            ds = preprocess_timeseries.preprocess(ds, in_edw=False)
        print("Preprocessed fluxes")

    if preprocess["mask"]:
        if preprocess["sequestration"]:
            ds = preprocess_timeseries.create_EDW_criteria_masks(ds, sequestration=True)
        else:
            ds = preprocess_timeseries.create_EDW_criteria_masks(ds, sequestration=False)
        print("Created masks")

    ba.hook_up_ds(ds, verify=verify)

    return ba


def check_mask_ds_attrs(traj_ds, mask_ds):
    """
    Check if the mask dataset and trajectory dataset come from the same run

    Parameters
    ----------
    traj_ds : xarray.Dataset
        Trajectory dataset
    mask_ds : xarray.Dataset
        Mask dataset

    Returns
    -------
    bool
        True if the datasets come from the same run, False otherwise
    """
    traj_id = f"EDW_wfluxes_B_{traj_ds.attrs['T0']}_{traj_ds.attrs['simulation_time']}d_dt{traj_ds.attrs['dt']}_odt{traj_ds.attrs['output_dt']}"
    try:
        assert mask_ds.attrs["run"] == traj_id
        return True
    except AssertionError:
        print("Mask and trajectory do not match!")
        print("Mask id: " + mask_ds.attrs["run"])
        print("Traj id: " + traj_id)
        return False
    
    
def check_events_mask_trajs(events_object, mask_ds):
    """
    Check if the events object and mask dataset come from the same run
    by checking the trajectory labels (using assert).

    Parameters
    ----------
    events_object : event_identification.batch_analysis
        Batch analysis object
    mask_ds : xarray.Dataset
        Mask dataset

    Returns
    -------
    None
    """
    assert set(events_object.event_dict["cs_DIC_total"].keys()) == set(mask_ds.trajectory.values), "Events and mask trajectories do not match!"