import xarray as xr

import sys
import os
import glob

sys.path.append('/nethome/4302001/tracer_backtracking/tools')
import preprocess_timeseries

data_dir = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectories/"
output_dir_DIC = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_DIC_changes/"
output_dir_masks = "/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectory_masks/"  

forward_september_runs = list(sorted(glob.glob(data_dir + "EDW_wfluxes_B_*-09-01_1095d_dt90_odt24.zarr")))
backward_september_runs = list(sorted(glob.glob(data_dir + "EDW_wfluxes_B_*-09-01_1095d_dt-90_odt24.zarr")))
forward_march_runs = list(sorted(glob.glob(data_dir + "EDW_wfluxes_B_*-03-01_1095d_dt90_odt24.zarr")))
backward_march_runs = list(sorted(glob.glob(data_dir + "EDW_wfluxes_B_*-03-01_1095d_dt-90_odt24.zarr")))


def open_and_process_ds(filename):
    """
    Preprocess a dataset of EDW trajectories.

    Parameters
    ----------
    filename : str
        Path to the dataset

    Returns
    -------
    ds : xr.Dataset
        Preprocessed dataset
    """
    ds = xr.open_zarr(filename, chunks=None)
    ds = preprocess_timeseries.preprocess(ds, in_edw=True)
    ds = preprocess_timeseries.create_EDW_criteria_masks(ds, sequestration=True)
    return ds


def chain_mask(criteria_list, stats=True):
    """
    Combine a list of criteria into a single mask by taking the logical AND of all criteria.
    Print statistics about the number of True and False values for each criterion and the combined mask.

    Parameters
    ----------
    criteria_list : list of xr.DataArray
        List of criteria to combine into a single mask
    stats : bool, optional
        Whether to print statistics about the number of True and False values for each criterion and the combined mask

    Returns
    -------
    cumulative_mask : xr.DataArray
        Mask that is the logical AND of all criteria in the list
    """
    # Initialize a mask with all True values of the same shape as the criteria
    cumulative_mask = xr.ones_like(criteria_list[0], dtype=bool)

    for idx, criterion in enumerate(criteria_list, start=1):
        # Calculate the number of True and False values for the current criterion
        nTrue = criterion.sum().values
        nFalse = (~criterion).sum().values

        # Update the cumulative mask
        previous_cumulative_mask = cumulative_mask.copy()
        cumulative_mask = cumulative_mask * criterion

        # Calculate the number of True and False values after combining criteria
        n_total_Trues = cumulative_mask.sum().values
        n_total_Falses = (~cumulative_mask).sum().values

        # Calculate the number of new exclusions
        new_exclusions = (previous_cumulative_mask * ~criterion).sum().values

        if stats:
            # Print the results
            print(f"Criterion {idx}")
            print(f"True: {nTrue} (total: {n_total_Trues}, new exclusions: {new_exclusions})")
            print(f"False: {nFalse} (total: {n_total_Falses}) \n")
    if 'obs' in cumulative_mask.coords:
        cumulative_mask = cumulative_mask.drop_vars(['obs', 'time'])
    return cumulative_mask


def recompute_last_sigma0(ds, edw_run_mask, sequestration_threshold=0.05):
    """
    Recompute the last_sigma0 field in the dataset, which is used to identify densification events.
    This is done by forward filling the last_sigma0 field onto positions where the EDW run mask is False.
    Then, the densification events are identified by comparing the sigma0 field to the last_sigma0 field.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the sigma0 and last_sigma0 fields
    edw_run_mask : xr.DataArray
        Mask indicating whether a particle is in EDW
    sequestration_threshold : float, optional
        Threshold for the difference between sigma0 and last_sigma0 to identify densification events.
        Densities are allowed to increase by this amount while still being considered part of the same densification event.
    
    Returns
    -------
    ds : xr.Dataset
        Dataset containing the sigma0, last_sigma0 and sequestered fields
    """

    ds["last_sigma0"] = ds.sigma0.where(edw_run_mask).ffill('obs').where(~edw_run_mask)
    heavier_than_last = (ds.sigma0 >= ds.last_sigma0 - sequestration_threshold)
    lighter_than_last = (ds.sigma0 < ds.last_sigma0 - sequestration_threshold)

    codes = (heavier_than_last*0 + 1).where(heavier_than_last).fillna(0) + \
            (lighter_than_last*0 + 2).where(lighter_than_last).fillna(0) + \
            (edw_run_mask * 0 + 3).where(edw_run_mask).fillna(0)
    # Forward fill codes onto indices from heavier_than_last. This shows what preceded the heavier_than_last code label.
    # Then select only positions that originally had heavier_than_last and lighter_than_last
    # Then select only code 3, since this means its
    # a heavier_than_last that was preceded by EDW
    heavier_out_of_EDW = (codes.where((codes == 2) + (codes == 3)).ffill(dim='obs').where(codes < 3) == 3)
    lighter_out_of_EDW = (codes.where((codes == 1) + (codes == 3)).ffill(dim='obs').where(codes < 3) == 3)
    ds[f"sequestered_{str(sequestration_threshold).replace('.', '')}"] = heavier_out_of_EDW * ~ds['in_mixed_layer']
    ds[f"lightened_{str(sequestration_threshold).replace('.', '')}"] = lighter_out_of_EDW * ~ds['in_mixed_layer']
    return ds


def analysis(filename, mode='forward', mask_export=False, DIC_export=True):
    """
    Analyze a single EDW trajectory dataset of a single run. Compute DIC changes in EDW and in different EDW masks.

    Parameters
    ----------
    filename : str
        Path to the dataset
    mode : str, optional
        Whether the run is forward or backward
    mask_export : bool, optional
        Whether to export the trajectory masks
    DIC_export : bool, optional
        Whether to export the DIC changes

    Returns
    -------
    None
    """

    basename = os.path.splitext(os.path.basename(filename))[0]
    print(f"Opening: {basename}")
    ds = open_and_process_ds(filename)
    print(f"Opened.")

    print("Creating start in EDW mask")
    start_in_edw = chain_mask([ds.isel(obs=0).in_edw == True,
                               (ds.isel(obs=0).EDW_part_of_biggest_blob > 0.1) + (ds.isel(obs=0).EDW_part_of_smaller_blob > 0.1),
                               ds.isel(obs=0).EDW_layer_thickness > 50])
    print("EDW run mask")
    edw_run_mask = chain_mask([ds.in_edw == True,
                               (ds.EDW_part_of_biggest_blob > 0.1) + (ds.EDW_part_of_smaller_blob > 0.1),
                               ds.EDW_layer_thickness > 50],
                              stats=False)
    mask_dict = {}
    if mode == 'forward':
        print("Creating EDW persistent mask")
        mask_dict["forw_persistent_mask"] = chain_mask([start_in_edw,
                                           (ds.isel(obs=365).in_edw == True),
                                           ((ds.isel(obs=365).EDW_part_of_biggest_blob > 0.1) +
                                            (ds.isel(obs=365).EDW_part_of_smaller_blob > 0.1)),
                                           (ds.isel(obs=365).EDW_layer_thickness > 50),
                                           ds.isel(obs=365).in_mixing_layer == False])

        print("Creating EDW persistent throughout run mask")
        mask_dict["forw_persistent_mask_full"] = chain_mask([start_in_edw,
                                                (ds.isel(obs=slice(0, 365)).in_edw == True).all('obs'),
                                                ((ds.isel(obs=slice(0, 365)).EDW_part_of_biggest_blob > 0.1) +
                                                 (ds.isel(obs=slice(0, 365)).EDW_part_of_smaller_blob > 0.1)).all('obs'),
                                                (ds.isel(obs=slice(0, 365)).EDW_layer_thickness > 50).all('obs'),
                                                ds.isel(obs=365).in_mixing_layer == False])

        print("forward ventilation mask")
        mask_dict["forw_any_outcropping_column_mask"] = chain_mask([start_in_edw,
                                                       (ds.isel(obs=slice(0, 365)).EDW_outcropping_column_mask == True).any('obs')])
        mask_dict["forw_part_of_outcropping_blob_mask"] = chain_mask([start_in_edw,
                                                         (ds.isel(obs=slice(0, 365)).EDW_part_of_outcropping_blob == True).any('obs')])
        mask_dict["forw_been_in_mixed_layer"] = chain_mask([start_in_edw,
                                               (ds.isel(obs=slice(0, 365)).in_mixed_layer == True).any('obs')])
        mask_dict["forw_been_in_mixing_layer"] = chain_mask([start_in_edw,
                                                (ds.isel(obs=slice(0, 365)).in_mixing_layer == True).any('obs')])
        mask_dict["forw_ventilating_mask_1y"] = chain_mask([start_in_edw,
                                               ds.isel(obs=365*1).in_mixed_layer == True])
        mask_dict["forw_ventilating_mask_2y"] = chain_mask([start_in_edw,
                                               ds.isel(obs=365*2).in_mixed_layer == True])
        mask_dict["forw_ventilating_mask_3y"] = chain_mask([start_in_edw,
                                               ds.isel(obs=365*3).in_mixed_layer == True])

        print("forward densification mask")
        ds=recompute_last_sigma0(ds, edw_run_mask, sequestration_threshold=0.05)
        ds=recompute_last_sigma0(ds, edw_run_mask, sequestration_threshold=0.01)
        ds=recompute_last_sigma0(ds, edw_run_mask, sequestration_threshold=0)

        mask_dict["forw_densification_mask_005_1y"] = chain_mask([start_in_edw,
                                                      ds.isel(obs=365*1).sequestered_005 == True])
        mask_dict["forw_densification_mask_005_2y"] = chain_mask([start_in_edw,
                                                      ds.isel(obs=365*2).sequestered_005 == True])
        mask_dict["forw_densification_mask_005_3y"] = chain_mask([start_in_edw,
                                                      ds.isel(obs=365*3).sequestered_005 == True])
        mask_dict["forw_densification_mask_005_2y_yearpersist"] = chain_mask([start_in_edw,
                                                      (ds.isel(obs=slice(365, 365*2)).sequestered_005 == True).all('obs')])
        mask_dict["forw_densification_mask_005_3y_yearpersist"] = chain_mask([start_in_edw,
                                                      (ds.isel(obs=slice(365*2, 365*3)).sequestered_005 == True).all('obs')])
        
        mask_dict["forw_densification_mask_001_1y"] = chain_mask([start_in_edw,
                                                      ds.isel(obs=365*1).sequestered_001 == True])
        mask_dict["forw_densification_mask_001_2y"] = chain_mask([start_in_edw,
                                                      ds.isel(obs=365*2).sequestered_001 == True])
        mask_dict["forw_densification_mask_001_3y"] = chain_mask([start_in_edw,
                                                      ds.isel(obs=365*3).sequestered_001 == True])
        mask_dict["forw_densification_mask_001_2y_yearpersist"] = chain_mask([start_in_edw,
                                                        (ds.isel(obs=slice(365, 365*2)).sequestered_001 == True).all('obs')])
        mask_dict["forw_densification_mask_001_3y_yearpersist"] = chain_mask([start_in_edw,
                                                        (ds.isel(obs=slice(365*2, 365*3)).sequestered_001 == True).all('obs')])
        
        mask_dict["forw_densification_mask_0_1y"] = chain_mask([start_in_edw,
                                                      ds.isel(obs=365*1).sequestered_0 == True])
        mask_dict["forw_densification_mask_0_2y"] = chain_mask([start_in_edw,
                                                      ds.isel(obs=365*2).sequestered_0 == True])
        mask_dict["forw_densification_mask_0_3y"] = chain_mask([start_in_edw,
                                                      ds.isel(obs=365*3).sequestered_0 == True])
        mask_dict["forw_densification_mask_0_2y_yearpersist"] = chain_mask([start_in_edw,
                                                        (ds.isel(obs=slice(365, 365*2)).sequestered_0 == True).all('obs')])
        mask_dict["forw_densification_mask_0_3y_yearpersist"] = chain_mask([start_in_edw,
                                                        (ds.isel(obs=slice(365*2, 365*3)).sequestered_0 == True).all('obs')])

    if mode == 'backward':
        print("backward subduction mask")
        mask_dict["backw_subduction_mask_1y"] = chain_mask([start_in_edw,
                                            (ds.isel(obs=1*365).in_mixed_layer == True)])
        mask_dict["backw_subduction_mask_2y"] = chain_mask([start_in_edw,
                                            (ds.isel(obs=2*365).in_mixed_layer == True)])
        mask_dict["backw_subduction_mask_3y"] = chain_mask([start_in_edw,
                                            (ds.isel(obs=3*365).in_mixed_layer == True)])
        mask_dict["backw_any_outcropping_column_mask"] = chain_mask([start_in_edw,
                                                        (ds.isel(obs=slice(0, 365)).EDW_outcropping_column_mask == True).any('obs')])
        mask_dict["backw_part_of_outcropping_blob_mask"] = chain_mask([start_in_edw,
                                                          (ds.isel(obs=slice(0, 365)).EDW_part_of_outcropping_blob == True).any('obs')])


    if mask_export:
        print("Computing mask changes")
        mask_ds_list = []
        mask_ds_list.append(start_in_edw.rename("start_in_edw"))
        for mask_name, mask in mask_dict.items():
            mask_ds_list.append(mask.rename(mask_name))
        mask_ds = xr.merge(mask_ds_list)
        mask_ds.attrs['description'] = "EDW trajectory_masks"
        mask_ds.attrs['run'] = basename

        print(f"Saving {basename}")
        mask_ds.to_netcdf(output_dir_masks + basename + "_masks.nc")
        mask_ds.close()


    if DIC_export:
        print("Computing DIC changes")
        result_ds_list = []
        result_ds_list.append(ds.DIC[start_in_edw, 0].rename("DIC_start").drop_vars(['obs', 'time']))
        result_ds_list.append(ds.DIC[start_in_edw, 365].rename("DIC_1y").drop_vars(['obs', 'time']))
        result_ds_list.append(ds.DIC[start_in_edw, 365*2].rename("DIC_2y").drop_vars(['obs', 'time']))
        if mode == 'forward':
            result_ds_list.append(ds.DIC[start_in_edw, 365*3].rename("DIC_3y").drop_vars(['obs', 'time']))
        elif mode == 'backward':
            result_ds_list.append(ds.DIC[start_in_edw, 365*3-1].rename("DIC_3y").drop_vars(['obs', 'time'])) # hack to get last timestep

        for mask_name, mask in mask_dict.items():
            result_ds_list.append(ds.DIC[mask, 365].rename(f"DIC_1y_{mask_name}").drop_vars(['obs', 'time']))
            result_ds_list.append(ds.DIC[mask, 365*2].rename(f"DIC_2y_{mask_name}").drop_vars(['obs', 'time']))
            if mode == 'forward':
                result_ds_list.append(ds.DIC[mask, 365*3].rename(f"DIC_3y_{mask_name}").drop_vars(['obs', 'time']))
            elif mode == 'backward':
                result_ds_list.append(ds.DIC[mask, 365*3-1].rename(f"DIC_3y_{mask_name}").drop_vars(['obs', 'time']))
            if "3y" in mask_name:
                result_ds_list.append(ds.cs_DIC_total[mask, 365*3].rename(f"cum_DIC_3y_total_{mask_name}").drop_vars(['obs', 'time'])) 
                result_ds_list.append(ds.cs_DIC_bio[mask, 365*3].rename(f"cum_DIC_3y_bio_{mask_name}").drop_vars(['obs', 'time']))
                result_ds_list.append(ds.cs_DIC_bio_soft[mask, 365*3].rename(f"cum_DIC_3y_bio_soft_{mask_name}").drop_vars(['obs', 'time']))
                result_ds_list.append(ds.cs_DIC_bio_carbonate[mask, 365*3].rename(f"cum_DIC_3y_bio_carbonate_{mask_name}").drop_vars(['obs', 'time']))
                result_ds_list.append(ds.cs_DIC_diff[mask, 365*3].rename(f"cum_DIC_3y_diff_{mask_name}").drop_vars(['obs', 'time']))
                result_ds_list.append(ds.cs_DIC_residual[mask, 365*3].rename(f"cum_DIC_3y_residual_{mask_name}").drop_vars(['obs', 'time']))
            elif "2y" in mask_name:
                result_ds_list.append(ds.cs_DIC_total[mask, 365*2].rename(f"cum_DIC_2y_total_{mask_name}").drop_vars(['obs', 'time'])) 
                result_ds_list.append(ds.cs_DIC_bio[mask, 365*2].rename(f"cum_DIC_2y_bio_{mask_name}").drop_vars(['obs', 'time']))
                result_ds_list.append(ds.cs_DIC_bio_soft[mask, 365*2].rename(f"cum_DIC_2y_bio_soft_{mask_name}").drop_vars(['obs', 'time']))
                result_ds_list.append(ds.cs_DIC_bio_carbonate[mask, 365*2].rename(f"cum_DIC_2y_bio_carbonate_{mask_name}").drop_vars(['obs', 'time']))
                result_ds_list.append(ds.cs_DIC_diff[mask, 365*2].rename(f"cum_DIC_2y_diff_{mask_name}").drop_vars(['obs', 'time']))
                result_ds_list.append(ds.cs_DIC_residual[mask, 365*2].rename(f"cum_DIC_2y_residual_{mask_name}").drop_vars(['obs', 'time']))
            else:
                result_ds_list.append(ds.cs_DIC_total[mask, 365*1].rename(f"cum_DIC_1y_total_{mask_name}").drop_vars(['obs', 'time'])) 
                result_ds_list.append(ds.cs_DIC_bio[mask, 365*1].rename(f"cum_DIC_1y_bio_{mask_name}").drop_vars(['obs', 'time']))
                result_ds_list.append(ds.cs_DIC_bio_soft[mask, 365*1].rename(f"cum_DIC_1y_bio_soft_{mask_name}").drop_vars(['obs', 'time']))
                result_ds_list.append(ds.cs_DIC_bio_carbonate[mask, 365*1].rename(f"cum_DIC_1y_bio_carbonate_{mask_name}").drop_vars(['obs', 'time']))
                result_ds_list.append(ds.cs_DIC_diff[mask, 365*1].rename(f"cum_DIC_1y_diff_{mask_name}").drop_vars(['obs', 'time']))
                result_ds_list.append(ds.cs_DIC_residual[mask, 365*1].rename(f"cum_DIC_1y_residual_{mask_name}").drop_vars(['obs', 'time']))
            ds[mask_name] = mask

        result_ds = xr.merge(result_ds_list)
        result_ds.attrs['description'] = "DIC changes in EDW and in different EDW masks"
        result_ds.attrs['run'] = basename

        print(f"Saving {basename}")
        result_ds.to_netcdf(output_dir_DIC + basename + ".nc")
        result_ds.close()

    ds.close()
    


if __name__ == '__main__':
    print("Started script")
    for run in forward_september_runs:
        analysis(run, mode='forward', mask_export=True, DIC_export=True)
    for run in backward_september_runs:
        analysis(run, mode='backward', mask_export=True, DIC_export=True)
    for run in forward_march_runs:
        analysis(run, mode='forward', mask_export=True, DIC_export=True)
    for run in backward_march_runs:
        analysis(run, mode='backward', mask_export=True, DIC_export=True)
