import xarray as xr
import numpy as np
import gsw


def preprocess(ds, reverse_time=None, in_edw=True, fluxes=True, entry_exit=False, 
               cumsums=True, sigma=True, in_situ_density=True, single_time=True,
               redundancy_removal=True, convert_flags_to_bool=True):
    """
    Preprocess a dataset containing Lagrangian trajectories, by adding biogeochemical fluxes,
    cumulative sums, and densities. Automatically takes into account whether the run is backwards in time.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the trajectories
    reverse_time : bool, optional
        Whether the run is backwards in time, by default None, so the sign is infered.
        If True, the DDIC fluxes are reversed.
    in_edw : bool, optional
        Whether to only keep the part of the trajectory that starts in the EDW, by default True
    fluxes : bool, optional
        Whether to compute biogeochemical fluxes, by default True
    entry_exit : bool, optional
        LEGACY: Whether to compute entry and exit events, by default False. 
        This is now done in postprocessing.
    cumsums : bool, optional
        Whether to compute cumulative sums, by default True. Note that cumsums backwards in time
        along the obs dimension should be interpreted as total increase in DIC between obs = 0 and obs = i.
        Along the time-dimension, this is the total increase in DIC between time t_i and t_end. 
    sigma : bool, optional
        Whether to compute sigma0, by default True
    in_situ_density : bool, optional
        Whether to compute in-situ density, by default True
    single_time : bool, optional
        Whether to create an array with a time dimension that has no trajectory associated with it, by default True
    redundancy_removal : bool, optional
        Whether to remove variables that are no longer moved, to save space
    convert_flags_to_bool : bool, optional
        Some flags from the Parcels run are unnecessarily a float. Convert these to bools?

    Returns
    -------
    xarray.Dataset
        Dataset containing the preprocessed trajectories

    """
    if single_time:
        traj_nonan_till_end = int(np.isfinite(ds.isel(obs=-1).time).where(np.isfinite(ds.isel(obs=-1).time)).dropna('trajectory').trajectory[0])
        ds.time.sel(trajectory=traj_nonan_till_end).drop('trajectory')
        ds = ds.assign_coords(time=ds.time.sel(trajectory=traj_nonan_till_end).drop('trajectory'))
    if in_edw:
        start_in_EDW_Lagrangian = (ds.EDW_Lagrangian.isel(obs=0) == 1).drop('obs')
        keeptraj = start_in_EDW_Lagrangian.where(start_in_EDW_Lagrangian).dropna(dim='trajectory').trajectory
        ds = ds.sel(trajectory=keeptraj)

    if fluxes:
        odt_days = np.abs(float(np.diff(ds.isel(trajectory=0, obs=slice(0,3)).time.data)[0]) / 1e9 / 60 / 60 / 24)
        assert odt_days > 0 and odt_days < 1000, "odt_days is not in a reasonable range"

        time_sign = np.sign(float(np.diff(ds.isel(trajectory=0, obs=slice(0,3)).time.data)[0]))

        if reverse_time is True:
            time_sign = -1
        elif reverse_time is False:
            time_sign = 1


        # Compute fluxes using a forward difference. Takes into account the time step and time sign.
        ds['DDIC'] = time_sign * (xr.concat([ds["DIC"].isel(obs=0).expand_dims('obs').T*0, ds["DIC"].diff("obs")], dim='obs') / odt_days).astype(np.float32)
        ds['DPO4'] = time_sign * xr.concat([ds["PO4"].isel(obs=0).expand_dims('obs').T*0, ds["PO4"].diff("obs")], dim='obs') / odt_days
        ds['DNO3'] = time_sign * xr.concat([ds["NO3"].isel(obs=0).expand_dims('obs').T*0, ds["NO3"].diff("obs")], dim='obs') / odt_days
        ds['DALK'] = time_sign * xr.concat([ds["ALK"].isel(obs=0).expand_dims('obs').T*0, ds["ALK"].diff("obs")], dim='obs') / odt_days

        # The diff fluxes are always forward in time, so their sign does not need to be reversed.
        ds['DPO4_bio'] = ds['DPO4'] - ds['DPO4_diff']
        ds['DNO3_bio'] = ds['DNO3'] - ds['DNO3_diff']
        ds['DALK_bio'] = ds['DALK'] - ds['DALK_diff']

        # set redfield ratios
        r_cp = 122
        r_np = 16
        r_cn = 122/16

        # use full expression with phosphate, since stoichiometry is not always met
        ds['DDIC_bio_soft'] = r_cp * ds['DPO4_bio']
        ds['DDIC_bio_carbonate'] = 0.5 * (ds['DALK_bio'] + ds['DNO3_bio'])

        ds['DDIC_bio'] = ds['DDIC_bio_soft'] + ds['DDIC_bio_carbonate']

        # computing the residual flux as the residual
        ds['DDIC_residual'] = ds['DDIC'] - ds['DDIC_bio'] - ds['DDIC_diff']

    if entry_exit:
        # entry and exit identification
        # LEGACY: computing this fully in postprocessing now
        ds["lower_T_boundary_entry"] = (ds.EDW_entryevent.astype(int) & 1).astype(bool)
        ds["upper_T_boundary_entry"] = (ds.EDW_entryevent.astype(int) & 2).astype(bool)
        ds["stratification_entry"] = (ds.EDW_entryevent.astype(int) & 4).astype(bool)
        ds["region_entry"] = (ds.EDW_entryevent.astype(int) & 8).astype(bool)

        ds["lower_T_boundary_exit"] = (ds.EDW_exitevent.astype(int) & 1).astype(bool)
        ds["upper_T_boundary_exit"] = (ds.EDW_exitevent.astype(int) & 2).astype(bool)
        ds["stratification_exit"] = (ds.EDW_exitevent.astype(int) & 4).astype(bool)
        ds["region_exit"] = (ds.EDW_exitevent.astype(int) & 8).astype(bool)

    if cumsums:
        # cumsums
        ds['cs_DIC_total'] = ds.DDIC.cumsum('obs')
        ds['cs_DIC_bio'] = ds.DDIC_bio.cumsum('obs')
        ds['cs_DIC_bio_soft'] = ds.DDIC_bio_soft.cumsum('obs')
        ds['cs_DIC_bio_carbonate'] = ds.DDIC_bio_carbonate.cumsum('obs')
        ds['cs_DIC_diff'] = ds.DDIC_diff.cumsum('obs')
        ds['cs_DIC_residual'] = ds.DDIC_residual.cumsum('obs')

    if sigma or in_situ_density:
        p = gsw.p_from_z(-ds.z, ds.lat)
        SA = gsw.SA_from_SP(ds.S, p, ds.lon, ds.lat)
        CT = gsw.CT_from_pt(SA, ds.T)
        if sigma:
            ds["sigma0"] = gsw.sigma0(SA, CT).astype(np.float32)

    if in_situ_density:
        ds["rho"] = gsw.rho(SA, CT, p).astype(np.float32)

    if redundancy_removal:
        ds = ds.drop_vars(["MLD_entryevent", "MLD_exitevent", "EDW_entryevent", "EDW_exitevent"])

    if convert_flags_to_bool:
        ds["EDW_part_of_biggest_blob"] = ds["EDW_part_of_biggest_blob"].fillna(0).astype(bool)
        ds["EDW_part_of_smaller_blob"] = ds["EDW_part_of_smaller_blob"].fillna(0).astype(bool)
        ds["EDW_part_of_outcropping_blob"] = ds["EDW_part_of_outcropping_blob"].fillna(0).astype(bool)
        ds["EDW_outcropping_column_mask"] = ds["EDW_outcropping_column_mask"].fillna(0).astype(bool)
        ds["EDW_Lagrangian"] = ds["EDW_Lagrangian"].fillna(0).astype(bool)
        ds["EDW_Eulerian"] = ds["EDW_Eulerian"].fillna(0).astype(bool)

    return ds


def create_EDW_criteria_masks(ds, test=False, last_sigma0=True, last_rho=True, sequestration=True, sequestration_threshold=0.05):
    """
    Create masks for the different criteria for the EDW

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the trajectories
    test : bool, optional
        Whether to test if all EDW entries are accounted for, by default False
    last_sigma0 : bool, optional
        Whether to compute the last sigma0 value in the EDW, for locations where
        the particle is not in the EDW anymore, by default True
    last_rho : bool, optional
        Whether to compute the last in-situ density value in the EDW, for locations where
        the particle is not in the EDW anymore, by default True

    Returns
    -------
    xarray.Dataset
        Dataset containing the masks

    """

    # !!! BEWARE THAT np.array([np.nan]).astype(bool) gives array([True]) !!!
    ds["in_temp_range"] = (ds.T >= 17) * (ds.T <= 20.5)
    ds["below_temp_range"] = (ds.T < 17)
    ds["above_temp_range"] = (ds.T > 20.5)

    # 2023-07-27: THIS IS ASSUMING dTdz has the wrong sign
    ds["in_strat_range"] = (np.abs(ds.dTdz) <= 0.01)

    ds["in_edw"] = ds["in_temp_range"] * ds["in_strat_range"]

    ds["thick_enough"] = ds["EDW_layer_thickness"] > 50
    ds["in_edw_strict"] = ds["in_edw"] * ((ds.EDW_part_of_biggest_blob > 0.1) +
                                            (ds.EDW_part_of_smaller_blob > 0.1)) * ds["thick_enough"]                          

    ds["in_mixing_layer"] = ds.z <= ds.MLDturb
    ds["in_mixed_layer"] = ds.z <= ds.MLDtemp

    ds["out_of_bounds"] = np.isnan(ds.DIC)

    # temp_range_crossings
    ds["temp_range_crossing"] = ds["in_temp_range"].astype(int).diff('obs').fillna(0).astype(bool)  # does not count NaNs
    ds["temp_range_entry_from_below"] = (ds["below_temp_range"].astype(int).diff('obs') == -1).fillna(0).astype(bool)  # counts NaNs
    ds["temp_range_exit_to_below"] = (ds["below_temp_range"].astype(int).diff('obs') == 1).fillna(0).astype(bool)  # counts NaNs
    ds["temp_range_entry_from_above"] = (ds["above_temp_range"].astype(int).diff('obs') == -1).fillna(0).astype(bool)  # counts NaNs
    ds["temp_range_exit_to_above"] = (ds["above_temp_range"].astype(int).diff('obs') == 1).fillna(0).astype(bool)  # counts NaNs

    # strat_range_crossings
    ds["strat_range_entry"] = (ds["in_strat_range"].astype(int).diff('obs') == 1).fillna(0).astype(bool)  # counts NaNs
    ds["strat_range_exit"] = (ds["in_strat_range"].astype(int).diff('obs') == -1).fillna(0).astype(bool)  # counts NaNs

    # mixing_layer_crossings
    ds["mixing_layer_entry"] = (ds["in_mixing_layer"].astype(int).diff('obs') == 1).fillna(0).astype(bool)  # counts NaNs
    ds["mixing_layer_exit"] = (ds["in_mixing_layer"].astype(int).diff('obs') == -1).fillna(0).astype(bool)  # counts NaNs

    # mixed_layer_crossings
    ds["mixed_layer_entry"] = (ds["in_mixed_layer"].astype(int).diff('obs') == 1).fillna(0).astype(bool)  # counts NaNs
    ds["mixed_layer_exit"] = (ds["in_mixed_layer"].astype(int).diff('obs') == -1).fillna(0).astype(bool) # counts NaNs

    # thickness_regime_crossings
    ds["thickness_regime_entry"] = (ds["thick_enough"].astype(int).diff('obs') == 1).fillna(0).astype(bool)  # counts NaNs
    ds["thickness_regime_exit"] = (ds["thick_enough"].astype(int).diff('obs') == -1).fillna(0).astype(bool)  # counts NaNs

    # edw_crossings
    ds["edw_entry"] = (ds["in_edw_strict"].astype(int).diff('obs') == 1).fillna(0).astype(bool)  # counts NaNs
    ds["edw_exit"] = (ds["in_edw_strict"].astype(int).diff('obs') == -1).fillna(0).astype(bool)  # counts NaNs

    # out_of_bounds_crossings
    ds["goes_out_of_bounds"] = (ds["out_of_bounds"].astype(int).diff('obs') == 1).fillna(0).astype(bool)  # counts NaNs

    if last_sigma0:
        EDW_mask = ds.in_edw_strict.copy()
        EDW_mask.loc[dict(obs=0)] = True
        ds["last_sigma0"] = ds.sigma0.where(EDW_mask).ffill('obs').where(~EDW_mask).astype(np.float32)

    if last_rho:
        EDW_mask = ds.in_edw_strict.copy()
        EDW_mask.loc[dict(obs=0)] = True
        ds["last_rho"] = ds.rho.where(EDW_mask).ffill('obs').where(~EDW_mask).astype(np.float32)

    if sequestration:
        if not last_sigma0:
            raise ValueError("last_sigma0 must be True to compute sequestration")
        heavier_than_last = (ds.sigma0 >= ds.last_sigma0 - sequestration_threshold)
        lighter_than_last = (ds.sigma0 < ds.last_sigma0 - sequestration_threshold)

        codes = (heavier_than_last*0 + 1).where(heavier_than_last).fillna(0) + \
                (lighter_than_last*0 + 2).where(lighter_than_last).fillna(0) + \
                (EDW_mask * 0 + 3).where(EDW_mask).fillna(0)
        # Forward fill codes onto indices from heavier_than_last. This shows what preceded the heavier_than_last code label.
        # Then select only positions that originally had heavier_than_last and lighter_than_last
        # Then select only code 3, since this means its
        # a heavier_than_last that was preceded by EDW
        heavier_out_of_EDW = (codes.where(codes >= 2).ffill(dim='obs').where(codes < 3) == 3)
        ds["sequestered"] = heavier_out_of_EDW * ~ds['in_mixing_layer']

    if test:
        entries = ds.temp_range_entry_from_below * ds.edw_entry + \
            ds.temp_range_entry_from_above * ds.edw_entry + \
            ds.strat_range_entry * ds.edw_entry + \
            ds.thickness_regime_entry * ds.edw_entry

        assert np.sum(ds["edw_entry"].isel(obs=slice(1, None)).astype(bool)) == np.sum(
            entries.isel(obs=slice(1, None)).astype(bool)), "not all EDW entries are accounted for"

    return ds
