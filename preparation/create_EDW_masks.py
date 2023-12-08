import numpy as np
import bottleneck as bn
import xarray as xr
from glob import glob
import tqdm
import argparse
from scipy import ndimage
import datetime

meshdir = "/nethome/4302001/local_data/mesh/"
mesh_z = xr.open_dataset(f"{meshdir}mesh_zgr_PSY4V2_deg_NA_GoM_subset.nc")
mesh_h = xr.open_dataset(f"{meshdir}mesh_hgr_PSY4V2_deg_NA_GoM_subset.nc")
mesh = xr.merge([mesh_h, mesh_z]).isel(x=slice(70, 290), y=slice(50, 250))

cum_thickness = mesh.e3t_1d.cumsum('z')


def load_temperature(date_string):
    dirread = "/nethome/4302001/local_data/FREEGLORYS2V4/"
    file = f"{dirread}freeglorys2v4-NorthAtlanticGoM-daily_T_{date_string}.nc"
    ds_in = xr.open_dataset(file).isel(x=slice(70, 290), y=slice(50, 250))
    return ds_in


temp_2005_03_01 = load_temperature("2005-03-01")

cell_volume = (mesh.e3t_0 * mesh.e2t * mesh.e1t)
cell_volume = cell_volume.where(cell_volume < 200e3 * 200e3 * 500e3).where(cell_volume >= 0)
cell_volume = cell_volume.rename({"z": "deptht"}).assign_coords(deptht=temp_2005_03_01.deptht) * (temp_2005_03_01.votemper * 0 + 1)


def identify_edw_model(ds_T, strat_crit, temp_crit):
    T_grad = ds_T.votemper.differentiate(coord='deptht')

    criterion_temperature = (ds_T.votemper > temp_crit[0]) * (ds_T.votemper < temp_crit[1])
    criterion_strat = np.fabs(T_grad) < strat_crit
    edw_criterion = (criterion_strat * criterion_temperature).rename("EDW_criterion").assign_attrs(
        {"strat_criterion": str(strat_crit), "temp_criterion": str(temp_crit)})
    cell_thickness_3d = mesh_z.e3t_0.rename({'z': 'deptht'}).assign_coords(deptht=edw_criterion.deptht) * (edw_criterion * 0 + 1)
    EDW_total_thickness = cell_thickness_3d.where(edw_criterion).sum(dim='deptht', skipna=True).rename("EDW_total_thickness")

    ds_edw = xr.merge([edw_criterion, EDW_total_thickness])

    return ds_edw


def compute_thicknesses_and_outcropping(EDW_mask, cum_thickness=cum_thickness):
    """
    Compute the thickness of EDW and the outcropping mask

    Parameters
    ----------
    EDW_mask : xr.DataArray
        Boolean array indicating the EDW
    cum_thickness : xr.DataArray
        Cumulative thickness of the depth layers

    Returns
    -------
    layer_thickness_3D : xr.DataArray
        Thickness of EDW in each layer, as a 3D array  
    outcropping_column_mask : xr.DataArray
        Boolean array indicating whether the cell is part of an outcropping column
    """
    outcropping_column_mask = None
    
    mask_diff = EDW_mask.astype(int).diff('deptht', label='lower')
    n_layers = np.ceil(np.abs(mask_diff).sum('deptht')/2).max()
    layer_thickness_3D_arr = (EDW_mask.values * 0).astype(np.float32)
    for layer in range(int(n_layers)):
        layer_bottom_idx = mask_diff.argmin('deptht')
        layer_top_idx = mask_diff.argmax('deptht')

        if layer == 0:
            # add the surface layer if EDW reaches to the surface
            layer_top_idx = xr.where(layer_top_idx > layer_bottom_idx, 0, layer_top_idx)
            outcropping_column_mask = (layer_top_idx == 0) * (layer_bottom_idx > 0)
        layer_mask = xr.where((EDW_mask.deptht >= EDW_mask.deptht.isel(deptht=np.maximum(layer_top_idx+1, 0))) &
                              (EDW_mask.deptht <= EDW_mask.deptht.isel(deptht=layer_bottom_idx)),
                              1, 0)

        thickness = cum_thickness.isel(z=layer_bottom_idx) - cum_thickness.isel(z=layer_top_idx)

        if layer == 0:
            # add the surface layer if EDW reaches to the surface
            layer_mask = layer_mask + xr.where(EDW_mask.deptht == EDW_mask.deptht.isel(deptht=0), EDW_mask, 0)
            outcropping_column_mask = layer_mask * outcropping_column_mask
            thickness = thickness + cum_thickness.isel(z=0)

        layer_thickness_3D_arr += (layer_mask * thickness).values
        mask_diff = xr.where(mask_diff.deptht > mask_diff.isel(deptht=layer_bottom_idx).deptht, mask_diff, 0)

    layer_thickness_3D = xr.DataArray(layer_thickness_3D_arr, dims=['deptht', 'y', 'x'], coords={
                                      'deptht': EDW_mask.deptht, 'y': EDW_mask.y, 'x': EDW_mask.x})

    return layer_thickness_3D, outcropping_column_mask


def compute_edw_volume(edw_criterion, cell_volume=cell_volume):
    """
    Compute the volume of EDW

    Parameters
    ----------
    edw_criterion : xr.DataArray
        Boolean array indicating the EDW
    cell_volume : xr.DataArray
        Volume of each cell

    Returns
    -------
    float
        Volume of EDW in m^3
    """
    edw_volume = (edw_criterion * cell_volume).sum(skipna=True)
    return float(edw_volume)


def contiguity_label(mask, outcropping_column_mask=None, minvolume=1e11):
    """
    Label contiguous regions in an EDW mask

    Parameters
    ----------
    mask : xr.DataArray
        Boolean array indicating the EDW
    outcropping_column_mask : xr.DataArray
        Boolean array indicating whether the cell is part of an outcropping column
    minvolume : float
        Minimum volume of a blob to be considered (in m^3).
        Default is 1e11 m^3 (about 0.3% of EDW volume in winter)
    """
    # Remove Madeira mode water
    mask = xr.where(mask.nav_lon < -35, mask, False).transpose('deptht', 'y', 'x', )

    # Identify contiguous regions
    labeled_EDW, nlabels = ndimage.label(mask)
    labeled_EDW = xr.DataArray(labeled_EDW, dims=['deptht', 'y', 'x'], coords={'deptht': mask.deptht, 'y': mask.y, 'x': mask.x})

    # Sort the labels by number of cells
    labels, counts = np.unique(labeled_EDW, return_counts=True)
    sorted_volumes = np.zeros_like(counts)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_labels = labels[sorted_indices]
    sorted_counts = counts[sorted_indices]

    # Compute the volumes
    for idx, label in enumerate(sorted_labels):
        sorted_volumes[idx] = cell_volume.where(labeled_EDW == label).sum(skipna=True)

    # Check which blobs are big enough
    EDW_blobs_big_enough = sorted_volumes > minvolume

    # Identify the biggest blob (skip the first one because it is the background)
    part_of_biggest_blob = labeled_EDW == sorted_labels[EDW_blobs_big_enough][1]
    if outcropping_column_mask is not None:
        part_of_outcropping_blob = outcropping_column_mask * False
        if (part_of_biggest_blob * outcropping_column_mask).any():
            part_of_outcropping_blob += part_of_biggest_blob
    # Identify smaller blobs
    part_of_smaller_blob = part_of_biggest_blob * False
    if sorted_labels[EDW_blobs_big_enough].size > 2:
        for label in sorted_labels[EDW_blobs_big_enough][2:]:
            blob = (labeled_EDW == label)
            if outcropping_column_mask is not None:
                if (blob * outcropping_column_mask).any():
                    part_of_outcropping_blob += blob
            part_of_smaller_blob = part_of_smaller_blob + blob

    if outcropping_column_mask is not None:
        return part_of_biggest_blob, part_of_smaller_blob, part_of_outcropping_blob
    else:
        return part_of_biggest_blob, part_of_smaller_blob


def create_edw_masks_file(strat_crit, temp_crit, extras=["thicknesses", "contiguity"]):
    """
    Create files with EDW masks and associated information.

    Parameters
    ----------
    strat_crit : float
        Maximum stratification criterion
    temp_crit : list
        Minimum and maximum temperature criterion
    extras : list
        List of extra variables to compute. Can be "thicknesses" and/or "contiguity".
    """
    dirread = "/nethome/4302001/local_data/FREEGLORYS2V4/"
    dirwrite = "/nethome/4302001/local_data/FREEGLORYS2V4_EDW/"

    glorys_files = sorted(glob(dirread + 'freeglorys2v4-NorthAtlanticGoM-daily_T_*.nc'))
    # glorys_files = [dirread + 'freeglorys2v4-NorthAtlanticGoM-daily_T_1994-09-27.nc']
    for file in tqdm.tqdm(glorys_files):
        base_name = "freeglorys2v4-NorthAtlanticGoM-daily"
        date = file[-13:-3]
        try:
            ds_in = xr.open_dataset(file).isel(x=slice(70, 290), y=slice(50, 250))
            ds_out = identify_edw_model(ds_in, strat_crit, temp_crit)

            ds_out.attrs["EDW_total_volume"] = compute_edw_volume(ds_out.EDW_criterion)
            EDW_excl_madeira = xr.where(ds_out.nav_lon <= -35, ds_out.EDW_criterion, False).transpose('deptht', 'y', 'x', )
            EDW_excl_madeira_excl_thin = xr.where(ds_out.EDW_total_thickness <= 50, EDW_excl_madeira, False).transpose('deptht', 'y', 'x', )
            ds_out.attrs["EDW_volume_excl_Madeira"] = compute_edw_volume(EDW_excl_madeira)
            ds_out.attrs["EDW_volume_excl_Madeira_excl_thin"] = compute_edw_volume(EDW_excl_madeira_excl_thin)

            if "thicknesses" in extras:
                layer_thickness_3D, outcropping_column_mask = compute_thicknesses_and_outcropping(EDW_excl_madeira,
                                                                                              cum_thickness)
                if outcropping_column_mask is not None:
                    ds_out = xr.merge([ds_out, layer_thickness_3D.rename("EDW_layer_thickness"),
                                   outcropping_column_mask.rename("EDW_outcropping_column_mask")])
                else:
                    ds_out = xr.merge([ds_out, layer_thickness_3D.rename("EDW_layer_thickness")])

            if "contiguity" in extras:
                part_of_biggest_blob, part_of_smaller_blob, part_of_outcropping_blob = contiguity_label(EDW_excl_madeira,
                                                                                                    outcropping_column_mask)
                if outcropping_column_mask is not None:
                    ds_out = xr.merge([ds_out, part_of_biggest_blob.rename("EDW_part_of_biggest_blob"),
                                       part_of_smaller_blob.rename("EDW_part_of_smaller_blob"),
                                       part_of_outcropping_blob.rename("EDW_part_of_outcropping_blob")])
                else:
                    ds_out = xr.merge([ds_out, part_of_biggest_blob.rename("EDW_part_of_biggest_blob"),
                                   part_of_smaller_blob.rename("EDW_part_of_smaller_blob")])
                ds_out.attrs["EDW_biggest_blob_volume"] = compute_edw_volume(ds_out.EDW_part_of_biggest_blob)
                ds_out.attrs["EDW_smaller_blob_volume"] = compute_edw_volume(ds_out.EDW_part_of_smaller_blob)

            ds_in.close()

            ds_out.attrs["Created on:"] = str(datetime.datetime.now())

            ds_out.to_netcdf(f"{dirwrite}{base_name}_EDW_{date}_s{strat_crit}-t{temp_crit[0]}_{temp_crit[1]}.nc")
            ds_out.close()
        except:
            print(f"Error with {file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute EDW masks.')
    parser.add_argument('--sc', metavar='Stratification_criterion', type=float, default=0.01, help='Stratification criterion')
    parser.add_argument('--tc', metavar='Temperature_range_crition1 Temperature_range_criterion2',
                        type=float, nargs=2, default=[17.0, 20.5], help='Temperature range criterion')

    args = parser.parse_args()

    create_edw_masks_file(strat_crit=args.sc, temp_crit=args.tc)
