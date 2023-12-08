import numpy as np
import xarray as xr

from glob import glob
from datetime import timedelta as delta
from datetime import datetime as time
import sys
import os

import argparse

import parcels

sys.path.append("/nethome/4302001/tracer_backtracking/simulations/kernels")
import EDW_Sampler_fluxes
import default_kernels

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="Advect BGC sampling particles in the North Atlantic")
    parser.add_argument('-T0', default="1992-03-01", type=str, help="Particle initialization time. Must be formatted as YYYY-MM-DD.")
    parser.add_argument('-T', default=120, type=int, help='Simulation time (days)')
    parser.add_argument('-dt', default=3*60, type=int, help='Advection timestep for advection in minutes')
    parser.add_argument('-odt', '--outputdt', default=24, type=int, help='Output timestep in hours')
    parser.add_argument('-tinterp', '--tracer_interp_method', default='linear_invdist_land_tracer',
                        type=str, help='Set tracer interpolation method')
    parser.add_argument('--backwards', action='store_true', help='Run the simulation in backwards mode (False by default)')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    # -------- SETTINGS ---------
    if args.backwards:
        dt = -args.dt        # minutes
    else:
        dt = args.dt         # minutes
    T0 = args.T0             # particle release date
    T = args.T               # simulation time
    yr = T0[0:4]
    mon = T0[5:7]
    output_dt = args.outputdt  # output_dt
    tracer_interp_method = args.tracer_interp_method

    assert (output_dt * 60) % np.abs(dt) == 0, "Output timestep must be a multiple of the advection timestep"

    print(
        f"Simulation settings: \n T0 = {T0} \n T = {T}\n dt = {dt}\n output_dt = {output_dt}\n tracer_interp_method = {tracer_interp_method}\n backwards = {args.backwards}\n")

    # ------ FIELDSET CONSTRUCTION --------

    vars_phys_3D = ["U", "V", "W", "S", "T", "KZlog10"]
    vars_phys_2D = ["MLDturb", "MLDtemp", "MLDdens"]
    vars_bio_3D = ["ALK", "CHL", "DIC", "NO3", "NPPV", "O2", "PO4"]

    assert tracer_interp_method in ["linear", "nearest",
                                    "linear_invdist_land_tracer", "cgrid_velocity", "cgrid_tracer", "bgrid_velocity"]

    # Set paths
    data_dir = "/nethome/4302001/local_data/"

    path_phy = data_dir + "FREEGLORYS2V4/"
    path_bio = data_dir + "FREEBIORYS2V4/"
    path_edw = data_dir + "FREEGLORYS2V4_EDW/"
    path_fluxes = data_dir + "FREEBIORYS2V4_fluxes/"

    mesh_horiz = data_dir + "mesh/mesh_hgr_PSY4V2_deg_NA_GoM_subset.nc"
    mesh_verti = data_dir + "mesh/mesh_zgr_PSY4V2_deg_NA_GoM_subset.nc"

    # Physics files
    ufiles = sorted(
        [path_phy + f"freeglorys2v4-NorthAtlanticGoM-daily_U_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    vfiles = sorted(
        [path_phy + f"freeglorys2v4-NorthAtlanticGoM-daily_V_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    wfiles = sorted(
        [path_phy + f"freeglorys2v4-NorthAtlanticGoM-daily_W_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    tfiles = sorted(
        [path_phy + f"freeglorys2v4-NorthAtlanticGoM-daily_T_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    sfiles = sorted(
        [path_phy + f"freeglorys2v4-NorthAtlanticGoM-daily_S_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    kzfiles = sorted(
        [path_phy + f"freeglorys2v4-NorthAtlanticGoM-daily_KZLN10_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    phy2dfiles = sorted(
        [path_phy + f"freeglorys2v4-NorthAtlanticGoM-daily_2D_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])

    edwfiles = sorted(
        [path_edw + f"freeglorys2v4-NorthAtlanticGoM-daily_EDW_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}_s0.01-t17.0_20.5.nc" for i in range(-1, T + 1)])

    for file in ufiles + vfiles + wfiles + tfiles + sfiles + kzfiles + phy2dfiles + edwfiles:
        assert os.path.isfile(file), "File not found: " + file

    # BGC files
    alkfiles = sorted(
        [path_bio + f"freebiorys2v4-NorthAtlanticGoM-daily_alk_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    chlfiles = sorted(
        [path_bio + f"freebiorys2v4-NorthAtlanticGoM-daily_chl_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    dicfiles = sorted(
        [path_bio + f"freebiorys2v4-NorthAtlanticGoM-daily_dic_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    no3files = sorted(
        [path_bio + f"freebiorys2v4-NorthAtlanticGoM-daily_no3_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    nppvfiles = sorted(
        [path_bio + f"freebiorys2v4-NorthAtlanticGoM-daily_nppv_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    o2files = sorted(
        [path_bio + f"freebiorys2v4-NorthAtlanticGoM-daily_o2_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    po4files = sorted(
        [path_bio + f"freebiorys2v4-NorthAtlanticGoM-daily_po4_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])

    dALKdt_difffiles = sorted(
        [path_fluxes + f"freebiorys2v4-NorthAtlanticGoM-daily_flux_difv_alk_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    dDICdt_difffiles = sorted(
        [path_fluxes + f"freebiorys2v4-NorthAtlanticGoM-daily_flux_difv_dic_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    dNO3dt_difffiles = sorted(
        [path_fluxes + f"freebiorys2v4-NorthAtlanticGoM-daily_flux_difv_no3_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    dPO4dt_difffiles = sorted(
        [path_fluxes + f"freebiorys2v4-NorthAtlanticGoM-daily_flux_difv_po4_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])
    # dO2dt_difffiles = sorted([path_fluxes + f"freebiorys2v4-NorthAtlanticGoM-daily_flux_difv_o2_{np.datetime64(T0) + np.timedelta64(np.sign(dt) * i, 'D')}.nc" for i in range(-1, T + 2)])

    for file in alkfiles + chlfiles + dicfiles + no3files + nppvfiles + o2files + po4files + dALKdt_difffiles + dDICdt_difffiles + dNO3dt_difffiles + dPO4dt_difffiles:
        assert os.path.isfile(file), "File not found: " + file

    vars_phys_3D = ["U", "V", "W", "S", "T", "KZlog10"]
    vars_phys_2D = ["MLDturb", "MLDtemp", "MLDdens"]
    vars_bio_3D = ["ALK", "CHL", "DIC", "NO3", "NPPV", "O2", "PO4"]
    vars_bio_fluxes = ["dALKdt_diff", "dDICdt_diff", "dNO3dt_diff", "dPO4dt_diff",
                       #    "dO2dt_diff"
                       ]

    filenames_phy = {'U': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': ufiles},
                     'V': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': vfiles},
                     'W': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': wfiles},
                     'S': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': sfiles},
                     'T': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': tfiles},
                     'KZlog10': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': kzfiles},
                     'MLDturb': {'lon': mesh_horiz, 'lat': mesh_horiz, 'data': phy2dfiles},
                     'MLDtemp': {'lon': mesh_horiz, 'lat': mesh_horiz, 'data': phy2dfiles},
                     'MLDdens': {'lon': mesh_horiz, 'lat': mesh_horiz, 'data': phy2dfiles},
                     }
    filenames_bio = {'ALK': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': alkfiles},
                     'CHL': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': chlfiles},
                     'DIC': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': dicfiles},
                     'NO3': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': no3files},
                     'NPPV': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': nppvfiles},
                     'O2': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': o2files},
                     'PO4': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': po4files},
                     'dALKdt_diff': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': dALKdt_difffiles},
                     'dDICdt_diff': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': dDICdt_difffiles},
                     'dNO3dt_diff': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': dNO3dt_difffiles},
                     'dPO4dt_diff': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': dPO4dt_difffiles},
                     # 'dO2dt_diff' : {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_verti, 'data': dO2dt_difffiles},
                     }

    filenames = {**filenames_phy, **filenames_bio}

    variables_phys = {'U': 'vozocrtx',
                      'V': 'vomecrty',
                      'W': 'vovecrtz',
                      'T': 'votemper',
                      'S': 'vosaline',
                      'KZlog10': 'vokzln10',
                      'MLDturb': 'somxlavt',
                      'MLDtemp': 'somxlt02',
                      'MLDdens': 'somxl010'
                      }
    variables_bio = {"ALK": "alk",
                     "CHL": "chl",
                     "DIC": "dic",
                     "NO3": "no3",
                     "NPPV": "nppv",
                     "O2": "o2",
                     "PO4": "po4",
                     "dALKdt_diff": "alk_difv_flux",
                     "dDICdt_diff": "dic_difv_flux",
                     "dNO3dt_diff": "no3_difv_flux",
                     "dPO4dt_diff": "po4_difv_flux",
                     # "dO2dt_diff" : "o2_difv_flux"
                     }
    variables = {**variables_phys, **variables_bio}

    dimensions = {}
    for var in vars_phys_3D + vars_bio_3D + vars_bio_fluxes:
        if var in ["U", "V", "W"]:
            dimensions[var] = {'lon': 'glamf', 'lat': 'gphif', 'depth': 'gdepw_1d', 'time': 'time_counter'}
        else:
            # For now leave all depths at gdepw. Later check if gdept should be used for normal tracers. Check this with Erik.
            # update 2023-03-14: only velocities should use f-points, all other variables should use t-points
            dimensions[var] = {'lon': 'glamt', 'lat': 'gphit', 'depth': 'gdept_1d', 'time': 'time_counter'}

    for var in vars_phys_2D:
        dimensions[var] = {'lon': 'glamt', 'lat': 'gphit', 'time': 'time_counter'}

    fieldset = parcels.FieldSet.from_nemo(filenames, variables, dimensions)

    fieldset.S.interp_method = tracer_interp_method
    fieldset.T.interp_method = tracer_interp_method
    fieldset.KZlog10.interp_method = tracer_interp_method
    fieldset.MLDturb.interp_method = tracer_interp_method
    fieldset.MLDtemp.interp_method = tracer_interp_method
    fieldset.MLDdens.interp_method = tracer_interp_method

    # fieldset.EDW.interp_method = "nearest"

    fieldset.ALK.interp_method = tracer_interp_method
    fieldset.CHL.interp_method = tracer_interp_method
    fieldset.DIC.interp_method = tracer_interp_method
    fieldset.NO3.interp_method = tracer_interp_method
    fieldset.NPPV.interp_method = tracer_interp_method
    fieldset.O2.interp_method = tracer_interp_method
    fieldset.PO4.interp_method = tracer_interp_method

    fieldset.dALKdt_diff.interp_method = tracer_interp_method
    fieldset.dDICdt_diff.interp_method = tracer_interp_method
    fieldset.dNO3dt_diff.interp_method = tracer_interp_method
    fieldset.dPO4dt_diff.interp_method = tracer_interp_method
    # fieldset.dO2dt_diff.interp_method = tracer_interp_method

    vars_edw = ["EDW_criterion", "EDW_total_thickness", "EDW_layer_thickness", "EDW_outcropping_column_mask",
                "EDW_part_of_biggest_blob", "EDW_part_of_smaller_blob", "EDW_part_of_outcropping_blob"]
    filenames_edw = {'EDW_criterion': {'lon': edwfiles[0], 'lat': edwfiles[0], 'depth': edwfiles[0], 'data': edwfiles},
                     'EDW_layer_thickness': {'lon': edwfiles[0], 'lat': edwfiles[0], 'depth': edwfiles[0], 'data': edwfiles},
                     'EDW_outcropping_column_mask': {'lon': edwfiles[0], 'lat': edwfiles[0], 'depth': edwfiles[0], 'data': edwfiles},
                     'EDW_part_of_biggest_blob': {'lon': edwfiles[0], 'lat': edwfiles[0], 'depth': edwfiles[0], 'data': edwfiles},
                     'EDW_part_of_smaller_blob': {'lon': edwfiles[0], 'lat': edwfiles[0], 'depth': edwfiles[0], 'data': edwfiles},
                     'EDW_part_of_outcropping_blob': {'lon': edwfiles[0], 'lat': edwfiles[0], 'depth': edwfiles[0], 'data': edwfiles},
                     'EDW_total_thickness': {'lon': edwfiles[0], 'lat': edwfiles[0], 'data': edwfiles},
                     }
    variables_edw = dict(zip(vars_edw, vars_edw))

    dimensions_edw = {}
    for var in variables_edw:
        if var != "EDW_total_thickness":
            dimensions_edw[var] = {'lon': 'nav_lon', 'lat': 'nav_lat', 'depth': 'deptht', 'time': 'time_counter'}
        else:
            dimensions_edw[var] = {'lon': 'nav_lon', 'lat': 'nav_lat', 'time': 'time_counter'}
            parcels.FieldSet.from_nemo(filenames, variables, dimensions)

    fieldset_edw = parcels.FieldSet.from_netcdf(filenames_edw, variables_edw, dimensions_edw, allow_time_extrapolation=True)

    fieldset.add_field(fieldset_edw.EDW_criterion)
    fieldset.add_field(fieldset_edw.EDW_layer_thickness)
    fieldset.add_field(fieldset_edw.EDW_outcropping_column_mask)
    fieldset.add_field(fieldset_edw.EDW_part_of_biggest_blob)
    fieldset.add_field(fieldset_edw.EDW_part_of_smaller_blob)
    fieldset.add_field(fieldset_edw.EDW_part_of_outcropping_blob)
    fieldset.add_field(fieldset_edw.EDW_total_thickness)

    ds_z = xr.open_dataset(mesh_verti)
    fieldset.add_field(
        parcels.Field(
            "dz",
            # dz is the the thickness of the local w-cell.
            np.swapaxes(np.tile(ds_z.e3t_1d.data, (1, 1, 1)), 0, 2),
            lon=1,
            lat=1,
            depth=ds_z.gdept_1d.data,
            interp_method="nearest",
            allow_time_extrapolation=True,
            # mesh="flat",
        )
    )
    ds_z.close()

    fieldset.add_constant('sampling_interval', output_dt * 60 * 60)

    # ------ PARTICLESET CONSTRUCTION --------

    dlat = 0.25
    dlon = 0.25
    ddepth = 30

    latarr = np.arange(20, 45, dlat)
    lonarr = np.arange(-80, -35., dlon)
    deptharr = np.arange(10, 900, ddepth)
    # deptharr = np.array([50])

    latmesh, lonmesh, depthmesh = np.meshgrid(latarr, lonarr, deptharr)
    # The gulfstream runs from 38°N, 75°W to 46°N, 35°W
    # gulfmask = latmesh - 38 < (46 - 38)/(-35 - - 75)*(lonmesh - - 75)

    lons = lonmesh.flatten()
    lats = latmesh.flatten()
    depths = depthmesh.flatten()

    releasetimes = np.repeat(np.datetime64(f'{T0}T00:00'), lats.shape[0])

    class SampleParticle(parcels.JITParticle):
        timer = parcels.Variable('timer', initial=0, to_write=False)
        # Physical
        U = parcels.Variable('U', initial=0)
        V = parcels.Variable('V', initial=0)
        W = parcels.Variable('W', initial=0)
        T = parcels.Variable('T', initial=0)
        S = parcels.Variable('S', initial=0)
        KZ = parcels.Variable('KZ', initial=0)
        # BGC
        ALK = parcels.Variable('ALK', initial=0)
        CHL = parcels.Variable('CHL', initial=0)
        DIC = parcels.Variable('DIC', initial=0, dtype=np.float64)
        NO3 = parcels.Variable('NO3', initial=0)
        NPPV = parcels.Variable('NPPV', initial=0)
        O2 = parcels.Variable('O2', initial=0)
        PO4 = parcels.Variable('PO4', initial=0)
        # BGC fluxes
        DALK_diff = parcels.Variable('DALK_diff', initial=0)
        DDIC_diff = parcels.Variable('DDIC_diff', initial=0)
        DNO3_diff = parcels.Variable('DNO3_diff', initial=0)
        DPO4_diff = parcels.Variable('DPO4_diff', initial=0)
        # MLD
        MLDturb = parcels.Variable('MLDturb', initial=0)
        MLDtemp = parcels.Variable('MLDtemp', initial=0)
        MLDdens = parcels.Variable('MLDdens', initial=0)
        # EDW
        EDW_Eulerian = parcels.Variable('EDW_Eulerian', initial=0)
        EDW_layer_thickness = parcels.Variable('EDW_layer_thickness', initial=0)
        EDW_total_thickness = parcels.Variable('EDW_total_thickness', initial=0)
        EDW_outcropping_column_mask = parcels.Variable('EDW_outcropping_column_mask', initial=0)
        EDW_part_of_biggest_blob = parcels.Variable('EDW_part_of_biggest_blob', initial=0)
        EDW_part_of_smaller_blob = parcels.Variable('EDW_part_of_smaller_blob', initial=0)
        EDW_part_of_outcropping_blob = parcels.Variable('EDW_part_of_outcropping_blob', initial=0)
        EDW_Lagrangian = parcels.Variable('EDW_Lagrangian', initial=0)
        # Events
        EDW_exitevent = parcels.Variable('EDW_exitevent', initial=0)
        EDW_entryevent = parcels.Variable('EDW_entryevent', initial=0)
        MLD_exitevent = parcels.Variable('MLD_exitevent', initial=0)
        MLD_entryevent = parcels.Variable('MLD_entryevent', initial=0)
        in_region = parcels.Variable('in_region', initial=0)
        # Physical gradients
        DZ = parcels.Variable('DZ', initial=0)
        T_p1 = parcels.Variable('T_p1', initial=0)
        T_m1 = parcels.Variable('T_m1', initial=0)
        dTdz = parcels.Variable('dTdz', initial=0)
        S_p1 = parcels.Variable('S_p1', initial=0)
        S_m1 = parcels.Variable('S_m1', initial=0)

    pset_init = parcels.ParticleSet(fieldset,
                                    SampleParticle,
                                    lon=lons,
                                    lat=lats,
                                    depth=depths,
                                    time=releasetimes)

    def initialize_on_EDW(particle, fieldset, time):
        particle.EDW_Eulerian = fieldset.EDW_criterion[particle]
        particle.EDW_total_thickness = fieldset.EDW_total_thickness[particle]
        particle.EDW_layer_thickness = fieldset.EDW_layer_thickness[particle]
        if particle.EDW_Eulerian == False or particle.EDW_total_thickness < 30:
            particle.delete()

    pset_init.execute(pset_init.Kernel(initialize_on_EDW), dt=0)

    # ------ PARTICLE SIMULATION --------

    pset_run = parcels.ParticleSet(fieldset,
                                   SampleParticle,
                                   lon=pset_init.lon,
                                   lat=pset_init.lat,
                                   depth=pset_init.depth,
                                   time=np.repeat(np.datetime64(f'{T0}T00:00'), pset_init.lon.shape[0]))
    del pset_init

    pfile = pset_run.ParticleFile(
        name=f"/storage/shared/oceanparcels/output_data/data_Daan/EDW_trajectories/EDW_wfluxes_B_{T0}_{T}d_dt{dt}_odt{output_dt}.zarr", outputdt=delta(hours=output_dt))

    pfile.add_metadata("T0", str(T0))
    pfile.add_metadata("simulation_time", str(T))
    pfile.add_metadata("dt", str(dt))
    pfile.add_metadata("output_dt", str(output_dt))
    pfile.add_metadata("tracer_interp_method", tracer_interp_method)
    pfile.add_metadata("time_execution", time.now().strftime("%Y-%m-%d_%H:%M:%S"))

    pset_run.execute(pset_run.Kernel(EDW_Sampler_fluxes.EDW_Sampler_diffusion) +
                     pset_run.Kernel(EDW_Sampler_fluxes.EDW_Sampler_fluxes), dt=0)

    pset_run.execute(parcels.AdvectionRK4_3D +
                     pset_run.Kernel(EDW_Sampler_fluxes.EDW_Sampler_diffusion) +
                     pset_run.Kernel(EDW_Sampler_fluxes.EDW_Sampler_fluxes),
                     runtime=delta(days=T),
                     dt=delta(minutes=dt),
                     output_file=pfile,
                     verbose_progress=True,
                     recovery={parcels.ErrorCode.ErrorOutOfBounds: default_kernels.delete_particle,
                               parcels.ErrorCode.ErrorInterpolation: default_kernels.delete_particle_interp}
                     )
    pfile.close()

    print(
        f"Finished simulation with settings: \n T0 = {T0} \n T = {T}\n dt = {dt}\n output_dt = {output_dt}\n tracer_interp_method = {tracer_interp_method}\n backwards = {args.backwards}\n")
