# HISTORY:
# 2023-07-04 - Changed EDW constraint to 17<%<20.5, dTdz<0.01
# 2023-07-05 - Fixed EDW constraints
# 2023-07-05 - Fixed sampling near surface layer

import math


def EDW_Sampler_diffusion(particle, fieldset, time):
    # Essential: Diffusive fluxes
    if particle.depth < 0.5058:
        sample_depth = 0.5058
    else:
        sample_depth = particle.depth
    if particle.timer == fieldset.sampling_interval / 2:
        dt_days = fieldset.sampling_interval / 86400
        particle.DALK_diff = fieldset.dALKdt_diff[time, sample_depth, particle.lat, particle.lon] * (dt_days)
        particle.DDIC_diff = fieldset.dDICdt_diff[time, sample_depth, particle.lat, particle.lon] * (dt_days)
        particle.DNO3_diff = fieldset.dNO3dt_diff[time, sample_depth, particle.lat, particle.lon] * (dt_days)
        particle.DPO4_diff = fieldset.dPO4dt_diff[time, sample_depth, particle.lat, particle.lon] * (dt_days)


def EDW_Sampler_fluxes(particle, fieldset, time):
    """
    Sample biogeochemical quantities, velocities, temperature and salinities
    for mode water analysis and biogeochemical timeseries creation.
    """

    EDW_lower_bound = 17.0
    EDW_upper_bound = 20.5
    EDW_stratification_bound = 0.01
    EDW_east_bound = -35
    EDW_north_bound = 50
    EDW_west_bound = -80
    EDW_south_bound = 15

    if particle.depth < 0.5058:
        sample_depth = 0.5058
    else:
        sample_depth = particle.depth

    particle.timer += math.fabs(particle.dt)
    if particle.timer >= fieldset.sampling_interval:
        particle.timer = 0

    if particle.timer == 0:
        # Optional: velocities
        particle.U, particle.V, particle.W = fieldset.UVW[particle]

        # Essential: new BGC
        particle.ALK = fieldset.ALK[time, sample_depth, particle.lat, particle.lon]
        particle.CHL = fieldset.CHL[time, sample_depth, particle.lat, particle.lon]
        particle.DIC = fieldset.DIC[time, sample_depth, particle.lat, particle.lon]
        particle.NO3 = fieldset.NO3[time, sample_depth, particle.lat, particle.lon]
        particle.NPPV = fieldset.NPPV[time, sample_depth, particle.lat, particle.lon]
        particle.O2 = fieldset.O2[time, sample_depth, particle.lat, particle.lon]
        particle.PO4 = fieldset.PO4[time, sample_depth, particle.lat, particle.lon]

        # Optional: mixing
        particle.KZ = 10**fieldset.KZlog10[time, sample_depth, particle.lat, particle.lon]

        # Optional: Sampling new edw constraints
        T_new = fieldset.T[time, sample_depth, particle.lat, particle.lon]
        DZ_new = fieldset.dz[time, sample_depth, particle.lat, particle.lon]
        if particle.depth > DZ_new and particle.depth > 1.04:
            T_p1_new = fieldset.T[time, sample_depth + DZ_new/2, particle.lat, particle.lon]
            T_m1_new = fieldset.T[time, sample_depth - DZ_new/2, particle.lat, particle.lon]
        dTdz_new = -(T_p1_new - T_m1_new)/DZ_new

        if particle.lon < EDW_east_bound and particle.lon > EDW_west_bound and particle.lat > EDW_south_bound and particle.lat < EDW_north_bound:
            in_region_new = 1
        else:
            in_region_new = 0

        # Optional: Recording entry and exit events
        particle.EDW_exitevent = 0
        particle.EDW_entryevent = 0

        if particle.T < EDW_lower_bound and T_new >= EDW_lower_bound:
            particle.EDW_entryevent += 1  # Lower T-boundary crossing
        elif particle.T >= EDW_lower_bound and T_new < EDW_lower_bound:
            particle.EDW_exitevent += 1
        elif particle.T > EDW_upper_bound and T_new <= EDW_upper_bound:
            particle.EDW_entryevent += 2  # Upper T-boundary crossing
        elif particle.T <= EDW_upper_bound and T_new > EDW_upper_bound:
            particle.EDW_exitevent += 2
        if math.fabs(particle.dTdz) > EDW_stratification_bound and math.fabs(dTdz_new) <= EDW_stratification_bound:
            particle.EDW_entryevent += 4  # Crossing stratification
        elif math.fabs(particle.dTdz) <= EDW_stratification_bound and math.fabs(dTdz_new) > EDW_stratification_bound:
            particle.EDW_exitevent += 4
        if particle.in_region == 0 and in_region_new == 1:
            particle.EDW_entryevent += 8  # Crossing region
        elif particle.in_region == 1 and in_region_new == 0:
            particle.EDW_exitevent += 8

        MLDturb_new = fieldset.MLDturb[time, sample_depth, particle.lat, particle.lon]
        MLDtemp_new = fieldset.MLDtemp[time, sample_depth, particle.lat, particle.lon]
        MLDdens_new = fieldset.MLDdens[time, sample_depth, particle.lat, particle.lon]

        # Optional: recording MLD events
        particle.MLD_entryevent = 0
        particle.MLD_exitevent = 0

        if particle.depth > particle.MLDturb and particle.depth <= MLDturb_new:
            particle.MLD_entryevent += 1  # Mixing layer
        elif particle.depth <= particle.MLDturb and particle.depth > MLDturb_new:
            particle.MLD_exitevent += 1
        if particle.depth > particle.MLDtemp and particle.depth <= MLDtemp_new:
            particle.MLD_entryevent += 2  # Mixed layer
        elif particle.depth <= particle.MLDtemp and particle.depth > MLDtemp_new:
            particle.MLD_exitevent += 2

        # Essential: mixed layer
        particle.MLDturb = MLDturb_new
        particle.MLDtemp = MLDtemp_new
        particle.MLDdens = MLDdens_new

        # Essential: EOS
        particle.T = T_new
        particle.S = fieldset.S[time, sample_depth, particle.lat, particle.lon]

        # Optional: gradients
        particle.DZ = DZ_new
        if particle.depth > particle.DZ and particle.depth > 1.04:
            # EOS
            particle.T_p1 = T_p1_new
            particle.T_m1 = T_m1_new
            particle.S_p1 = fieldset.S[time, sample_depth + particle.DZ/2, particle.lat, particle.lon]
            particle.S_m1 = fieldset.S[time, sample_depth - particle.DZ/2, particle.lat, particle.lon]
            particle.dTdz = dTdz_new

        # Essential: EDW
        if in_region_new:
            particle.in_region = True
            particle.EDW_Eulerian = fieldset.EDW_criterion[time, sample_depth, particle.lat, particle.lon]
            particle.EDW_total_thickness = fieldset.EDW_total_thickness[time, sample_depth, particle.lat, particle.lon]
            particle.EDW_layer_thickness = fieldset.EDW_layer_thickness[time, sample_depth, particle.lat, particle.lon]
            particle.EDW_outcropping_column_mask = fieldset.EDW_outcropping_column_mask[time, sample_depth, particle.lat, particle.lon]
            particle.EDW_part_of_biggest_blob = fieldset.EDW_part_of_biggest_blob[time, sample_depth, particle.lat, particle.lon]
            particle.EDW_part_of_smaller_blob = fieldset.EDW_part_of_smaller_blob[time, sample_depth, particle.lat, particle.lon]
            particle.EDW_part_of_outcropping_blob = fieldset.EDW_part_of_outcropping_blob[time, sample_depth, particle.lat, particle.lon]
            if particle.T > EDW_lower_bound and particle.T < EDW_upper_bound and math.fabs(particle.dTdz) < EDW_stratification_bound:
                particle.EDW_Lagrangian = True
            else:
                particle.EDW_Lagrangian = False
        else:
            particle.in_region = False
            particle.EDW_Lagrangian = False
            particle.EDW_Eulerian = False
