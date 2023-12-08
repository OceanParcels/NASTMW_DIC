#!/bin/bash -l
#
#SBATCH -J EDW_sim
#SBATCH -p normal           # partition (short: <3h, normal: <5d) 
#SBATCH -t 5-00:00:00       # time in d-hh:mm:ss you want to reserve for the job
#SBATCH -n 2      
#SBATCH --mem=40G      
#SBATCH -o log_EDW.%j.o     # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e log_EDW.%j.e     # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM
#SBATCH --mail-user=b.j.h.r.reijnders@uu.nl 
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source $HOME/miniconda3/bin/activate parcels

cd /nethome/4302001/NASTMW_DIC/simulations

for year in {1990..2015}
do
    echo "Starting simulations in ${year}"
    python3 NA_sim_EDW_sequestration_fluxes.py -T0 ${year}-09-01 -T 1095 -dt=90 &
    python3 NA_sim_EDW_sequestration_fluxes.py -T0 ${year}-09-01 -T 1095 -dt=-90 &
    wait
done