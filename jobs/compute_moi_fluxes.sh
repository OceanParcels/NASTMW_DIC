#!/bin/bash -l
#
#SBATCH -J moi_fluxes       #name of your job   
#SBATCH -p normal           # request normal partition, job takes > 1 hour (this line can also be left out because 'normal' is the default)  
#SBATCH -t 1-00:00:00         # time in d-hh:mm:ss you want to reserve for the job
#SBATCH -n 1                # the number of cores you want to use for the job, SLURM automatically determines how many nodes are needed
#SBATCH -o log_moi_fluxes.%j.o  # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e log_moi_fluxes.%j.e  # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM


cd /nethome/4302001
conda activate xorca_env      # this passes your conda environment to all the compute nodes
cd /nethome/4302001/NASTMW_DIC/preparation

srun python3 compute_vertical_fluxes.py