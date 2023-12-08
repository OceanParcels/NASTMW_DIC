#!/bin/bash -l
#
#SBATCH -J multiyear_event_aggregation       #name of your job   
#SBATCH -p normal           # request normal partition, job takes > 1 hour (this line can also be left out because 'normal' is the default)  
#SBATCH -t 4-00:00:00         # time in d-hh:mm:ss you want to reserve for the job
#SBATCH -n 1                # the number of cores you want to use for the job, SLURM automatically determines how many nodes are needed
#SBATCH --mem=140G          
#SBATCH -o multiyear_event_analysis.%j.o  # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e multiyear_event_analysis.%j.e  # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM

cd /nethome/4302001
conda activate base      # this passes your conda environment to all the compute nodes
cd /nethome/4302001/NASTMW_DIC/analysis

srun python3 EDW_DIC_multiyear_event_aggregation.py