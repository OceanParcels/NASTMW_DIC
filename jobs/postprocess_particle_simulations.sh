#!/bin/bash -l
#
#SBATCH -J postprocess_particle_simulations       #name of your job   
#SBATCH -p normal           # request normal partition, job takes > 1 hour (this line can also be left out because 'normal' is the default)  
#SBATCH -t 4-00:00:00         # time in d-hh:mm:ss you want to reserve for the job
#SBATCH --mem=120G 
#SBATCH -n 8                # the number of cores you want to use for the job, SLURM automatically determines how many nodes are needed
#SBATCH -o log_postprocess_particle_simulations.%j.o  # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e log_postprocess_particle_simulations.%j.e  # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM

cd /nethome/4302001
conda activate base      # this passes your conda environment to all the compute nodes

cd /nethome/4302001/output_data/EDW_trajectories

shopt -s nullglob
runs=( EDW_wfluxes_B_*_1095d_*.nc )

echo "Collected the following runs:"
printf '%s\n' "${runs[@]}"

cd /nethome/4302001/NASTMW_DIC/analysis

for run in "${runs[@]}"; do
    echo "BASH: Processing file: $run"
    python3 EDW_DIC_postprocessing.py -c $run &
    
    ((count++))
    # Process 4 runs at a time in parallel
    if ((count % 4 == 0)); then
        wait
    fi
done
wait
