#!/bin/bash -l
#
#SBATCH -J moi_climatology       #name of your job   
#SBATCH -p normal           # request normal partition, job takes > 1 hour (this line can also be left out because 'normal' is the default)  
#SBATCH -t 1-00:00:00         # time in d-hh:mm:ss you want to reserve for the job
#SBATCH -n 1                # the number of cores you want to use for the job, SLURM automatically determines how many nodes are needed
#SBATCH -o log_moi_climatology.%j.o  # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e log_moi_climatology.%j.e  # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM

cd /nethome/4302001
conda activate xorca_env      # this passes your conda environment to all the compute nodes
cd /nethome/4302001/NASTMW_DIC/preparation

# for MONTH in 3 9
# do
#     srun python3 compute_climatology.py $MONTH S --year_extent 2000 2015
#     srun python3 compute_climatology.py $MONTH T --year_extent 2000 2015
#     srun python3 compute_climatology.py $MONTH U --year_extent 2000 2015
#     srun python3 compute_climatology.py $MONTH V --year_extent 2000 2015
#     srun python3 compute_climatology.py $MONTH W --year_extent 2000 2015
#     srun python3 compute_climatology.py $MONTH 2D --year_extent 2000 2015
#     srun python3 compute_climatology.py $MONTH KZLN10 --year_extent 2000 2015
# done

for MONTH in 1 2 4 5 6 7 8 10 11 12
do
    srun python3 compute_climatology.py $MONTH S
    srun python3 compute_climatology.py $MONTH T
    srun python3 compute_climatology.py $MONTH U
    srun python3 compute_climatology.py $MONTH V
    srun python3 compute_climatology.py $MONTH W
    srun python3 compute_climatology.py $MONTH 2D
    srun python3 compute_climatology.py $MONTH KZLN10
done

# iterate over months 1 till 12
for MONTH in {1..12}
do
    srun python3 compute_climatology.py $MONTH EDW
done

# iterate over months 1 till 12
for MONTH in {1..12}
do
    srun python3 compute_climatology.py $MONTH alk
    srun python3 compute_climatology.py $MONTH chl
    srun python3 compute_climatology.py $MONTH po4
    srun python3 compute_climatology.py $MONTH dic
    srun python3 compute_climatology.py $MONTH no3
    srun python3 compute_climatology.py $MONTH nppv
    srun python3 compute_climatology.py $MONTH o2
    srun python3 compute_climatology.py $MONTH si
done
