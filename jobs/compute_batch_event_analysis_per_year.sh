#!/bin/bash -l
#
#SBATCH -J event_analysis_per_year       #name of your job   
#SBATCH -p normal           # request normal partition, job takes > 1 hour (this line can also be left out because 'normal' is the default)  
#SBATCH -t 5-00:00:00         # time in d-hh:mm:ss you want to reserve for the job
#SBATCH -n 10                # the number of cores you want to use for the job, SLURM automatically determines how many nodes are needed
#SBATCH --mem=145G          
#SBATCH -o log_event_analysis_per_year.%j.o  # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e log_event_analysis_per_year.%j.e  # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM

cd /nethome/4302001
conda activate base      # this passes your conda environment to all the compute nodes
cd /nethome/4302001/NASTMW_DIC/analysis

runs=(
    "EDW_wfluxes_B_1995-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_1995-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_1996-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_1996-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_1997-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_1997-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_1998-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_1998-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_1999-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_1999-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2000-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2000-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2001-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2001-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2002-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2002-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2003-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2003-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2004-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2004-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2005-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2005-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2006-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2006-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2007-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2007-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2008-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2008-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2009-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2009-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2010-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2010-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2011-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2011-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2012-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2012-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2013-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2013-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2014-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2014-09-01_1095d_dt90_odt24"
    "EDW_wfluxes_B_2015-09-01_1095d_dt-90_odt24"
    "EDW_wfluxes_B_2015-09-01_1095d_dt90_odt24"
)

for window in 1 6 20; do
    for run in "${runs[@]}"; do
        echo "Processing file: $run with window $window"
        python3 EDW_DIC_event_analysis_per_year.py -n $run -rw $window --skip_labeling&
        
        ((count++))
        # Process 12 runs at a time in parallel
        if ((count % 10 == 0)); then
            wait
        fi
    done
done
wait



