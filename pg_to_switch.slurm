#!/bin/sh

## reporting
#SBATCH --error=logs/%A_%a.err
#SBATCH --output=logs/%A_%a.out
# change the following lines to have one pound sign instead of two
# if you want to receive notices about jobs
##SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
##SBATCH --mail-user=someone@somewhere.com

#SBATCH --mem-per-cpu=6400

#SBATCH --job-name=pg_to_switch.py
#SBATCH --partition=shared
#SBATCH --time=3-00:00:00
#SBATCH --array=1-24

module load lang/Anaconda3
# conda activate requires `conda init bash` or similar, which may not work here
source activate switch-pg

# run pg_to_switch.py with arguments from the command line
# note: --case-index runs one case from the list of cases requested, which may
# be given by --case-id xxxx --case-id yyyy, etc. or omitted (in which case it
# will be all possible cases)
echo Running: python pg_to_switch.py "$@" --case-index "$SLURM_ARRAY_TASK_ID"
srun --unbuffered python pg_to_switch.py "$@" --case-index "$SLURM_ARRAY_TASK_ID"
