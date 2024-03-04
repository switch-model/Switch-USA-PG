#!/bin/sh

## reporting
#SBATCH --error=logs/%A_%a.err
#SBATCH --output=logs/%A_%a.out
# change the following lines to have one pound sign instead of two
# if you want to receive notices about jobs
##SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
##SBATCH --mail-user=8083489586@vtext.com

#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
##SBATCH --mem-per-cpu=6400

#SBATCH --job-name=demand_scenarios
#SBATCH --partition=shared
#SBATCH --time=3-00:00:00
#SBATCH --array=1-4

module load lang/Anaconda3
# conda activate requires `conda init bash` or similar, which may not work here
source activate switch-pg

echo ============================================================
echo running job: switch solve-scenarios --scenario-queue sq/$SLURM_ARRAY_JOB_ID --job-id "$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID" "$@"
echo ============================================================
echo
srun --unbuffered switch solve-scenarios --scenario-queue sq/$SLURM_ARRAY_JOB_ID --job-id "$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID" "$@"
