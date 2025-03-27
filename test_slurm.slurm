#!/bin/sh
#SBATCH --account=share-iv-mtp       # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=test_tinus
#SBATCH --time=0-00:15:00         # format: D-HH:MM:SS

#SBATCH --partition=CPUQ          # Asking for a GPU
#SBATCH --mem=5G       # Asking for 64GB RAM
#SBATCH --cpus-per-task=1 # 4 CPU cores               
#SBATCH --nodes=1
#SBATCH --output=slurm_outputs/%x%j.txt      # Specifying 'stdout'
#SBATCH --error=slurm_outputs/%x%j.err        # Specifying 'stderr'

#SBATCH --mail-user=tinusfa@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

python -m src.run_tests

echo "Job finished at $(date)"