#!/bin/sh
#SBATCH --account=share-iv-mtp       # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=37_2_4_8_16_32_monday
#SBATCH --time=0-5:00:00         # format: D-HH:MM:SS

#SBATCH --partition=CPUQ          # Asking for a GPU
#SBATCH --mem=128G       # Asking for 64GB RAM
#SBATCH --cpus-per-task=4 # 4 CPU cores               
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


module load Anaconda3/2024.02-1

conda activate project

export GRB_WLSACCESSID="a1398bc9-8320-4b99-91fc-29b07ff4e4f7"
export GRB_WLSSECRET="87c9d38e-3a81-40a9-8d23-fa6fd14a0645"
export GRB_LICENSE_FILE="/cluster/home/tinusfa/master_project/gurobi.lic"



export GRB_WLSACCESSID="a1398bc9-8320-4b99-91fc-29b07ff4e4f7" >> ~/.bashrc
export GRB_WLSSECRET="87c9d38e-3a81-40a9-8d23-fa6fd14a0645" >> ~/.bashrc
export GRB_LICENSE_FILE="/cluster/home/tinusfa/master_project/gurobi.lic" >> ~/.bashrc
source ~/.bashrc

python -m src.run_batches -m 37_v1a_multi_weeks_2 37_v1a_multi_weeks_4 37_v1a_multi_weeks_8 37_v1a_multi_weeks_16 37_v1a_multi_weeks_32 -b 37_2_4_8_16_32_monday

echo "Job finished at $(date)"

echo "Pushing to git"

git add .
git commit -m "results from 37_2_4_8_16_32_monday"
git push