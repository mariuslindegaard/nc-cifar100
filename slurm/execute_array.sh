#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --constraint=40GB
#SBATCH --constraint=high-capacity
#SBATCH --time=06:00:00
#SBATCH --exclude=node021,node071,node072,node073,node074,node075,node076
#SBATCH --mem=20000
# #SBATCH --qos=cbmm
# #SBATCH -p cbmm
# #SBATCH --output=./output.log

# #SBATCH --mail-user=lindegrd@mit.edu
# #SBATCH --mail-type=ALL

# memory 1000 MiB
# gpu_mem 10904 MiB
date;hostname;id;pwd

nvidia-smi

echo
echo "running script ${configs_path_base}${SLURM_ARRAY_TASK_ID}.sh"
chmod +x "${configs_path_base}${SLURM_ARRAY_TASK_ID}.sh"
srun -n 1 "${configs_path_base}${SLURM_ARRAY_TASK_ID}.sh"

