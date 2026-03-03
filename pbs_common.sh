# Common HPC environment setup.
# Source this file from job scripts: source "$(dirname "${BASH_SOURCE[0]}")/pbs_common.sh"

module load anaconda
module load cuda/12.6.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/Collin/envs/backdoor

export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "stubs" | tr '\n' ':')
export HF_HOME=/scratch/Collin/.cache/huggingface

echo "GPU allocated:"
nvidia-smi
