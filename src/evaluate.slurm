#!/bin/bash

#=========================================================================================
# Slurm SBATCH Directives
#=========================================================================================
#
#SBATCH --job-name=qwen_semantics_exp_FAST   # A descriptive name for your job
#SBATCH --output=/scratch/pachary4/functional_competence/prompt_semantics/src/job_output/qwen_semantics_exp_FAST_%j.out # File to redirect stdout, %j is the job ID
#SBATCH --error=/scratch/pachary4/functional_competence/prompt_semantics/src/job_output/qwen_semantics_exp_FAST_%j.err  # File to redirect stderr, %j is the job ID
#
#SBATCH --partition=contrib-gpuq        # The partition (queue) to submit to
#SBATCH --qos=gpu                       # The Quality of Service
#
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --ntasks-per-node=4             # Request 4 CPU cores
#SBATCH --mem=64G                       # Request 64GB of system RAM
#
# MODIFIED: Requesting the fastest available GPU, the H100 80GB
#SBATCH --gres=gpu:H100.80gb:1

#
# MODIFIED: Reduced the time limit since the H100 will be much faster. 1 day is very safe.
#SBATCH --time=1-00:00:00               # Set a 1-day time limit (D-HH:MM:SS)

#=========================================================================================
# Environment and Execution
#=========================================================================================

# --- 1. Print Job Information ---

# to see ID and state of GPUs assigned
module load gnu12/12.3.0
module load cuda/12.6.3  # this will set $CUDA_HOME
module load python/3.12.1-33
module load git/2.27.1


# --- 2. Setup Your Environment ---
source /home/pachary4/.virtualenvs/functional/bin/activate

# --- 3. Navigate to the Script Directory ---
cd /scratch/pachary4/functional_competence/prompt_semantics/src

set echo
umask 0027


cd /scratch/pachary4/functional_competence/prompt_semantics/src
unset CUDA_VISIBLE_DEVICES

export WANDB_CACHE_DIR="/scratch/pachary4/.cache/wandb"
export WANDB_DATA_DIR="/scratch/pachary4/.cache/wandb"

python qwen_decoder_only.py