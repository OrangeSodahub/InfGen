#!/bin/bash

#SBATCH --job-name YOUR_JOB_NAME        # Job name
### Logging
#SBATCH --output=job_out/%j.out                 # Stdout (%j expands to jobId)
#SBATCH --error=job_out/%j.err                  # Stderr (%j expands to jobId)
### Node info
#SBATCH --nodes=1                       # Single node or multi node
#SBATCH --nodelist=sota-6
#SBATCH --time 20:00:00                 # Max time (hh:mm:ss)
#SBATCH --gres=gpu:4                    # GPUs per node
#SBATCH --mem=256G                       # Recommend 32G per GPU
#SBATCH --ntasks-per-node=4             # Tasks per node
#SBATCH --cpus-per-task=256               # Recommend 8 per GPU
### Whatever your job needs to do

export REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
export HTTPS_PROXY="https://192.168.0.10:443/"
export https_proxy="https://192.168.0.10:443/"

export TEST_VAL_PRED=True
export WANDB=True

cd ~/infgen/
PYTHONPATH=".":$PYTHONPATH python3 train.py --config configs/train/pretrain_scalable_map.yaml --save_ckpt_path output/ours_map_pretrain
