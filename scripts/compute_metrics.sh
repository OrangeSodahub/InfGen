#!/bin/bash

export TORCH_LOGS='0'
export TF_CPP_MIN_LOG_LEVEL='2'
export TF_ENABLE_ONEDNN_OPTS='0'
export PYTHONPATH='.'

NUM_WORKERS=$1
SIM_DIR=$2

echo 'Start running ...'
python infgen/metrics/compute_metrics.py --compute_metric --num_workers "$NUM_WORKERS" --sim_dir "$SIM_DIR" ${@:3}

echo 'Done!
