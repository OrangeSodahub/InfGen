#! /bin/bash

# env
export REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
export HTTPS_PROXY="https://192.168.0.10:443/"
export https_proxy="https://192.168.0.10:443/"

export WANDB=1

# args
DEVICES=$1
CONFIG='configs/ours_long_term.yaml'
SAVE_CKPT_PATH='output/scalable_infgen_long'

# run
PYTHONPATH=".":$PYTHONPATH python3 run.py \
                            --train \
                            --devices $DEVICES \
                            --config $CONFIG \
                            --save_ckpt_path $SAVE_CKPT_PATH
