#! /bin/bash

cd ~/infgen/

export REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
export HTTPS_PROXY="https://192.168.0.10:443/"
export https_proxy="https://192.168.0.10:443/"

export WANDB=1

# args
DEVICES=$1
CONFIG='configs/ours_long_term.yaml'
CKPT_PATH=''

# run
PYTHONPATH=".":$PYTHONPATH python3 run.py \
                            --devices $DEVICES \
                            --config $CONFIG \
                            --ckpt_path $CKPT_PATH ${@:2}

