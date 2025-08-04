#!/bin/bash

echo "Starting running..."

SPLIT=$1

# multi-GPU training
PYTHONPATH='.':$PYTHONPATH python3 data_preprocess.py \
                        --split $SPLIT ${@:2}
