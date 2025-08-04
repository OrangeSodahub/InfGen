#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

cd ~/infgen/

trap "echo -e \"${RED}Stopping script...${NC}\"; kill -- -$$" SIGINT

while true; do
    echo -e "${RED}Start running ...${NC}"
    PYTHONPATH='.':$PYTHONPATH setsid python data_preprocess.py --split training &
    PID=$!

    sleep 1200

    echo -e "${RED}Sending SIGINT to process group $PID...${NC}"
    PGID=$(ps -o pgid= -p $PID | tail -n 1 | tr -d ' ')
    kill -- -$PGID
    wait $PID

    sleep 10
done