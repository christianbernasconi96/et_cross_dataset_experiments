#!/bin/bash

while getopts "S:T:" opt; do
  case $opt in
    S) DATA_SRC=$OPTARG
    ;;
    T) DATA_TGT=$OPTARG
    ;;
  esac
done

bash scripts/run_all_classifier.sh -S $DATA_SRC -T $DATA_TGT
bash scripts/run_all_kenn.sh -S $DATA_SRC -T $DATA_TGT
bash scripts/run_all_L2AWE.sh -S $DATA_SRC -T $DATA_TGT