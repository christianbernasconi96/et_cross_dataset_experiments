#!/bin/bash

while getopts "S:T:" opt; do
  case $opt in
    S) DATA_SRC=$OPTARG
    ;;
    T) DATA_TGT=$OPTARG
    ;;
  esac
done


# read -n 1 -s -r -p $'Modify Load into CreateAndSave in rw_options \n'
for s in $(seq 0 2)
do
    bash scripts/train_classifier_adapter_bert_ms.sh -s $s -S $DATA_SRC -T $DATA_TGT
done