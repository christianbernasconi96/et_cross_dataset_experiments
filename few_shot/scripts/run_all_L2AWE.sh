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
for i in 0 1 2
  do
  for k in 5 10 20
    do
      bash scripts/train_L2AWE_adapter_bert_ms.sh -s 0 -i $i -k $k -S $DATA_SRC -T $DATA_TGT
    done
  done