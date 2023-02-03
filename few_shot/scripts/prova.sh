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
# for i in 0
for i in 0
do
  for k in 5
  do
      bash scripts/train_prova.sh -d -s 0 -i $i -k $k -S $DATA_SRC -T $DATA_TGT -X cross
  done
done