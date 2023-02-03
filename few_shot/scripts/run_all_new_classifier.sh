#!/bin/bash

while getopts "T:" opt; do
  case $opt in
    T) DATA_TGT=$OPTARG
  esac
done


for i in 0 1 2
do
  for k in 5 10 20
  do
    bash scripts/train_new_classifier_adapter_bert_ms.sh -s 0 -i $i -k $k -T $DATA_TGT
  done
done