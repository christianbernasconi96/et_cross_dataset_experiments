#!/bin/bash

CMD='python'
SEED=0
while getopts "ds:i:k:S:T:" opt; do
  case $opt in
    d) CMD='debugpy-run -p :5681'
    ;;
    s) SEED=$OPTARG
    ;;
    S) DATA_SRC=$OPTARG
    ;;
    T) DATA_TGT=$OPTARG
    ;;
    i) instance=$OPTARG
    ;;
    k) shots=$OPTARG
  esac
done

SRC2TGT="${DATA_SRC}2${DATA_TGT}"
TGT2SRC="${DATA_TGT}2${DATA_SRC}"

# prepare model yaml
MODEL_CONFIG_TEMPLATE='configs/model/classifier_bert.yaml'
NEW_MODEL_CONFIG_PATH='configs/model/generated_config_files/'$SRC2TGT'_classifier_bert.yaml'
cp $MODEL_CONFIG_TEMPLATE $NEW_MODEL_CONFIG_PATH
sed -i 's/SRC2TGT.json/'$SRC2TGT'.json/g' $NEW_MODEL_CONFIG_PATH
sed -i 's/TGT2SRC.json/'$TGT2SRC'.json/g' $NEW_MODEL_CONFIG_PATH
sed -i 's/DATA_SRC/'$DATA_SRC'/g' $NEW_MODEL_CONFIG_PATH

DATA_CONFIG_TEMPLATE='configs/data/'$SRC2TGT'_bert_ms.yaml'
NEW_DATA_CONFIG_PATH='configs/data/generated_data_files/'$SRC2TGT'_k'$shots'_instance'$instance'_classifier_bert.yaml'
cp $DATA_CONFIG_TEMPLATE $NEW_DATA_CONFIG_PATH
sed -i 's/kSHOTS/k'$shots'/g' $NEW_DATA_CONFIG_PATH
sed -i 's/sINSTANCE/s'$instance'/g' $NEW_DATA_CONFIG_PATH

for e in 1
do
  $CMD trainers/trainer_bert.py fit \
  --seed_everything=$SEED \
  --data configs/data/common.yaml \
  --data $NEW_DATA_CONFIG_PATH \
  --trainer configs/trainer_common.yaml \
  --trainer.callbacks=ModelCheckpoint \
  --trainer.callbacks.dirpath=/home/remote_hdd/trained_models/$DATA_TGT/cross_dataset_few_shots \
  --trainer.callbacks.filename=$DATA_SRC'_classifier_adapter_bert_ms' \
  --trainer.callbacks.monitor=val_loss \
  --trainer.callbacks.save_weights_only=True \
  --trainer.callbacks=EarlyStopping \
  --trainer.callbacks.patience=5 \
  --trainer.callbacks.monitor=val_loss \
  --trainer.callbacks.mode=min \
  --trainer.callbacks.verbose=True \
  --trainer.max_epochs=$e \
  --trainer.limit_train_batches=1 \
  --model configs/model/common.yaml \
  --model $NEW_MODEL_CONFIG_PATH \
  --logger configs/logger.yaml \
  --logger.project='cross_dataset_'$DATA_TGT'_few_shots_k'$shots \
  --logger.name=$e'_'$DATA_SRC'_classifier_adapter_bert_ms'
done