# %%
import os
import pickle
import sys
import pandas as pd
import entity_typing_framework.main_module.main_module as mm
import torch
from utils import forward_et_standard
import json


DATA_SRC = 'figer'
TOKENIZER_PARAMETERS = {
  'figer': {
      'M': 6,
      'R': 19,
      'L': 19,
      'T': 80
    },
  'bbn' :{
      'M': 5,
      'R': 13,
      'L': 13,
      'T': 80
    },
  'ontonotes_shimaoka':{
      'M': 7,
      'R': 9,
      'L': 9,
      'T': 80
    }
}

DATA_TGT ='ontonotes_shimaoka'
TOKENIZER_PLACEHOLDER = 'bert-large-cased_M{M}L{L}R{R}T{T}_light.pickle'
TOKENIZED_FILE_NAME = TOKENIZER_PLACEHOLDER.format(**TOKENIZER_PARAMETERS[DATA_SRC])
DATA_TGT_PATH = f'/home/remote_hdd/tokenized_datasets/{DATA_TGT}/cross_dataset/{DATA_SRC}2{DATA_TGT}/{TOKENIZED_FILE_NAME}'

PROJECTOR = 'classifier'
SEED = 0
VERSION = f'-v{SEED}' if SEED != 0 else ''
MODEL_TGT_PATH = f'/home/remote_hdd/trained_models/{DATA_TGT}/cross_dataset/{DATA_SRC}_{PROJECTOR}_adapter_bert_ms{VERSION}.ckpt'



MAIN_MODULE = {
  'classifier' : mm.CrossDatasetMainModule,
  'kenn' : mm.CrossDatasetKENNMultilossMainModule,
  'L2AWE' : mm.CrossDatasetMainModule
}

# %%
### LOADs

# models
model = mm.MainModule.load_ET_Network_for_test_(MODEL_TGT_PATH)

# prepare type utils
type2id_src_ = model.src_classifier.type2id
type2id_tgt = model.src_classifier.type2id

# tgt test set
tokenized_test = pickle.load(open(DATA_TGT_PATH, 'rb'))['tokenized_datasets']['test']
tokenized_test_sentences = tokenized_test.tokenized_data['tokenized_sentences']
input_ids = tokenized_test_sentences['input_ids']
attention_mask = tokenized_test_sentences['attention_mask']
one_hot_types = tokenized_test.tokenized_data['one_hot_types']

# predict
network_output_for_inference, y_true = forward_et_standard(model, input_ids, attention_mask, one_hot_types, MAIN_MODULE[PROJECTOR].get_output_for_inference)
y_true = y_true.cuda()
    



# %%
