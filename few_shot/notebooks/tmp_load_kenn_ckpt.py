# %%
import torch
import numpy as np
KENN_SETUP = 'vertical'
KB_MODE = 'hybrid'
VERSION = '-v2'
DATA = 'bbn'
CKPT_NAME = f'{DATA}_kenn_{KENN_SETUP}_{KB_MODE}_adapter_bert_ms{VERSION}.ckpt'
# CKPT_NAME = f'{DATA}_kenn_cross_adapter_bert_ms{VERSION}.ckpt'
CKPT_PATH = f'/home/remote_hdd/trained_models/figer/cross_dataset_few_shots/{CKPT_NAME}'
ckpt = torch.load(CKPT_PATH)
state_dict = ckpt['state_dict']
#%%
kenn_keys = [k for k in state_dict if k.startswith('input_projector.ke.knowledge_enhancer')]
weights = []
for kk in kenn_keys:
  clause = kk.split('.')[-2]
  weight = state_dict[kk].item()
  weights.append(weight)
  print(clause, ':', weight)
print('MEAN:', np.mean(weights))
# %%
