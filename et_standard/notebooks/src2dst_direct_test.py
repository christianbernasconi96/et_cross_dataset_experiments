# %%
import os
import pickle
import sys
import pandas as pd
from entity_typing_framework.main_module.inference_manager import IncrementalThresholdOrMaxInferenceManager, ThresholdOrMaxInferenceManager
from entity_typing_framework.main_module.metric_manager import MetricManager, MetricManagerForIncrementalTypes
import pandas as pd
import entity_typing_framework.main_module.main_module as mm
import torch
from utils import forward_et_standard, forward_incremental
import json


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

TOKENIZER_PLACEHOLDER = 'bert-large-cased_M{M}L{L}R{R}T{T}_light.pickle'

# TOKENIZER_CONFIG = {  
#   'figer': 'bert-large-cased_M6L19R19T80_light.pickle',
#   'bbn': 'bert-large-cased_M5L13R13T80_light.pickle',
#   'ontonotes_shimaoka': 'bert-large-cased_M7L9R9T80_light.pickle'
# }


MAIN_MODULE = {
  'classifier' : mm.MainModule,
  'kenn_top_down' : mm.KENNMultilossMainModule,
  'kenn_bottom_up' : mm.KENNMultilossMainModule,
  'kenn_hybrid' : mm.KENNMultilossMainModule,
  'box' : mm.BoxEmbeddingMainModule
}

REDIRECT_LOG = True
LOG_FILE_PATH = '/home/cbernasconi/et/experiments/cross_dataset/results/cross_dataset_log.txt'

if REDIRECT_LOG:
    sys.stdout = open(LOG_FILE_PATH,'at')

# %%
def print_metrics(metrics):
  for k, v in metrics.items():
    print(f'{k}: {round(v.item(), 3)}')

# %%

def predict_on_dst_given_source_model(model, data_src, denoising_seed):
  dict_of_cross_dataset_metrics = defaultdict(list)
  for data_dst in DATA_DST:
    tokenized_file_name = TOKENIZER_PLACEHOLDER.format(**TOKENIZER_PARAMETERS[data_src])
    
    if data_dst != data_src:

      DATA_DST_PATH = f'/home/remote_hdd/tokenized_datasets/{data_dst}/cross_dataset/{data_src}2{data_dst}/{tokenized_file_name}'
      
      
      # prepare mappings
      # MAPPINGS_PATH = f'/home/cbernasconi/et/experiments/cross_dataset/cross_dataset_mappings/{data_src}2{data_dst}.json'
      MAPPINGS_PATH = f'/home/cbernasconi/et/experiments/cross_dataset/mappings/{data_src}2{data_dst}.json'

      
      with open(MAPPINGS_PATH, 'r') as f:
        mappings = json.loads(f.read())
    
    
    else:


      DATA_DST_PATH = f'/home/remote_hdd/tokenized_datasets/{data_dst}/et_standard/{tokenized_file_name}'

      mappings = {t: t for t in model.type2id.keys()}

    # DATA_SRC_PATH = f'/home/remote_hdd/tokenized_datasets/{data_src}/et_standard/{TOKENIZER_CONFIG[data_src]}'
    # DATA_SRC_PATH = f'/home/remote_hdd/tokenized_datasets/{data_src}/et_standard/{tokenized_file_name}'
    # TYPES_PATH_SRC = os.path.join(DATA_SRC_PATH.replace(tokenized_file_name, ''), 'types_list.txt')
    TYPES_PATH_DST = os.path.join(DATA_DST_PATH.replace(tokenized_file_name, ''), 'types_list.txt')
    # TYPES_PATH_SRC = f'/home/remote_hdd/datasets/{data_src}/all_types.txt'
    # TYPES_PATH_DST = f'/home/remote_hdd/datasets/{data_dst}/all_types.txt'

    # print(DATA_SRC_PATH)
    # print(DATA_DST_PATH)
    # print(TYPES_PATH_SRC)
    # print(TYPES_PATH_DST)



    # read mappings from csv, obsolete if create_mappings.py is used
    # mappings = pd.read_csv(f'./cross_datasets_mappings/{DATA_DST}.csv', index_col=DATA_DST)
    # mappings = mappings.replace('-', None)

    types_src = list(model.type2id.keys())
    types_dst = open(TYPES_PATH_DST, 'r').read().splitlines()
    # dst2src = { t: mappings.loc[t, DATA_SRC] for t in types_dst }
    src2dst = mappings
    # prepare type utils
    type2id_src = {t: idx for idx, t in enumerate(types_src)} 
    type2id_dst = {t: idx for idx, t in enumerate(types_dst)}
    # type2id_evaluation = { f : type2id_all[f] for f in fathers }

    # load dst test set
    tokenized_test = pickle.load(open(DATA_DST_PATH, 'rb'))['tokenized_datasets']['test']
    tokenized_test_sentences = tokenized_test.tokenized_data['tokenized_sentences']
    input_ids = tokenized_test_sentences['input_ids']
    attention_mask = tokenized_test_sentences['attention_mask']
    one_hot_types = tokenized_test.tokenized_data['one_hot_types']

    # prepare inference managers
    inference_manager = ThresholdOrMaxInferenceManager(name=None, threshold=.5, type2id=model.type2id, transitive=TRANSITIVE)

    # predict
    network_output_for_inference, y_true = forward_et_standard(model, input_ids, attention_mask, one_hot_types, MAIN_MODULE[PROJECTOR].get_output_for_inference)
    y_true = y_true.cuda()


    inferred_types_src = inference_manager.infer_types(network_output_for_inference.cuda())

    # list of dst types that can be predicted
    mapped_types_dst = [t for t in set(list(src2dst.values())) if t]

    # map src inferred types to dst inferred types
    inferred_types_dst = torch.zeros_like(y_true)
    idx_to_drop = []

    for dst_t_name, dst_t_id in type2id_dst.items():

      if dst_t_name in mapped_types_dst:
        idx_to_convert = [type2id_src[src] for src, dst in src2dst.items() if dst == dst_t_name and src in type2id_src]

        for id in idx_to_convert:
          inferred_types_dst[:, dst_t_id] += inferred_types_src[:, id]
      else:
        idx_to_drop.append(dst_t_id)
        
    inferred_types_dst = torch.clip(inferred_types_dst, 0, 1)

    inference_manager = ThresholdOrMaxInferenceManager(name=None, threshold=.5, type2id=type2id_dst, transitive=TRANSITIVE)
    inferred_types_dst = inferred_types_dst.to(torch.float32)
    inferred_types_dst = inference_manager.infer_types(inferred_types_dst.cuda())


    # global metrics (unmapped types will drop performance)
    # macro example on the entire TGT test dataset
    metric_manager = MetricManager(len(type2id_dst), 'cuda', type2id_dst, 'test')
    metric_manager.update(inferred_types_dst.cuda(), y_true.cuda())
    metrics = metric_manager.compute()
    print_metrics(metrics)
    print('-'*50)


    # macro_types_mapped_only 

    # metrics filtered by keeping only mapped types
    idx_to_maintain = torch.tensor([i for i in range(inferred_types_dst.shape[1]) if i not in idx_to_drop])
    y_true_filtered = y_true.index_select(dim=1, index=idx_to_maintain.cuda())
    inferred_types_dst_filtered = inferred_types_dst.index_select(dim=1, index=idx_to_maintain.cuda())
    metric_manager_filtered = MetricManager(y_true_filtered.shape[1], 'cuda', None, 'test_filtered')
    metric_manager_filtered.update(inferred_types_dst_filtered.cuda(), y_true_filtered.cuda())
    metrics_filtered = metric_manager_filtered.compute()
    print_metrics(metrics_filtered)
    print('-'*50)


    # metrics filtered by keeping only mapped types and examples with at least one mapped type

    test_incremental_only_exclusive_metric_manager = MetricManager(y_true_filtered.shape[1], 'cuda', None, 'exclusive_test_filtered')

    idx = torch.sum(y_true_filtered, dim=1).nonzero().squeeze()
    exclusive_y_true_filtered = y_true_filtered.index_select(dim=0, index=idx)
    if torch.sum(y_true_filtered):
        exclusive_inferred_types_dst_filtered = inferred_types_dst_filtered.index_select(dim=0, index=idx)
        test_incremental_only_exclusive_metric_manager.update(exclusive_inferred_types_dst_filtered, exclusive_y_true_filtered)
    exclusive_metrics_filtered = test_incremental_only_exclusive_metric_manager.compute()
    print_metrics(exclusive_metrics_filtered)
    print('-'*50)

    # metrics per type on all examples
    type_metric_manager = MetricManagerForIncrementalTypes(len(type2id_dst), 'cuda', 'test_type_all')
    type_metric_manager.update(inferred_types_dst.cuda(), y_true.cuda())
    metrics_per_type = type_metric_manager.compute(type2id_dst)
    print_metrics(metrics_per_type)
    print('-'*50)


    # metrics per type filtered by keeping only mapped types and examples with at least one mapped type
    test_type_exclusive_metric_manager = MetricManagerForIncrementalTypes(len(type2id_dst), 'cuda', 'test_type_exclusive')

    idx = torch.sum(y_true_filtered, dim=1).nonzero().squeeze()
    exclusive_y_true = y_true.index_select(dim=0, index=idx)

    if torch.sum(y_true_filtered):
        exclusive_inferred_types_dst = inferred_types_dst.index_select(dim=0, index=idx)
        test_type_exclusive_metric_manager.update(exclusive_inferred_types_dst, exclusive_y_true)

    exclusive_metrics_per_type = test_type_exclusive_metric_manager.compute(type2id_dst)
    print_metrics(exclusive_metrics_per_type)
    print('-'*50)

    def update_with_metrics(incremental_dict, metrics):
      item_metrics = {k:v.item() for k, v in metrics.items()}
      incremental_dict.update(item_metrics)
      return incremental_dict

    # accumuli dizionari
    dict_for_pandas = {}
    dict_for_pandas['src'] = data_src
    dict_for_pandas['denoising'] = denoising_seed

    dict_for_pandas = update_with_metrics(dict_for_pandas, metrics)
    dict_for_pandas = update_with_metrics(dict_for_pandas, metrics_filtered)
    dict_for_pandas = update_with_metrics(dict_for_pandas, exclusive_metrics_filtered)
    dict_for_pandas = update_with_metrics(dict_for_pandas, metrics_per_type)
    dict_for_pandas = update_with_metrics(dict_for_pandas, exclusive_metrics_per_type)


    dict_of_cross_dataset_metrics[data_dst].append(dict_for_pandas)

  return dict_of_cross_dataset_metrics

# %%

TRANSITIVE = False
PROJECTOR='classifier'


DENOISING_SEED = [10, 20, 30] # None or 10 or 20 or 30
DENOISING_SETUPS = ['TTA', 'TTR', 'TTAR']
DATA_SRC=['ontonotes_shimaoka', 'figer', 'bbn']
DATA_DST=['ontonotes_shimaoka', 'bbn', 'figer', ]
# DATA_DST=['figer']

# %%

from collections import defaultdict

dict_of_cross_dataset_metrics = defaultdict(list)

SEED = 0
VERSION = '' if SEED == 0 else f'-v{SEED}'


for data_src in DATA_SRC:

  MODEL_SRC_PATH = f'/home/remote_hdd/trained_models/{data_src}/et_standard/{PROJECTOR}_adapter_bert_ms{VERSION}.ckpt'

  # prepare model
  model = mm.MainModule.load_ET_Network_for_test_(MODEL_SRC_PATH).cuda()

  cross_dataset_metrics_on_dst = predict_on_dst_given_source_model(model, data_src, None)

  for dst, dst_metrics in cross_dataset_metrics_on_dst.items():

    dict_of_cross_dataset_metrics[dst].extend(dst_metrics)

  for denoising_seed in DENOISING_SEED:    
    for denoising_setup in DENOISING_SETUPS:
      MODEL_SRC_PATH = f'/home/cbernasconi/remote_hdd_denoised/trained_models/ms_december_run_on_denoised_datasets/{data_src}/{denoising_setup}/{denoising_seed}/et_standard/{PROJECTOR}_adapter_bert{VERSION}.ckpt'

      # prepare model
      model = mm.MainModule.load_ET_Network_for_test_(MODEL_SRC_PATH).cuda()

      cross_dataset_metrics_on_dst = predict_on_dst_given_source_model(model, data_src, denoising_seed)
      
      for dst, dst_metrics in cross_dataset_metrics_on_dst.items():

        dict_of_cross_dataset_metrics[dst].extend(dst_metrics)


OUT_DIR = '/home/cbernasconi/et/experiments/cross_dataset/results/'
filename = '{}_cross_dataset_no_training.csv'

os.makedirs(OUT_DIR, exist_ok=True)

for dst_dataset, rows in dict_of_cross_dataset_metrics.items(): 
  df = pd.DataFrame(rows)

  df.to_csv(os.path.join(OUT_DIR, filename.format(dst_dataset)), index=False)

# %%
 