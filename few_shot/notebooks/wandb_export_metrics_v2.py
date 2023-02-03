# %%
import wandb
import os
import sys
import pandas as pd
import json 
import warnings
warnings.filterwarnings('ignore')

wandb_api = wandb.Api()
WANDB_ENTITY = 'insides-lab-unimib-wandb'
WANDB_PROJECT = 'cross_dataset_{data}_few_shots_k{k}'
K = [5, 10, 20]
N_INSTANCES = 3

DATA=['figer', 'bbn', 'ontonotes_shimaoka']
# DATA=['figer']
REDIRECT_LOG = False
FREQ_TEST_GT = 2

# set paths
OUT_DIR_PATH = '../results/wandb_export/{data}/{k}'
LOG_FILE_PATH = os.path.join('{}', 'log.txt')
OUT_FILE_DATA = 'metrics.csv'
OUT_FILE_DATA_AGGREGATED = 'metrics_aggregated.csv'
OUT_FILE_DATA_AGGREGATED_SINGLE_TYPES = 'metrics_aggregated_single_types.csv'
OUT_FILE_DATA_MACRO_TYPES_FILTERED = f'macro_types_filtered_ftest_gt{FREQ_TEST_GT}.csv'
OUT_FILE_DATA_TRANSPOSED = 'metrics_transposed.csv'
OUT_FILE_DATA_TRANSPOSED_AGGREGATED = 'metrics_transposed_aggregated.csv'
# %%
def key_ok(key):
  return key.startswith('test') or key == 'epoch' or 'threshold' in key

# %%

for data in DATA:
  df_freq = pd.read_csv(f'./types_freq/{data}.csv')
  types_to_keep = list(df_freq[df_freq['freq_test'] > FREQ_TEST_GT]['type'].values)
  for k in K:
    out_dir_data_path = OUT_DIR_PATH.format(data=data, k=k)
    os.makedirs(out_dir_data_path, exist_ok=True)
    if REDIRECT_LOG:
      sys.stdout = open(LOG_FILE_PATH.format(out_dir_data_path),'wt')
    print()
    print('####### Data', data,'######')
    # save metrics for each run
    df_data = pd.DataFrame(columns=['metric', 'projector', 'instance',
                                    'precision', 'recall', 'f1'])
    df_data_aggregated = pd.DataFrame(columns=['metric', 'projector',
                                    'precision/mean', 'recall/mean', 'f1/mean',
                                    'precision/std', 'recall/std', 'f1/std'])
    

    # get runs from wandb
    wandb_project = WANDB_PROJECT.format(data=data, k=k)
    wandb_path = f'{WANDB_ENTITY}/{wandb_project}'
    print('Processing runs from', wandb_path)
    runs = wandb_api.runs(wandb_path)
    
    # save metrics for each run
    df_project = pd.DataFrame()
    for run in runs:
      run_id = run.id
      name = run.name
      config = json.loads(run.json_config)
      # skip empty run
      if config:
        summary = run.summary
        instance = config['fit']['value']['seed_everything']
        metrics = { k : v for k,v in summary.items() if key_ok(k) }
        row = {
          'name' : name,
          'instance' : 0,
          'run_id' : run_id
        }
        row.update(metrics)

        if len(df_project) > 0:
          df_project = df_project.append(row, ignore_index=True)
        else:
          df_project = pd.DataFrame([row])
      else:
        print('Empty run detected! Run details:')
        print('- url:', f'https://wandb.ai/{wandb_path}/runs/{run_id}')
        print('- name:', name)
        print('- instance:', instance)
    
    # check if the number of runs is correct
    if df_project[['name','instance']].groupby('name').count().mean().values[0] == N_INSTANCES:
      # replace NaN with 0
      df_project = df_project.replace('NaN', 0)

      # prepare and append row for df_data
      cols = list(df_project.columns)
      non_metrics_cols = ['threshold',
                          'epoch', 'name', 'instance', 'run_id']
      cols_metrics = [c for c in cols if c not in non_metrics_cols]
      for i, df_row in df_project.iterrows():
        metric_base_keys = set([c.replace('/precision', '').replace('/recall', '').replace('/f1', '') for c in cols_metrics])
        for metric_base_key in metric_base_keys:
          row = {
            'projector': df_row['name'],
            'instance': df_row['instance'],
            'metric': metric_base_key,
            'precision': df_row[f'{metric_base_key}/precision'],
            'recall': df_row[f'{metric_base_key}/recall'],
            'f1': df_row[f'{metric_base_key}/f1']
          }

          df_data = df_data.append(row, ignore_index=True)
      
      # save transposed df
      print(os.path.join(out_dir_data_path, OUT_FILE_DATA_TRANSPOSED))
      df_project.to_csv(os.path.join(out_dir_data_path, OUT_FILE_DATA_TRANSPOSED), index=False)

      # AGGREGATED METRICS
      # prepare and save aggregated csv
      df_project = df_project.drop(['instance', 'run_id'], axis=1)
      cols_mean = df_project.columns[1:]
      df_project_mean = df_project.groupby('name').mean()
      df_project_mean.columns = [f'{c}/mean' for c in cols_mean]
      df_project_std = df_project.groupby('name').std()
      df_project_std.columns = [f'{c}/std' for c in cols_mean]
      df_project_aggregated = pd.DataFrame.join(df_project_mean, df_project_std)

      # save transposed aggregated df
      df_project_aggregated.to_csv(os.path.join(out_dir_data_path, OUT_FILE_DATA_TRANSPOSED_AGGREGATED), index=False)


      # prepare and append aggregated row for df_data_aggregated
      cols = list(df_project_aggregated.columns)
      non_metrics_cols = ['threshold/mean', 'threshold/std',
                          'epoch/mean', 'epoch/std']
      cols_mean = [c for c in cols if c.endswith('/mean') and c not in non_metrics_cols]
      for projector in df_project_aggregated.index:
        metric_base_keys = set([c.replace('/precision/mean', '').replace('/recall/mean', '').replace('/f1/mean', '') for c in cols_mean])
        for metric_base_key in metric_base_keys:
          row = {
            'projector': projector,
            'metric': metric_base_key,
            'precision/mean': df_project_aggregated.loc[projector, f'{metric_base_key}/precision/mean'],
            'recall/mean': df_project_aggregated.loc[projector, f'{metric_base_key}/recall/mean'],
            'f1/mean': df_project_aggregated.loc[projector, f'{metric_base_key}/f1/mean'],
            'precision/std': df_project_aggregated.loc[projector, f'{metric_base_key}/precision/std'],
            'recall/std': df_project_aggregated.loc[projector, f'{metric_base_key}/recall/std'],
            'f1/std': df_project_aggregated.loc[projector, f'{metric_base_key}/f1/std']
          }

          df_data_aggregated = df_data_aggregated.append(row, ignore_index=True)
      # save global dfs
      df_data.to_csv(f'{out_dir_data_path}/{OUT_FILE_DATA}', index=False)
      df_data_aggregated.to_csv(f'{out_dir_data_path}/{OUT_FILE_DATA_AGGREGATED}', index=False)
      
      # prepare df with only metrics per type
      df_single_types = df_data_aggregated.copy()
      df_single_types = df_single_types[df_single_types['metric'].apply(lambda x: x.startswith('test_'))] 
      df_single_types['metric'] = df_single_types['metric'].apply(lambda x: '/' + x.split('/')[0].replace('test_', '').replace('-', '/'))
      df_single_types.columns = ['type'] + list(df_single_types.columns)[1:]      # save
      df_single_types = df_single_types[df_single_types['type'].isin(types_to_keep)]
      # save
      df_single_types.to_csv(f'{out_dir_data_path}/{OUT_FILE_DATA_AGGREGATED_SINGLE_TYPES}', index=False)

      # preapare df macro types filtered
      df_macro_types = df_single_types.groupby('projector').mean()
      df_macro_types = df_macro_types.fillna(0)
      df_macro_types['f1/mean'] = 2 * df_macro_types['precision/mean'] * df_macro_types['recall/mean'] / (df_macro_types['precision/mean'] + df_macro_types['recall/mean'])
      df_macro_types = df_macro_types.drop(columns=['precision/std', 'recall/std', 'f1/std'])
      df_macro_types.to_csv(f'{out_dir_data_path}/{OUT_FILE_DATA_MACRO_TYPES_FILTERED}', index=True)

    else:
      print('ATTENTION! NUMBER OF RUN IS NOT CORRECT!')
      print('Please check the project', f'https://wandb.ai/{wandb_path}')
    
# %%
