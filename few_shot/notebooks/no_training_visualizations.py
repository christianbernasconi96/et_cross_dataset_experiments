# %%

import pandas as pd
import os
import numpy as np
import json

BASE_PATH = '/home/cbernasconi/et/experiments/cross_dataset_v2/results/'

DEFAULT_COLUMNS = ['src', 'denoising', 'denoising_method']

METRICS_COLUMNS = ['test/micro/','test/macro_example/', 'test/macro_types/']

OUTPUT_FOLDER = os.path.join(BASE_PATH, 'thesis_csvs/')

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# %%

def add_p_r_f(prefix_list):
    enriched_list = []
    for prefix in prefix_list:
        for m in ['precision', 'recall', 'f1']:
            enriched_list.append(prefix + m)
    return enriched_list


# %%

# Load csv no training

filename_template = '{}_cross_dataset_no_training.csv'
dfs = {}

for name in ['figer', 'bbn', 'ontonotes_shimaoka']:
    dfs[name] = pd.read_csv(os.path.join(BASE_PATH, filename_template.format(name)))
    dfs[name].fillna(-1, inplace = True)

# %%

def filter_df_by_values(df, row_filter={}):
  df_filtered = df.copy()
  for key, value in row_filter.items():
    df_filtered = df_filtered[df_filtered[key] == value]
  
  return df_filtered

def filter_columns(df, cols_to_keep):
    return df[cols_to_keep]

# %%

# In-domain results

filtered_dfs = {}

for name in ['figer', 'bbn', 'ontonotes_shimaoka']:
    temp = filter_df_by_values(dfs[name], 
                                    {'src': name, 'denoising': -1})
    filtered_dfs[name] = filter_columns(temp, DEFAULT_COLUMNS + add_p_r_f(METRICS_COLUMNS))


df_to_print = pd.concat(list(filtered_dfs.values()))

df_to_print.to_csv(os.path.join(OUTPUT_FOLDER, 'in_domain.csv'), index=False)

# %%

# cross-domain results

MAPPINGS_PATH = '/home/cbernasconi/et/experiments/cross_dataset/mappings/'

filtered_dfs = {}
dfs_to_save = {}

for dataset_name in ['figer', 'bbn', 'ontonotes_shimaoka']:
    type_cols = [c for c in dfs[dataset_name].columns if 'test_type_all' in c]
    temp = filter_df_by_values(dfs[dataset_name], 
                                    {'denoising': -1})
    filtered_dfs[dataset_name] = filter_columns(temp, 
                                        DEFAULT_COLUMNS + type_cols)

    filenames = [os.path.join(MAPPINGS_PATH, filename) for filename in os.listdir(MAPPINGS_PATH) if f'2{dataset_name}' in filename]
    mappings = [set(list(json.loads(open(f, 'r').read()).values())) for f in filenames]
    covered_types = list(filter(None, set.union(*mappings)))

    covered_types_metrics = []

    for c in type_cols:
        f = False
        for covered in covered_types:
            if c.startswith(f"test_type_all_{covered[1:].replace('/', '-')}"):
                covered_types_metrics.append(c)
                f = True
    #     if not f:
    #         print(f'{dataset_name}:{c}')
    
    # print('-'* 50)
    
    covered_types_metrics = list(set(covered_types_metrics))

    filtered_dfs[dataset_name] = filter_columns(filtered_dfs[dataset_name],
                                                DEFAULT_COLUMNS + covered_types_metrics)

    dfs_to_plots = {}

    df_tmp = filtered_dfs[dataset_name]

    all_types = ['/' + t.replace('test_type_all_', '').replace('/macro_types/f1', '').replace('-', '/') for t in type_cols if 'test_type_all_' in t and '/macro_types/f1' in t]
    
    for index, row in filtered_dfs[dataset_name].iterrows():
        
        df_lines = []
        
        # if row['src'] != dataset_name: 
        if True:
            for c_t in covered_types:
                col_to_keep = 'test_type_all_{}/macro_types/{}'

                c_t_colname = c_t[1:].replace('/', '-') 

                new_line = {'Type': c_t,
                            'precision': row[col_to_keep.format(c_t_colname, 'precision')],
                            'recall': row[col_to_keep.format(c_t_colname, 'recall')],
                            'f1-score': row[col_to_keep.format(c_t_colname, 'f1')],
                            'sourcedataset': row['src']}

                df_lines.append(new_line) 
        
        # else:
        #     for c_t in all_types:
        #         col_to_keep = 'test_type_all_{}/macro_types/{}'

        #         c_t_colname = c_t[1:].replace('/', '-') 
        #         if c_t_colname in list(row.index):
        #             new_line = {'Type': c_t,
        #                         'precision': row[col_to_keep.format(c_t_colname, 'precision')],
        #                         'recall': row[col_to_keep.format(c_t_colname, 'recall')],
        #                         'f1-score': row[col_to_keep.format(c_t_colname, 'f1')],
        #                         'sourcedataset': row['src']}

        #             df_lines.append(new_line) 

        dfs_to_plots[row['src']] = pd.DataFrame(df_lines)
    
    dfs_to_save = pd.concat(list(dfs_to_plots.values()))
    dfs_to_save.to_csv(os.path.join(OUTPUT_FOLDER, f"2{dataset_name}.csv"))


# %%

# Denoising In-domain results

filtered_dfs = {}

for name in ['figer', 'bbn', 'ontonotes_shimaoka']:
    temp = filter_df_by_values(dfs[name], 
                                    {'src': name})
    filtered_dfs[name] = filter_columns(temp, DEFAULT_COLUMNS + add_p_r_f(METRICS_COLUMNS))


df_to_print = pd.concat(list(filtered_dfs.values()))

df_to_print.to_csv(os.path.join(OUTPUT_FOLDER, 'denoising_in_domain.csv'), index=False)

# %%

# cross-domain & denoising


CROSS_DOMAIN_COLUMNS = ['metrics_per_types_GT_K/',
                        'test/micro/',
                        'test/macro_example/']

filtered_dfs = {}
for name in ['figer', 'bbn', 'ontonotes_shimaoka']:
    temp = dfs[name].groupby(['src', 'denoising_method']).mean().reset_index()
    filtered_dfs[name] = filter_columns(temp, DEFAULT_COLUMNS + add_p_r_f(CROSS_DOMAIN_COLUMNS))


    # df_to_print = pd.concat(list(filtered_dfs.values()))

    filtered_dfs[name].to_csv(os.path.join(OUTPUT_FOLDER, f'denoising_cross_domain{name}.csv'), index=False)


# %%
