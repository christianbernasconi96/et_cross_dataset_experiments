# %%
import pandas as pd
DATA_SRC = 'bbn'
DATA_DST = 'figer'
TYPES_PATH_SRC = f'/home/remote_hdd/datasets/{DATA_SRC}/all_types.txt'
TYPES_PATH_DST = f'/home/remote_hdd/datasets/{DATA_DST}/all_types.txt'
# TYPES_PATH_SRC = f'/home/remote_hdd/tokenized_datasets/{DATA_SRC}/et_standard/types_list.txt'
# TYPES_PATH_DST = f'/home/remote_hdd/tokenized_datasets/{DATA_DST}/et_standard/types_list.txt'
WEIGHT = '_'
# %%
# prepare mappings
mappings = pd.read_csv(f'./cross_datasets_mappings/{DATA_SRC}.csv', index_col=DATA_SRC)
mappings = mappings.replace('-', None)
# print(mappings.loc['/ORGANIZATION/CORPORATION','ontonotes'])
types_src = open(TYPES_PATH_SRC, 'r').read().splitlines()
types_dst = open(TYPES_PATH_DST, 'r').read().splitlines()
src2dst = { t: mappings.loc[t, DATA_DST] for t in types_src }
# %%
# create direct mapping clauses (i.e., for mapping /SUBSTANCE/FOOD -> /FOOD)
direct_mapping_clauses = [f'{WEIGHT}:nP{t_src},P{t_dst}' for t_src, t_dst in src2dst.items() if t_dst]
print('\n'.join(direct_mapping_clauses))
# %%
# create trasversal mapping clauses (i.e. for mapping /LOCATION/REGION -> /location)
# TODO: modify csv... trasversal mapping will be implicitly included in direct mapping
# %%
# create negative clauses for unmapped types
types_dst_unmapped = list(set(types_dst) - set(src2dst.values()))
negative_clauses = [ f'{WEIGHT}:nP{t_src},nP{t_dst}' for t_src in types_src for t_dst in types_dst_unmapped]
print('\n'.join(negative_clauses))
# NOTE: attenzione che se non si hanno i trasversal in un src2dst_clause dedicato si introducono delle clausole negative non volute...
# %%
