# %%
import pandas as pd
import json
DATA_SRC = 'figer'
DATA_DST = 'bbn'
OUT_PATH = f'/home/cbernasconi/et/experiments/cross_dataset/no_training_mappings/{DATA_DST}2{DATA_SRC}.json'
# %%
# prepare mappings
mappings = pd.read_csv(f'./no_training_mappings/{DATA_DST}.csv', index_col=DATA_DST)
mappings = mappings.replace('-', None)
dst2src = { t: mappings.loc[t, DATA_SRC] for t in mappings.index }
dst2src
# %%
# save mapping
with open(OUT_PATH, 'w') as f:
  f.write(json.dumps(dst2src))
# %% load mapping
with open(OUT_PATH, 'r') as f:
  asd = json.loads(f.read())
asd
# %%
