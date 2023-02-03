# %%
import pandas as pd
import json
DATA_SRC = 'figer'
DATA_TGT = 'bbn'
TGT2SRC_PATH = f'/home/cbernasconi/et/experiments/cross_dataset/mappings/{DATA_TGT}2{DATA_SRC}.json'
OUT_PATH = f'/home/cbernasconi/et/experiments/cross_dataset/kbs/{DATA_TGT}2{DATA_SRC}.txt'
TYPES_SRC_PATH =f'/home/remote_hdd/tokenized_datasets/{DATA_SRC}/et_standard/types_list.txt'
TYPES_TGT_PATH =f'/home/remote_hdd/tokenized_datasets/{DATA_TGT}/et_standard/types_list.txt'
HORIZONTAL_KB_PATH =f'/home/cbernasconi/et/experiments/horizontal_clauses/kb/{DATA_TGT}/horizontal_pairs_L0.txt'
# %%
# read mapping
tgt2src = json.loads(open(TGT2SRC_PATH, 'r').read())
# read types to preserve the order of type2id
types_src = open(TYPES_SRC_PATH, 'r').read().splitlines()
types_tgt = open(TYPES_TGT_PATH, 'r').read().splitlines()

# %%
# create predicates
predicates = [f'PSRC{t}' for t in types_src] + [f'PTGT{t}' for t in types_tgt]
predicates_str = ','.join(predicates)
print(predicates)
# %%
# create equivalence/generalization clauses
eq_gen_clauses = []
for t_tgt, t_src in tgt2src.items():
  if t_src:
    clause = f'_:nPSRC{t_src},PTGT{t_tgt}'
    eq_gen_clauses.append(clause)
eq_gen_clauses_str = '\n'.join(eq_gen_clauses)
print(eq_gen_clauses_str)
# %%
# check TGT types with zero co-occorrences
# opt1: paste previous code from create_smart_files.py
# -> opt2: manipulate horizontal kb <-
horizontal_kb = open(HORIZONTAL_KB_PATH, 'r').read().splitlines()
if horizontal_kb[0] != predicates_str[-len(horizontal_kb[0]):]:
  raise Exception('Types order mismatch!')
horizontal_pairs = [pair.split(':')[1].replace('nP', '').split(',') for pair in horizontal_kb[2:]]
horizontal_pairs
# %%
# create disjointness clauses
disjointness_clauses = []
for t1_tgt, t2_tgt in horizontal_pairs:
  t1_src = tgt2src[t1_tgt]
  t2_src = tgt2src[t2_tgt]
  if t1_src:
    clause = f'_:nPSRC{t1_src},nPTGT{t2_tgt}'
    disjointness_clauses.append(clause)
  if t2_src:
    clause = f'_:nPSRC{t2_src},nPTGT{t1_tgt}'
    disjointness_clauses.append(clause)
disjointness_clauses_str = '\n'.join(disjointness_clauses)
print(disjointness_clauses)
# %%
# create kb
kb = predicates_str + '\n\n' + eq_gen_clauses_str + '\n'# + disjointness_clauses_str + '\n'
with open(OUT_PATH, 'w') as f:
  f.write(kb)
print(kb)