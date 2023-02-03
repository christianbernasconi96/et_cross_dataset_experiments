# %%

types_PATH = '/home/remote_hdd/tokenized_datasets/ontonotes_shimaoka/et_standard/types_list.txt'

types = [t.replace('\n', '') for t in open(types_PATH).readlines()]

# %%

import entity_typing_framework.EntityTypingNetwork_classes.KENN_networks.kenn_utils as kenn_utils
import os

KB_MODE = ['bottom_up', 'top_down', 'hybrid']

DATA = 'ontonotes_shimaoka'

DST_KB_DIR = f'/home/cbernasconi/et/experiments/cross_dataset_v2/kenn_tmp/{DATA}'


tree = kenn_utils.create_tree(types, label2pred = True)
for kb_mode in KB_MODE:
    method = f"kenn_utils.generate_{kb_mode}"
    vertical_clauses = eval(method)(tree, '_')
    clauses = vertical_clauses
    kb = clauses
    # save KB
    filename = f'{kb_mode}.txt'
    dst_kb_dir = DST_KB_DIR
    os.makedirs(dst_kb_dir, exist_ok=True)
    with open(os.path.join(dst_kb_dir, filename), 'w') as out:
        out.write(kb)
# %%
