# %%

from collections import defaultdict
import numpy as np
import json

def check_full(picked_examples, shots, all_types):
    for t in all_types:
        if picked_examples[t] < shots:
            return False
    
    return True


def filter_types_by_hierarchy(types, type_to_keep, son2anc):
    new_types = []

    for t in types:
        if t == type_to_keep or t in son2anc[type_to_keep]:
            new_types.append(t)
    
    return new_types

# %%

OUT_DIR = 'K_SHOT_DATASETS/'

# %%

for d in ['bbn', 'figer', 'ontonotes_shimaoka']:
    
    DATASET_PATH = f'/home/remote_hdd/datasets/{d}/train.json'
    TYPES_FILE = f'/home/remote_hdd/datasets/{d}/all_types.txt'

    for seed in [0, 1, 2]:
        for shots in [40]:
            print(f'{d}_{seed}_{shots}')

            OUTPUT_NAME = f'{d}_k{shots}_s{seed}.json'

            types = [l.replace('\n', '') for l in open(TYPES_FILE, 'r').readlines()]

            fathers = set(['/' + t.split('/')[1] for t in types if len(t.split('/')) > 2])
            fathers = fathers.union(set(['/' + t.split('/')[1] + '/' + t.split('/')[2] for t in types if len(t.split('/')) > 3]))
            son2fat = {t : '/'.join(t.split('/')[0:-1]) for t in types}


            son2anc = {}

            for son, fat in son2fat.items():
                son2anc[son] = [fat]
                if fat in son2fat and son2fat[fat] != '':
                    son2anc[son].append(son2fat[fat])


            lines = [eval(l) for l in open(DATASET_PATH, 'r').readlines()]

            np.random.seed(seed)

            np.random.shuffle(lines)

            picked_train_examples = defaultdict(int)
            few_shot_train_examples = []

            idx_l = 0

            train_full = False
            dev_full = False

            while idx_l < len(lines) and not train_full:
                
                picked = False

                l = lines[idx_l]
                
                if not train_full:
                    i = 0
                    while not picked and i < len(l['y_str']):

                        t = l['y_str'][i]

                        if t not in fathers:
                            if picked_train_examples[t] < shots:
                                picked_train_examples[t] += 1
                                
                                for anc in son2anc[t]:
                                    if anc != '':
                                        picked_train_examples[anc] += 1
                                
                                new_types = filter_types_by_hierarchy(l['y_str'], t, son2anc)

                                l['y_str'] = new_types

                                few_shot_train_examples.append(l)

                                picked = True
                                
                        i += 1

                idx_l += 1

                train_full = check_full(picked_train_examples, shots, types)

            with open(OUT_DIR + 'train_' + OUTPUT_NAME, 'w') as out:
                for l in few_shot_train_examples:
                    out.write(str(l) + '\n')

print('done')
# %%

for d in ['bbn', 'figer', 'ontonotes_shimaoka']:
    
    TYPES_FILE = f'/home/remote_hdd/datasets/{d}/all_types.txt'

    for seed in [0, 1, 2]:
        DATASET_PATH = f'K_SHOT_DATASETS/{d}/train_{d}_k40_s{seed}.json'
        for shots in [5, 10, 20]:
            print(f'{d}_{seed}_{shots}')

            OUTPUT_NAME = f'{d}_k{shots}_s{seed}.json'

            types = [l.replace('\n', '') for l in open(TYPES_FILE, 'r').readlines()]

            fathers = set(['/' + t.split('/')[1] for t in types if len(t.split('/')) > 2])
            fathers = fathers.union(set(['/' + t.split('/')[1] + '/' + t.split('/')[2] for t in types if len(t.split('/')) > 3]))
            son2fat = {t : '/'.join(t.split('/')[0:-1]) for t in types}


            son2anc = {}

            for son, fat in son2fat.items():
                son2anc[son] = [fat]
                if fat in son2fat and son2fat[fat] != '':
                    son2anc[son].append(son2fat[fat])


            lines = [eval(l) for l in open(DATASET_PATH, 'r').readlines()]

            picked_train_examples = defaultdict(int)
            few_shot_train_examples = []
            idx_l = 0

            train_full = False

            while idx_l < len(lines) and not train_full:
                
                picked = False

                l = lines[idx_l]
                
                if not train_full:
                    i = 0
                    while not picked and i < len(l['y_str']):

                        t = l['y_str'][i]

                        if t not in fathers:
                            if picked_train_examples[t] < shots:
                                picked_train_examples[t] += 1
                                
                                for anc in son2anc[t]:
                                    if anc != '':
                                        picked_train_examples[anc] += 1
                                
                                new_types = filter_types_by_hierarchy(l['y_str'], t, son2anc)

                                l['y_str'] = new_types

                                few_shot_train_examples.append(l)

                                picked = True
                                
                        i += 1

                idx_l += 1

                train_full = check_full(picked_train_examples, shots, types)

            with open(OUT_DIR + f'{d}/' + 'train_' + OUTPUT_NAME, 'w') as out:
                for l in few_shot_train_examples:
                    out.write(json.dumps(l) + '\n')

            # DEV

            picked_dev_examples = defaultdict(int)
            few_shot_dev_examples = []
            
            idx_l = 1

            dev_full = False

            while idx_l < len(lines) and not dev_full:
                
                picked = False

                l = lines[-idx_l]
                
                if not dev_full:
                    i = 0
                    while not picked and i < len(l['y_str']):

                        t = l['y_str'][i]

                        if t not in fathers:
                            if picked_dev_examples[t] < shots:
                                picked_dev_examples[t] += 1
                                
                                for anc in son2anc[t]:
                                    if anc != '':
                                        picked_dev_examples[anc] += 1
                                
                                new_types = filter_types_by_hierarchy(l['y_str'], t, son2anc)

                                l['y_str'] = new_types

                                few_shot_dev_examples.append(l)

                                picked = True
                                
                        i += 1

                idx_l += 1

                dev_full = check_full(picked_dev_examples, shots, types)

            with open(OUT_DIR + f'{d}/' + 'dev_' + OUTPUT_NAME, 'w') as out:
                for l in few_shot_dev_examples:
                    out.write(json.dumps(l) + '\n')
print('done')

# %%
