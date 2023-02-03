# %%

import json


# %%

PATH = '/home/remote_hdd/datasets/ontonotes_shimaoka/test.json'

lines = open(PATH, 'r').read().split('\n')[:-1]

correct_types = [eval(l)['y_str'] for l in lines]
# %%
dummy_predicted_types = [['/other'] for _ in correct_types]
# %%

total_annotations = 0
total_other = 0
correct_in_examples = []
other_in_examples = []

for pred, correct in zip(dummy_predicted_types, correct_types):
    if '/other' in correct:
        total_other += 1
        other_in_examples.append(1)
    else:
        other_in_examples.append(0)
    
    total_annotations += len(correct)
    correct_in_examples.append(len(correct))



# %%

import numpy as np
micro_precision = 1
micro_recall = total_other / total_annotations
micro_f1 = (2 * micro_precision * micro_recall)  / (micro_precision + micro_recall)
# %%


macro_example_precision = np.mean([p for p in other_in_examples])
macro_example_recall = np.mean([p / total for p, total in zip(other_in_examples, correct_in_examples)])
macro_example_f1 = (2 * macro_example_precision * macro_example_recall)  / (macro_example_precision + macro_example_recall)



# %%
