# %%
import os
import shutil

DATA = ['bbn', 'figer']
MODEL_DIR = '/home/remote_hdd/trained_models/{data}/cross_dataset_few_shots/'
OUT_DIR = os.path.join(MODEL_DIR, 'k_{shots}/instance_{instance}')
LAST_CHAR_K = {
                '5' : ['s', '3', '6'],
                '10' : ['1', '4', '7'],
                '20' : ['2', '5', '8']
                }
LAST_CHAR_INSTANCE = {
              '0' : ['s', '1', '2'],
              '1' : ['3', '4', '5'],
              '2' : ['6', '7', '8']
            }
# %%
# |block| = 3 = instances
# block(0) = 5-shots; block(1) = 10-shots; block(2) = 20-shots
# 0 1 2 -> instance_0


for data in DATA:
  print(data)
  model_dir = MODEL_DIR.format(data=data)
  for filename in os.listdir(model_dir):
    if filename.endswith('.ckpt'):
      model_path = os.path.join(model_dir, filename)
      last_char = filename[-6]
      # assign K
      for k, chars in LAST_CHAR_K.items():
        if last_char in chars:
          shots = k
          break
      # assign I
      for i, chars in LAST_CHAR_INSTANCE.items():
        if last_char in chars:
          instance = i
          break
      # prepare out dir and filepaths
      out_dir = OUT_DIR.format(shots=shots, instance=instance, data=data)
      os.makedirs(out_dir, exist_ok=True)
      if last_char == 's': # instance 0, no need to remove '-v*'
        model_name = filename
      else:
        model_name = filename[:-8] + '.ckpt'
      out_model_path = os.path.join(out_dir, model_name)
      print('copying:')
      print(model_path)
      print(out_model_path)
      print()
      shutil.copy(model_path, out_model_path)
  print('_'*50)
      
# %%
