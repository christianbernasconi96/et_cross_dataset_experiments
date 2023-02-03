import math
from tqdm import tqdm
import torch
import gc

def forward_incremental(model, input_ids, attention_mask, one_hot_types, get_output_for_inference=None, batch_size = 32):
  model.cuda()
  model.eval()
  n_batches = math.ceil(len(input_ids) / batch_size)
  y_preds_pretrain = []
  y_preds_incremental = []
  y_true = []
  
  for i in tqdm(range(n_batches)):
    input_ids_batch = input_ids[i*batch_size:(i+1)*batch_size].cuda()
    attention_mask_batch = attention_mask[i*batch_size:(i+1)*batch_size].cuda()
    one_hot_types_batch = one_hot_types[i*batch_size:(i+1)*batch_size].cuda()
    batch = [input_ids_batch, attention_mask_batch, one_hot_types_batch]
    y_preds_batch, y_true_batch = model.forward(batch)
    if get_output_for_inference:
      y_preds_batch = get_output_for_inference(None, y_preds_batch)
    # move to cpu
    y_preds_pretrain_batch_cpu = y_preds_batch[0].detach().cpu()
    y_preds_incremental_batch_cpu = y_preds_batch[1].detach().cpu()
    y_true_batch_cpu = y_true_batch.detach().cpu()
    # free gpu
    del y_preds_batch, y_true_batch
    torch.cuda.empty_cache()
    gc.collect()
    y_preds_pretrain += y_preds_pretrain_batch_cpu
    y_preds_incremental += y_preds_incremental_batch_cpu
    y_true += y_true_batch_cpu

  y_preds = (torch.vstack(y_preds_pretrain), torch.vstack(y_preds_incremental))
  y_true = torch.vstack(y_true)
  model.cpu()

  return y_preds, y_true


def forward_et_standard(model, input_ids, attention_mask, one_hot_types, get_output_for_inference=None, batch_size = 32):
  model.cuda()
  model.eval()
  n_batches = math.ceil(len(input_ids) / batch_size)
  y_preds = []
  y_true = []
  
  for i in tqdm(range(n_batches)):
    input_ids_batch = input_ids[i*batch_size:(i+1)*batch_size].cuda()
    attention_mask_batch = attention_mask[i*batch_size:(i+1)*batch_size].cuda()
    one_hot_types_batch = one_hot_types[i*batch_size:(i+1)*batch_size].cuda()
    batch = [input_ids_batch, attention_mask_batch, one_hot_types_batch]
    y_preds_batch, y_true_batch = model.forward(batch)
    if get_output_for_inference:
      y_preds_batch = get_output_for_inference(None, y_preds_batch)
    # move to cpu
    y_preds_batch_cpu = y_preds_batch.detach().cpu()
    y_true_batch_cpu = y_true_batch.detach().cpu()
    # free gpu
    del y_preds_batch, y_true_batch
    torch.cuda.empty_cache()
    gc.collect()
    y_preds += y_preds_batch_cpu
    y_true += y_true_batch_cpu

  y_preds = torch.vstack(y_preds)
  y_true = torch.vstack(y_true)
  model.cpu()

  return y_preds, y_true