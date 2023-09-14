import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import sys, os
sys.path.append('/n/home03/wayment/software/esm/')
import esm
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data.dataloader import DataLoader
from esm.data import MissingBmrbDataset
from model_down_bigdata import ProteinBertForSequence2Sequence
print(torch.cuda.get_device_name(0))
import gc
import argparse
from sklearn.metrics import roc_auc_score

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Example script with integer and string arguments')
    parser.add_argument('model', type=str, help='Saved model to analyze')
    parser.add_argument('--keyword', type=str,default='TEST', help='Keyword to save under')
    parser.add_argument('--split', type=int, default=0, help='An integer input')
    parser.add_argument('--version', type=str, default='t6', help='ESM version (t6, t12, t30, t33)')
    parser.add_argument('--embedding_layer', type=str, default='all', help='Embeddings to use (default: all)')
    parser.add_argument('--finetuning_method', type=str, default='axialAttn', help="Finetuning method: 'transformer','MLP','axialAttn'")
    parser.add_argument('--missing_class_wt', type=float, default=1.0, help='weight on missing class for cross entropy loss')

    args = parser.parse_args()
    batch_size=16
    finetune=True
    finetune_emb=True
    missing_loss_weight = args.missing_class_wt
    best_model = ProteinBertForSequence2Sequence(version=args.version, finetuning_method=args.finetuning_method,
                                                 embedding_layer=args.embedding_layer, missing_loss_weight=args.missing_class_wt).cuda()
    dcts = torch.load(args.model)
    best_model.load_state_dict(dcts["model_state_dict"])
    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    
    batch_size=16
    dyn_valid = MissingBmrbDataset(split='BMRB_jul04_%s_val' % args.split, root_path = os.path.expanduser('/n/home03/wayment/software/'))
    
    valid_loader = DataLoader(dataset=dyn_valid,batch_size=batch_size,shuffle=True,
                        collate_fn=dyn_valid.__collate_fn__,drop_last=True)
    
    convert = ['0', 'a','.']
    lst=[]
    
    for idx, batch in enumerate(valid_loader):
            seqs = torch.squeeze(batch['input_ids'])
            targets = batch['targets']
            inputs, targets = torch.tensor(seqs).cuda(), torch.tensor(targets).cuda()
            with torch.no_grad():
                outputs = best_model(inputs, targets=targets)
                loss_acc, value_prediction = outputs
    
                for i in range(batch_size):
                  seq = ''.join([alphabet.get_tok(x) for x in seqs[i].cpu().detach().numpy()])
                  seq = seq.split('<pad>')[0]
                  seq_len = len(seq)
    
                  logits = value_prediction[i].float().cpu().detach().numpy()
    
                  pred = value_prediction[i].float().argmax(-1).cpu().detach().numpy()
                  pred = ''.join([convert[int(y)] for y in pred[:seq_len]])
    
                  target = targets[i].cpu().detach().numpy()
                  target = ''.join([convert[int(y)] for y in target])[:seq_len]
    
                  assert len(pred)==seq_len
                  assert len(target)==seq_len
    
                  start_pos = target.find('a')
                  end_pos = target.rfind('a')
                  for j in range(seq_len):
                    if seq[j] != 'P' and j >= start_pos and j <= end_pos:
                      if target[j]=='.':
                        assn=0
                      elif target[j]=='a':
                        assn=1
                      p= np.exp(logits[1])/(np.exp(logits[1])+np.exp(logits[2]))
                      
                      lst.append({'residue': seq[j], 'pred': pred[j], 'assn': assn, 'p_present': p})
    
    melted_results = pd.DataFrame.from_records(lst)
    melted_results.to_json("%s_melted_res.json.zip" % args.keyword)
    score = roc_auc_score(melted_results.assn, melted_results.p_present)
    print('ROC AUC: %.3f' % score)
