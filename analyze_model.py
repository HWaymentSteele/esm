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

# USAGE: first arg is .pt model file to analyze, second arg is KEYWORD for plots, third is split (int)

best_model = ProteinBertForSequence2Sequence(version='t6', finetuning_method='axialAttn', embedding_layer='all', missing_loss_weight=1.0).cuda()
dcts = torch.load(sys.argv[1])
best_model.load_state_dict(dcts["model_state_dict"])
_, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

KEYWORD=sys.argv[2]
batch_size=16
dyn_valid = MissingBmrbDataset(split='BMRB_jul04_%s_val' % sys.argv[3] , root_path = os.path.expanduser('/n/home03/wayment/software/'))

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

              P_counts.append(seq.count('P')/len(seq))
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
                  
                  lst.append({'residue': seq[j], 'pred': pred[j], 'assn': assn, 'result': res, 'p_present': p})

melted_results = pd.DataFrame.from_records(lst)
melted_results.to_json("%s_melted_res.json.zip" % KEYWORD)
score = roc_auc_score(melted_results.assn, melted_results.p_present)
print('ROC AUC: %.3f' % score)
