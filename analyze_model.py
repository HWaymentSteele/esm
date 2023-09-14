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
import matplotlib.pyplot as plt

# USAGE: first arg is .pt model file to analyze, second arg is KEYWORD for plots, third is split (int)

best_model = ProteinBertForSequence2Sequence(version='t12', finetuning_method='axialAttn', embedding_layer='all', missing_loss_weight=1.0).cuda()
dcts = torch.load(sys.argv[1])
best_model.load_state_dict(dcts["model_state_dict"])
_, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

KEYWORD=sys.argv[2]
batch_size=16
dyn_valid = MissingBmrbDataset(split='BMRB_jul04_%s_val' % sys.argv[3] , root_path = os.path.expanduser('/n/home03/wayment/software/'))

valid_loader = DataLoader(dataset=dyn_valid,batch_size=batch_size,shuffle=True,
                    collate_fn=dyn_valid.__collate_fn__,drop_last=True)

convert = ['0', 'a','.']
pred_frac_assns, true_frac_assns, P_counts = [],[],[]
logit_melted=[]
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
              pred_frac_assns.append(pred.count('.')/len(pred))

              target = targets[i].cpu().detach().numpy()
              target = ''.join([convert[int(y)] for y in target])[:seq_len]
              true_frac_assns.append(target.count('.')/len(target))

              P_counts.append(seq.count('P')/len(seq))
              assert len(pred)==seq_len
              assert len(target)==seq_len

              start_pos = target.find('a')
              end_pos = target.rfind('a')
              for j in range(seq_len):
                if seq[j] != 'P' and j >= start_pos and j <= end_pos:
                  if pred[j]=='a' and target[j]=='.':
                    res = 'FN'
                  elif pred[j] == 'a' and target[j] == 'a':
                    res = 'TN'
                  elif pred[j] == '.' and target[j] == 'a':
                    res = 'FP'
                  elif pred[j] == '.' and target[j] == '.':
                    res = 'TP'
                  lst.append({'residue': seq[j], 'pred': pred[j], 'true': target[j], 'result': res, 'logits': logits[j],
                            'true_frac_missing':target.count('.')/len(target), 'pred_frac_missing': pred.count('.')/len(pred)})

melted_results = pd.DataFrame.from_records(lst)
tmp = melted_results.groupby("residue")["result"].value_counts(normalize=True).mul(100).round(2).unstack()
melted_results.to_json("%s_melted_res.json.zip" % KEYWORD)
plt.figure(figsize=(6,4))
plt.bar(tmp.index, tmp['FN'], label = "FN",color='k') 
plt.bar(tmp.index, tmp['FP'], bottom=tmp['FN'], label = "FP",color='grey') 
plt.bar(tmp.index, tmp['TN'], bottom=tmp['FN']+tmp['FP'], label = "TN") 
plt.bar(tmp.index, tmp['TP'], bottom=tmp['TN']+tmp['FN']+tmp['FP'], label = "TP") 
plt.legend()
plt.savefig('%s_breakdown_by_residue_type.pdf'% KEYWORD, bbox_inches='tight')


plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
rp, _ = pearsonr(true_frac_assns, P_counts)
rs, _ = spearmanr(true_frac_assns, P_counts)
plt.scatter(true_frac_assns, P_counts,alpha=0.2)
plt.ylabel('Fraction "P"')
plt.xlabel('True fraction missing')
plt.ylim([-0.1,1])


plt.title("R_p = %.2f, R_s = %.2f" % (rp, rs))

plt.subplot(1,2,2)
rp, _ = pearsonr(true_frac_assns, pred_frac_assns)
rs, _ = spearmanr(true_frac_assns, pred_frac_assns)
plt.scatter(true_frac_assns, pred_frac_assns,alpha=0.2)
plt.ylim([-0.1,1])

plt.title("R_p = %.2f, R_s = %.2f" % (rp, rs))
plt.ylabel('Predicted fraction missing')
plt.xlabel('True fraction missing')

plt.tight_layout()
plt.savefig('%s_corrs_total_fraction_missing.pdf'% KEYWORD, bbox_inches='tight')
