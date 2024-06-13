import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import sys, os
sys.path.append('/home/jupyter-hannah/software/esm')
import esm
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from esm.data import MissingBmrbDataset
from model_down_bigdata import ProteinBertForSequence2Sequence
import gc
import argparse
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")


RDB_PATH='/home/jupyter-hannah/software/RelaxDB/'

def calc_PatK(row,baseline=False, verbose=False):

    assns = np.array([float(list('xA.').index(x))-1 for x in row['assn_str']])
    logits = np.array(row['p_missing'])
    mask = np.where(assns==-1)

    if verbose:
        print(row['assn_str'])
        print(mask)
        print(sorted_assns)
        print(assns_of_top_scores)

    logits[mask] = np.NaN
    assns[mask] = np.NaN

    if baseline:
        logit_sort = np.random.choice(range(len(logits)),size=len(logits),replace=False)
    else:
        logit_sort = np.argsort(logits)

    sorted_logits = np.sort(logits)
    sorted_assns = np.array([assns[x] for x in logit_sort])

    k = int(np.nansum(assns))
    n_nans = mask[0].shape[0]
    assns_of_top_scores = sorted_assns[-1*(k+n_nans):] # taking into account top scores and that nans get sorted to end
    acc = np.nansum(assns_of_top_scores)/k

    return acc, k

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Example script with integer and string arguments')
    parser.add_argument('model', type=str, help='Saved model to analyze')
    parser.add_argument('--keyword', type=str,default='TEST', help='Keyword to save under')
    parser.add_argument('--finetune',action='store_true')
    parser.add_argument('--finetune_emb',action='store_true')
    parser.add_argument('--version', type=str, default='t6', help='ESM version (t6, t12, t30, t33)')
    parser.add_argument('--dataset', type=str,default='test_set_26oct2023', help='name of test set split')
    parser.add_argument('--embedding_layer', type=str, default='all', help='Embeddings to use (default: all)')
    parser.add_argument('--method', type=str, default='axialAttn', help="Finetuning method: 'transformer','MLP','axialAttn'")
    parser.add_argument('--missing_class_wt', type=float, default=1.0, help='weight on missing class for cross entropy loss')
    
    gc.collect()

    args = parser.parse_args()
    missing_loss_weight = args.missing_class_wt
    best_model = ProteinBertForSequence2Sequence(version=args.version,
      finetuning_method=args.method,
      embedding_layer=args.embedding_layer,
      finetune=args.finetune,
      finetune_emb = args.finetune_emb,
      missing_loss_weight=args.missing_class_wt).cuda()

    dcts = torch.load(args.model)
    best_model.load_state_dict(dcts["model_state_dict"])
    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

    dssp = pd.read_csv(RDB_PATH+'dssp_ESMmodels_mBMRB_26oct2023.csv',index_col=0)
    
    if 'relaxdb' in args.dataset:
        rdb = pd.read_json(RDB_PATH+'/101x_testset_alldata.json.zip')
    
    dyn_valid = MissingBmrbDataset(split=args.dataset, root_path = os.path.expanduser('/home/jupyter-hannah/software/'))
    batch_size = len(dyn_valid) # to make getting names work from split file, terrible I know

    valid_loader = DataLoader(dataset=dyn_valid,batch_size=batch_size,shuffle=False,
                        collate_fn=dyn_valid.__collate_fn__,drop_last=True)
    
    convert = ["x", 'A', '.']
    lst=[]
    prot_lst=[] 
    
    for idx, batch in enumerate(valid_loader):
        seqs = torch.squeeze(batch['input_ids'])
        targets = batch['targets']
        inputs, targets = torch.tensor(seqs).cuda(), torch.tensor(targets).cuda()
        with torch.no_grad():
            outputs = best_model(inputs, targets=targets)
            loss_acc, value_prediction = outputs
            
            print_lst = ["%.3f" % v for v in loss_acc[1].values()]
            print(args.model, args.dataset, '\t'.join(print_lst))

            for i in range(batch_size):
                seq = ''.join([alphabet.get_tok(x) for x in seqs[i].cpu().detach().numpy()])
                seq = seq.split('<pad>')[0]
                seq_len = len(seq)
                entry_ID = dyn_valid.names[i]
                
                dssp_str = dssp.loc[entry_ID]['DSSP']

                logits = value_prediction[i].float().cpu().detach().numpy()
                p = F.softmax(value_prediction[i]).float().cpu().detach().numpy()

                p_missing = p[:,-1]
                
                pred = value_prediction[i].float().argmax(-1).cpu().detach().numpy()
                pred = ''.join([convert[int(y)] for y in pred[:seq_len]])
                target = targets[i].cpu().detach().numpy()
                target = ''.join([convert[int(y)] for y in target])[:seq_len]
                if 'relaxdb' in args.dataset:
                    target = rdb.loc[rdb.entry_ID==entry_ID].iloc[0]['assn_str_with_rex_g3'].replace(' ','')[:300]
                    #25113 is longer than 300
                assert len(pred)==seq_len
                assert len(target)==seq_len
                
                start_pos = target.find('A')
                end_pos = target.rfind('A')
                
                prot_lst.append({'sequence': seq, 'assn_str': target,
                  'start_pos': start_pos, 'end_pos': end_pos, 'probs': p, 'logits': logits,
                  'p_missing': p_missing[:seq_len], 'entry_ID': entry_ID, 'dssp': dssp_str})
                
                for j in range(seq_len):
                    if seq[j] != 'P' and j >= start_pos and j <= end_pos:
                        if target[j]=='A':
                            assn=0
                        else: # '.' or 'r'
                            assn=1
                        lst.append({'residue': seq[j], 'pred': pred[j],
                                    'assn': assn,'target': target[j],
                                    'logits': logits[j], 'p_missing': p_missing[j],
                                    'entry_ID': entry_ID, 'seqpos': j, 'dssp': dssp_str[j]})
                        
    melted_results = pd.DataFrame.from_records(lst)
    melted_results.to_json("%s_melted_res.json.zip" % args.keyword)
  
    by_constr_results = pd.DataFrame.from_records(prot_lst)
    by_constr_results.to_json("%s_by_constr.json.zip" % args.keyword)
