import torch
import esm
import torch.nn as nn
import torch.nn.functional as F
from classifier_modules import *
from torchmetrics.functional import precision, recall, auroc, accuracy, specificity, f1_score
import numpy as np
import gzip
# loss_fct = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', model='focal_loss',
#                 alpha=[0.0001, 1.0, 10.0], gamma=5, reduction='mean', device='cuda', dtype=torch.float32, force_reload=False)

def metrics(logits, labels, ignore_index: int = 0):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
                
        preds = predictions[valid_mask] #only where labels are not 0
        preds = torch.clamp(preds, 1, 2) # clamp predictions of 0 category to "assigned" class
        preds -= 1 # then shift (1,2) to (0,1)
        
        l = labels[valid_mask] #only where labels are not 0
        l -=1 #shift (1,2) to (0,1)
        
        probs = F.softmax(logits[valid_mask])[:,-1] # take only for last category
        
        p = precision(preds, l, task="binary", num_classes=2, average='micro')
        r = recall(preds, l, task="binary", num_classes=2, average='micro')
        acc = accuracy(preds, l, task="binary", num_classes=2, average='micro')
        spec = specificity(preds, l, task="binary", num_classes=2, average='micro')
        f1 = f1_score(preds, l, task="binary", num_classes=2, average='micro')
        au = auroc(probs, l, task="binary", num_classes=2, average='micro')

        return acc, p, r, spec, f1, au
    
class Metrics(nn.Module):

    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return metrics(inputs, target, self.ignore_index)

class SequenceToSequenceClassificationHead(nn.Module):

    def __init__(self,
                finetuning_method: str,
                num_labels: int,
                hidden_size: list = [320],
                ignore_index: int=0,
                missing_loss_weight: float=1.0):

        super().__init__()
        #print('hidden_size', hidden_size)
        self.finetuning_method = finetuning_method

        if 'MLP' in self.finetuning_method:
            self.classify = SimpleMLP(hidden_size[0], 128, num_labels)

        elif 'conv' in self.finetuning_method:
            self.classify = SimpleConv(hidden_size[0], 128, num_labels)
            
        if self.finetuning_method == 'baseline_MLP':
            self.classify = SimpleMLP(hidden_size[0], 128, num_labels)

        if self.finetuning_method == 'baseline_conv':
            self.classify = SimpleConv(hidden_size[0], 128, num_labels)
            
        elif self.finetuning_method == 'attn_single':
            self.classify = AxialAttn(128, num_labels)
            
        if self.finetuning_method == '2D_attn':
            self.classify = AttnOnAttn(300, num_labels)
            
        if self.finetuning_method == '2D_attn_and_1D':
            print('doing 2D attn and 1D')
            self.classify = AttnOnAttn(300, num_labels, add_1D=True)

        self.num_labels = num_labels
        self._ignore_index = ignore_index
        self.loss_weights = torch.FloatTensor([0.0001, 1.0, missing_loss_weight]).cuda()

    def forward(self, sequence_output, targets=None):
        l = self.classify(sequence_output)
        #outputs = (sequence_logits,)
        logits = torch.ones(l.shape[0],l.shape[1],3).cuda()
        logits[:,:,2] = l.squeeze()
        outputs = (l,)
        
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(weight = self.loss_weights, ignore_index=self._ignore_index)
            
            classification_loss = loss_fct(logits.view(-1, 3), targets.view(-1))
            
            acc_fct = Metrics(ignore_index=self._ignore_index)
            acc, p, r, spec, f1, au = acc_fct(logits.view(-1, 3), targets.view(-1))
            metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'specificity': spec, 'f1-score': f1,'auroc': au}

            loss_and_metrics = (classification_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs

class ProteinBertForSequence2Sequence(nn.Module):

    def __init__(self, version='t6',
        embedding_layer='all',
        finetuning_method='conv',
        finetune=True,
        finetune_emb=True,
        need_head_weights=True,
        missing_loss_weight=1.0,
        save_attn_weights=False):
        
        super().__init__()
        self.embedding_layer = embedding_layer
        self.finetuning_method = finetuning_method
        self.num_labels = 1 
        self.version = version
        self.finetune=finetune
        self.finetune_emb = finetune_emb
        self.need_head_weights=need_head_weights
        self.missing_loss_weight = missing_loss_weight
        self.save_attn_weights = save_attn_weights

        if self.version=='t6':
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        elif self.version=='t12':
            model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        elif self.version=='t30':
            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        elif self.version=='t33':
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif self.version=='t36':
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        
        self.bert = model
        if self.embedding_layer != 'all':
            # if not 'all', embedding_layer should be an int less than total layers
            assert(int(self.embedding_layer) < self.bert.num_layers)
                        
        if 'all' in self.finetuning_method:
            self.hidden_size = [model.embed_dim * (self.bert.num_layers+1)]

        elif 'baseline' in self.finetuning_method:
            self.hidden_size = [23] # size of OHE embedding
        else: # old 'single' in self.finetuning_method:
            self.hidden_size = [model.embed_dim]
            
        self.classify = SequenceToSequenceClassificationHead(self.finetuning_method, self.num_labels, self.hidden_size)

    @torch.cuda.amp.autocast()
    def forward(self, input_ids, targets=None, names=None):
        for k, v in self.bert.named_parameters():
            if not self.finetune:
                v.requires_grad = False
            elif not self.finetune_emb and 'embed_tokens.weight' in k:
                v.requires_grad = False
            elif not self.finetune_emb and 'embed_positions.weight' in k:
                v.requires_grad = False

        if self.finetuning_method == 'MLP_all':
            outputs = self.bert(input_ids, repr_layers=range(self.bert.num_layers+1),
                                need_head_weights=self.need_head_weights)
            outs = torch.stack([v for _, v in sorted(outputs["representations"].items())],dim=2)
            shp = outs.shape
            outs = outs.view(shp[0], shp[1], -1)
            outputs = self.classify(outs, targets)
            
        elif 'baseline' in self.finetuning_method:
            ohes = nn.functional.one_hot(input_ids-1, num_classes=23).to(torch.float32) #N,L,22
            outputs = self.classify(ohes, targets)
            
        elif self.finetuning_method == '2D_attn_and_1D':
            emb_layer = int(self.bert.num_layers-1)
            outputs = self.bert(input_ids, repr_layers=[emb_layer],
                                need_head_weights=True)
            attn = outputs['attentions'][:,emb_layer,:,:] # [16, 6, 20, L, L] batch, layers, heads, len, len
            attn = attn.transpose(1,3)
            emb = outputs['representations'][emb_layer]
            outs = (attn, emb)
            outputs = self.classify(outs, targets)
            
        elif self.finetuning_method == '2D_attn':
            emb_layer = int(self.bert.num_layers-1)
            outputs = self.bert(input_ids, repr_layers=[emb_layer],
                                need_head_weights=True)
            outs = outputs['attentions'][:,emb_layer,:,:] # [16, 6, 20, L, L] batch, layers, heads, len, len
            outs = outs.transpose(1,3) 
            batch_size = outs.shape[0]

            if self.save_attn_weights:
                with torch.no_grad():
                    if names is None:
                        raise RuntimeError('Names not present but want to save')

                    numpy_array = outs.detach().cpu().numpy()
                    for i in range(batch_size):
                        # Save the NumPy array as a compressed gzip file
                        with gzip.open('%s_attn/%s.npy.gz' % (self.version, names[i]), 'wb') as f:
                            np.save(f, numpy_array[i])

            outputs = self.classify(outs, targets)
            
        else: # self.finetuning_method == 'conv_single', 'MLP_single','attn_single':
            emb_layer = int(self.bert.num_layers-1)

            outputs = self.bert(input_ids, repr_layers=[emb_layer])
            outs = outputs['representations'][emb_layer] # [batch, seq len, emb]
            outputs = self.classify(outs, targets)

        return outputs

    

class FrozenSeq2Seq(nn.Module):

    def __init__(self):
        super().__init__()
        self.finetuning_method = '2D_attn'
        self.num_labels = 1 #one is for pad class
        self.missing_loss_weight = 1.0
        self.hidden_size = 0
            
        self.classify = SequenceToSequenceClassificationHead(self.finetuning_method, self.num_labels, self.hidden_size)

    @torch.cuda.amp.autocast()
    def forward(self, heads, targets=None):
        outs = self.classify(heads, targets)
        return outs