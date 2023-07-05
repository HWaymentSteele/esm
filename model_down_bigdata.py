import torch
import esm
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

# from https://github.com/elttaes/Revisiting-PLMs/blob/90ae45755f176458f8b73eee33eb993aae01d460/Structure/ssp/esm/1b/train.py
# """ / model_down.py

#model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
#batch_converter = alphabet.get_batch_converter()

def _init_weights(module):
    """ Initialize the weights """
    if isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.utils.weight_norm(nn.Linear(hid_dim, out_dim), dim=None))
        self.apply(_init_weights)

    def forward(self, x):
        return self.main(x)

class SimpleConv(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm1d(in_dim), 
            weight_norm(nn.Conv1d(in_dim, hid_dim, 5, padding=2), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Conv1d(hid_dim, out_dim, 3, padding=1), dim=None))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.main(x)
        x = x.transpose(1, 2).contiguous()
        return x

import torch.nn.functional as F


class Transformer(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.,
                 num_heads: int = 4,
                 feed_forward_dim: int = 2048):
        super().__init__()

        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.self_attention = nn.MultiheadAttention(hid_dim, num_heads)
        self.linear2 = nn.Linear(hid_dim, out_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hid_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, hid_dim)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)

        # Self-attention
        x = x.permute(1, 0, 2)  # Reshape for multihead attention
        x, _ = self.self_attention(x, x, x)
        x = x.permute(1, 0, 2)  # Reshape back

        # Feed-forward
        x = self.feed_forward(x)

        x = self.linear2(x)
        return x

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

def accuracy(logits, labels, ignore_index: int = 0):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()

class Accuracy(nn.Module):

    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return accuracy(inputs, target) #,  self.ignore_index)


class SequenceToSequenceClassificationHead(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_labels: int,
                 ignore_index: int = 0,
                 missing_loss_weight: float=1.0):
        super().__init__()
        print('hidden_size', hidden_size)
        self.classify = Transformer(hidden_size, 1280, num_labels)
        #self.classify = SimpleMLP2D(hidden_size_1, hidden_size_2, 1280, num_labels)

        self.num_labels = num_labels
        self._ignore_index = ignore_index
        self.loss_weights = torch.FloatTensor([1.0,1.0, missing_loss_weight]).cuda()

    def forward(self, sequence_output, targets=None):
        sequence_logits = self.classify(sequence_output)
        outputs = (sequence_logits,)
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(weight = self.loss_weights, ignore_index=self._ignore_index)
            classification_loss = loss_fct(
                sequence_logits.view(-1, self.num_labels), targets.view(-1))
            acc_fct = Accuracy(ignore_index=self._ignore_index)
            metrics = {'accuracy': acc_fct(sequence_logits.view(-1, self.num_labels), targets.view(-1))}

            loss_and_metrics = (classification_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs

class ProteinBertForSequence2Sequence(nn.Module):

    def __init__(self, version='t6', finetune=True, finetune_emb=True, missing_loss_weight=1.0):
        super().__init__()
        self.num_labels = 3 #6
        self.version = version
        self.finetune=finetune
        self.finetune_emb = finetune_emb
        self.missing_loss_weight = missing_loss_weight

        if self.version=='t6':
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        elif self.version=='t12':
            model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        elif self.version=='t30':
            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        elif self.version=='t33':
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        
        self.bert = model
        print('num_layers, embed_dim', self.bert.num_layers+1, model.embed_dim)
        self.classify = SequenceToSequenceClassificationHead(
            model.embed_dim*(self.bert.num_layers+1),
            self.num_labels, missing_loss_weight=self.missing_loss_weight)
        #self.classify = SequenceToSequenceClassificationHead(
        #    model.embed_dim, self.bert.num_layers+1,
        #    self.num_labels, missing_loss_weight=self.missing_loss_weight)


    @torch.cuda.amp.autocast()
    def forward(self, input_ids, targets=None):
        for k, v in self.bert.named_parameters():
            if not self.finetune:
                v.requires_grad = False
            elif not self.finetune_emb and 'embed_tokens.weight' in k:
                v.requires_grad = False
            elif not self.finetune_emb and 'embed_positions.weight' in k:
                v.requires_grad = False
                 
        outputs = self.bert(input_ids, repr_layers=range(self.bert.num_layers+1))
        outs = torch.stack([v for _, v in sorted(outputs["representations"].items())],dim=2)
        shp = outs.shape
        outs = outs.view(shp[0], shp[1], -1)
        outputs = self.classify(outs, targets)

        return outputs
