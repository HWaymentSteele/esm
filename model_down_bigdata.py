import torch
import esm
import torch.nn as nn
from classifier_modules import *

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
                embedding_layer: str,
                finetuning_method: str,
                num_labels: int,
                hidden_size: list = [320],
                ignore_index: int=0,
                missing_loss_weight: float=1.0):

        super().__init__()
        print('hidden_size', hidden_size)
        self.finetuning_method = finetuning_method
        self.embedding_layer = embedding_layer

        if self.finetuning_method == 'transformer' and self.embedding_layer != 'all':
            self.classify = Transformer(hidden_size[0], 1280, num_labels)

        elif 'MLP' in self.finetuning_method:
            self.classify = SimpleMLP(hidden_size[0], 1280, num_labels)

        elif self.finetuning_method == 'axialAttn' and self.embedding_layer == 'all':
            self.classify = AxialAttn(hidden_size[0], hidden_size[1], num_labels)


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

    def __init__(self, version='t6',
        embedding_layer='all',
        finetuning_method='conv',
        finetune=True,
        finetune_emb=True,
        missing_loss_weight=1.0):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.finetuning_method = finetuning_method
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
        if self.embedding_layer != 'all':
            # if not 'all', embedding_layer should be an int less than total layers
            assert(int(self.embedding_layer) < self.bert.num_layers)
            
        print('num_layers, embed_dim', self.bert.num_layers+1, model.embed_dim)

        if self.finetuning_type == 'axialAttn':
            self.hidden_size = [model.embed_dim, self.bert.num_layers+1]
        elif self.finetuning_type == 'MLP_all':
            self.hidden_size = [model.embed_dim * self.bert.num_layers+1]
        elif self.finetuning_type == 'MLP_single':
            self.hidden_size = [model.embed_dim]

        self.classify = SequenceToSequenceClassificationHead(self.embedding_layer,
            self.finetuning_method, self.num_labels, self.hidden_size)

            # model.embed_dim*(self.bert.num_layers+1),
            # self.num_labels, missing_loss_weight=self.missing_loss_weight)

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
                 
        if self.finetuning_type == 'axialAttn':
            # not flattening. output should be batch x 256 x n_layers x embedding
            outputs = self.bert(input_ids, repr_layers=range(self.bert.num_layers+1))
            outs = torch.stack([v for _, v in sorted(outputs["representations"].items())],dim=2)
            #shp = outs.shape
            #outs = outs.view(shp[0], shp[1], -1)
            outputs = self.classify(outs, targets)

        elif self.finetuning_type == 'MLP_all':
            outputs = self.bert(input_ids, repr_layers=range(self.bert.num_layers+1))
            outs = torch.stack([v for _, v in sorted(outputs["representations"].items())],dim=2)
            shp = outs.shape
            outs = outs.view(shp[0], shp[1], -1)
            outputs = self.classify(outs, targets)
            
        elif self.finetuning_type == 'MLP_single':
            emb_layer = int(self.embedding_layer)
            outputs = self.bert(input_ids, repr_layers=[emb_layer])
            outs = outputs['representations'][emb_layer]
            outputs = self.classify(outs, targets)

        return outputs
