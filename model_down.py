import torch
import esm
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

# from https://github.com/elttaes/Revisiting-PLMs/blob/90ae45755f176458f8b73eee33eb993aae01d460/Structure/ssp/esm/1b/train.py
# """ / model_down.py

#model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
#batch_converter = alphabet.get_batch_converter()
import torch.nn.functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.):
        super().__init__()
        self.embedding = nn.Linear(in_dim, hid_dim)
        self.positional_encoding = PositionalEncoding(hid_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.transpose(0, 1)
        encoded = self.positional_encoding(embedded)
        transformed = self.transformer(encoded)
        transformed = transformed.transpose(0, 1)
        transformed = self.dropout(transformed)
        output = self.fc(transformed)
        return output.squeeze(2)

class PositionalEncoding(nn.Module):
    def __init__(self, hid_dim, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        pe = torch.zeros(max_len, hid_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hid_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hid_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

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

class SimpleConv2D(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, hid_dim, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Dropout2d(dropout, inplace=True),
            nn.Conv2d(hid_dim, out_dim, kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.main(x)
        x = x.transpose(1, 2).contiguous()
        return x
# def accuracy(logits, labels, ignore_index: int = -100):
#     with torch.no_grad():
#         valid_mask = (labels != ignore_index)
#         predictions = logits.float().argmax(-1)
#         correct = (predictions == labels) * valid_mask
#         return correct.sum().float() / valid_mask.sum().float()

def dyn_accuracy(logits, labels): 
    with torch.no_grad():
        # just care about model labels 2-5 and missing assignment label
        valid_mask = (labels >= 3)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()

class Accuracy(nn.Module):

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return dyn_accuracy(inputs, target) #,  self.ignore_index)

weights = [0.1,0.1,0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 10.0]
class_weights = torch.FloatTensor(weights).cuda()

class SequenceToSequenceClassificationHead(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_labels: int,
                 ignore_index: int = 0):
        super().__init__()
        #self.classify = SimpleConv(hidden_size, 1280, num_labels)
        self.classify = SimpleMLP(hidden_size, 1280, num_labels)
        self.num_labels = num_labels
        self._ignore_index = ignore_index

    def forward(self, sequence_output, targets=None):
        sequence_logits = self.classify(sequence_output)
        outputs = (sequence_logits,)
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=self._ignore_index)
            classification_loss = loss_fct(
                sequence_logits.view(-1, self.num_labels), targets.view(-1))
            acc_fct = Accuracy(ignore_index=self._ignore_index)
            metrics = {'accuracy': acc_fct(sequence_logits.view(-1, self.num_labels), targets.view(-1))}

            loss_and_metrics = (classification_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs

class ProteinBertForSequence2Sequence(nn.Module):

    def __init__(self, version='t6', finetune=True, finetune_emb=True):
        super().__init__()
        self.num_labels = 9
        self.version = version
        self.finetune=finetune
        self.finetune_emb = finetune_emb
        
        if self.version=='t6':
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        elif self.version=='t12':
            model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        elif self.version=='t30':
            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        elif self.version=='t33':
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        
        self.bert = model
        self.classify = SequenceToSequenceClassificationHead(
            model.embed_dim*(self.bert.num_layers+1), self.num_labels)

        print(self.bert.embed_dim)

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
        #outputs = self.bert(input_ids, repr_layers=[self.bert.num_layers])
        #outs = outputs['representations'][self.bert.num_layers]
        outs = torch.stack([v for _, v in sorted(outputs["representations"].items())],dim=2)
        shp = outs.shape
        outs = outs.view(shp[0], shp[1], -1)
        outputs = self.classify(outs, targets)

        return outputs
