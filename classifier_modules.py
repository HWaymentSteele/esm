import torch
import esm
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from axial_attention import AxialAttention, AxialPositionalEmbedding
import torch.nn.functional as F


##### Simple MLP #####

class SimpleMLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.15):
        super().__init__()
        self.main = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Linear(hid_dim, out_dim), dim=None))
        self.apply(self._init_weights)

    def forward(self, x):
        return self.main(x)

    @staticmethod
    def _init_weights(module):
        """ Initialize the weights """
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

##### Simple Conv #####

class SimpleConv(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        #print(in_dim, hid_dim, out_dim) # 480, 1280, 3
        
        self.main = nn.Sequential(
            nn.BatchNorm1d(in_dim), 
            weight_norm(nn.Conv1d(in_dim, hid_dim, 5, padding=2), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(hid_dim, out_dim, 3, padding=1), dim=None))

    def forward(self, x):
        x = x.transpose(1, 2) # N,L,E -> N,E,L
        x = self.main(x) # N,E,L -> N,3,L
        x = x.transpose(1, 2).contiguous() # N,3,L -> N,L,3
        return x
            
class AxialAttn(nn.Module):

    def __init__(self,
                 embedding_size: int,
                 out_dim: int,
                 dropout: float = 0.15,
                 num_heads: int = 1):
      
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_size = embedding_size
        # input should be batch x CROP_LEN x embedding
        
        self.pos_emb = AxialPositionalEmbedding( dim = 1, #embedding_size
        shape = (300, embedding_size),
                 emb_dim_index=3
                    )

        self.self_attention = AxialAttention(
                    dim = 1, #embedding_size,               # embedding dimension
                    dim_index = 3,         # where is the embedding dimension
                    heads = 1,             # number of heads for multi-head attention
                    num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                    sum_axial_out = False   # whether to sum contributions of attention on each axis, or run input thru sequentially.
                )

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)
        )

        self.linear = nn.Linear(embedding_size, out_dim)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.dropout(x)
        x= x.unsqueeze(-1) # add one more dim, to treat embedding as another axis
        # Self-attention
        x = self.pos_emb(x)
        x = self.self_attention(x)
        
        
        x = x.squeeze()

        # Feed-forward
        x = self.feed_forward(x)
        x = self.linear(x)

        return x

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)