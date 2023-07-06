import torch
import esm
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from axial_attention import AxialAttention


##### Simple MLP #####

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

###### Transformer #####

class Transformer(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.,
                 num_heads: int = 1,
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

###### Axial Attention #####

class AxialAttn(nn.Module):

    def __init__(self,
                 embedding_size: int,
                 n_layers: int,
                 out_dim: int,
                 dropout: float = 0.,
                 num_heads: int = 4,
                 feed_forward_dim: int = 1024):
      
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        # input should be batch x 256 x n_layers x embedding

        self.self_attention = AxialAttention(
                    dim = embedding_size,               # embedding dimension
                    dim_index = 3,         # where is the embedding dimension
                    heads = 1,             # number of heads for multi-head attention
                    num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                    sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
                )

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size-n_layers+1, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embedding_size-n_layers+1)
        )

        self.linear = nn.Linear(embedding_size-n_layers+1, out_dim)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.dropout(x)

        # Self-attention
        x = self.self_attention(x)

        # Pool
        pool = nn.MaxPool2d(kernel_size=self.n_layers, stride=1)
        x = pool(x)

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
