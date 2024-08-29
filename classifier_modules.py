import torch
import esm
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from axial_attention import AxialAttention, AxialPositionalEmbedding
import torch.nn.functional as F
import numpy as np

##### Simple MLP #####

class SimpleMLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.main = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Linear(hid_dim, out_dim), dim=None))
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.main(x)
        return x

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
                
class AttnOnAttn(nn.Module):

    def __init__(self,
                 L: int,
                 out_dim: int,
                 dropout: float = 0.1,
                 num_heads: int = 1,
                 add_1D=False):
      
        super().__init__()
        self.L = L
        self.embedding_size = 320
        self.n_heads=20
        self.out_dim = out_dim
        self.sel_res_count = 10
        self.num_labels=1
        self.add_1D = add_1D
        
        self.apply(self._init_weights)
        
        #self.sel_v = nn.Parameter(nn.init.kaiming_normal_(torch.empty(128, self.sel_res_count))).cuda()

        coords = [(i,j) for i in range(self.L) for j in range(self.L)]
        distances = np.array([(j-i) for i,j in coords])
        distances = distances.reshape(self.L, self.L)
        clip_val=32
        self.pos_embedding = torch.tensor(np.clip(distances, -1*clip_val, clip_val) + clip_val).cuda()
        self.pos_embedding_onehot = nn.functional.one_hot(self.pos_embedding, num_classes=clip_val*2+1).cuda().to(torch.float32)
        
        self.lin_inp_128 = weight_norm(nn.Linear(self.n_heads,20),dim=None)
        self.pos_to_128 = weight_norm(nn.Linear(65,20),dim=None)
        
        self.bilinear = FactorizedBilinear(self.embedding_size, self.L, 20)
                
        self.fc = nn.Sequential(weight_norm(nn.Linear(200, 100),dim=None),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                weight_norm(nn.Linear(100, 50),dim=None),
                                nn.ReLU(),
                                nn.Dropout(dropout),     
                                weight_norm(nn.Linear(50,self.num_labels),dim=None),
                               )

        self.sel_v_linear = weight_norm(nn.Linear(20, 10),dim=None)

    def forward(self, outs, embeddings_1d=None):
        if self.add_1D:
            x, embeddings_1d = outs
        else:
            x = outs

        if self.add_1D:
            bilinear = self.bilinear(embeddings_1d)
            x = x + bilinear
            
        x = self.lin_inp_128(x) # N,L,L,20 -> N,L,L,H
        pos = self.pos_to_128(self.pos_embedding_onehot) # L,L,H
        x = x + pos
        #v = x @ self.sel_v #N,L,L,128 -> N,L,L,10

        N, L, _, _ = x.shape
        x_flat = x.view(-1, 20)  # Reshape to (N*L*L, 128) for linear layer
        v = self.sel_v_linear(x_flat)  # Apply the linear layer, result shape (N*L*L, 10)
        v = v.view(N, L, L, 10)  # Reshape back to original dimensions N, L, L, 10
        
        v = F.softmax(v, dim=2)
        selected_values = torch.einsum("...bjx,...bjh->...bhx", x, v) #N,L,10,128
        flattened_sv = selected_values.reshape(x.shape[0],self.L,-1) #N,L,1280
        out = self.fc(flattened_sv)
        return out

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                

class FactorizedBilinear(nn.Module):
    #Use
    #m = FactorizedBilinear(320,300,20)
    def __init__(self, emb, L, rank):
        super(FactorizedBilinear, self).__init__()
        self.linear1 = weight_norm(nn.Linear(emb, rank, bias=False),dim=None)
        self.linear2 = weight_norm(nn.Linear(emb, rank, bias=False),dim=None)
        self.output = weight_norm(nn.Linear(rank, rank, bias=False),dim=None)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        dot = torch.einsum('bij,bkj->bikj', x1, x2)
        return self.output(dot)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                