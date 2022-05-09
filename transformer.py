import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat

class VIT(nn.Module):
    def __init__(self, input_channels=3, class_num=555, patch_size=28, image_size=224, dim=128, nhead=8, dropout=0.5, num_layers=6):
        super().__init__()

        pw = patch_size
        self.patch_dim = pw * pw * input_channels 
        self.trans_dim = dim
        self.class_num = class_num
        self.in_channels = input_channels

        patches = image_size // patch_size
        self.num_patches = patches * patches

        # self.pre_transformer = nn.ModuleList([])
        self.pre_transformer = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = pw, p2 = pw),
            nn.Linear(self.patch_dim, self.trans_dim),
        )

        self.classification_token = nn.Parameter(torch.randn(1,1,self.trans_dim))
        self.pos_embeddings = nn.Parameter(torch.randn(1,self.num_patches+1,self.trans_dim))

        layer = nn.TransformerEncoderLayer(d_model=self.trans_dim, nhead=nhead, batch_first=True, norm_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(layer, num_layers)

        self.norm = nn.LayerNorm(self.trans_dim)
        self.out = nn.Linear(self.trans_dim, self.class_num)
    
    def forward(self, X):
        patch_embeddings = self.pre_transformer(X)
        batch_size, n, _ = patch_embeddings.shape

        classification_tokens = repeat(self.classification_token, '1 n d -> b n d', b=batch_size)
        patches = torch.cat((classification_tokens, patch_embeddings), dim=1)
       
        patches += self.pos_embeddings[:,:n+1]
        trans_out = self.transformer(patches)
        
        # Take the output just from the token
        token = trans_out[:,0]
        
        return self.out(self.norm(token))