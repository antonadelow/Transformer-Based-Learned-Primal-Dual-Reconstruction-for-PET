import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange
from torch.nn.functional import pad
from utils.geometry import patch_sinogram
from utils.torchProjectionLayer import *

class UnetP(nn.Module):
    def __init__(self,n):
        super(UnetP,self).__init__()

        self.layerc = nn.Sequential(
                        nn.Conv2d(n,32,3,padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32,32,3,stride = 1,padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = nn.Sequential(
                        nn.Conv2d(32,64,3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64,64,3,stride = 1,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128,128,3,stride = 1,padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
        )

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4 = nn.Sequential(
                        nn.Conv2d(128,256,3,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256,256,3,stride = 1,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
        )

        self.layerUP31 = nn.Sequential(
                        nn.ConvTranspose2d(256,128,3,2,1,1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
        )

        self.layerUP32 = nn.Sequential(
                        nn.Conv2d(256,128,3,1,1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128,128,3,1,1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
        )

        self.layerUP41 = nn.Sequential(
                        nn.ConvTranspose2d(128,64,3,2,0,0),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
        )

        self.layerUP42 = nn.Sequential(
                        nn.Conv2d(128,64,3,1,1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64,64,3,1,1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
        )

        self.layerUP51 = nn.Sequential(
                        nn.ConvTranspose2d(64,32,3,2,0,0),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True)
        )

        self.layerUP52 = nn.Sequential(
                        nn.Conv2d(64,32,3,1,1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32,32,3,1,1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True)
        )

        self.layerUP61 = nn.Sequential(
                        nn.Conv2d(32,1,1,1,0)
        )


    def forward(self,x):
        l1 = self.layerc(x)
        l2 = self.layer2(self.pool1(l1))
        l3 = self.layer3(self.pool2(l2))
        mid = self.layer4(self.pool3(l3))

        l3up = self.layerUP31(mid)
        l3up = torch.cat((l3up,l3),dim=1)
        l3up = self.layerUP32(l3up)

        l4up = self.layerUP41(l3up)
        l4up = torch.cat((l4up,l2),dim=1)
        l4up = self.layerUP42(l4up)

        l5up = self.layerUP51(l4up)
        l5up = torch.cat((l5up,l1),dim=1)
        l5up = self.layerUP52(l5up)

        out = self.layerUP61(l5up)

        return out    

class Transformer(nn.Module):
  def __init__(self, in_channels, head_dim=16, heads=6, dropout=0.1):
    super(Transformer, self).__init__()
    self.head_dim = head_dim
    self.heads = heads

    self.embedding_dim = heads*head_dim
  
    self.softmax = nn.Softmax(dim=-1)
    self.Q = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), Rearrange('b n (h d) -> b h n d', h = self.heads))
    self.K = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), Rearrange('b n (h d) -> b h n d', h = self.heads))
    self.V = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), Rearrange('b n (h d) -> b h n d', h = self.heads))

    self.rearrange = Rearrange('b h n d -> b n (h d)')

    self.mlp = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.GELU(),
                             nn.Linear(self.embedding_dim, self.embedding_dim), nn.GELU(),
                             nn.Linear(self.embedding_dim, self.embedding_dim), nn.GELU())
    
    self.layernorm_att = nn.LayerNorm(self.embedding_dim)
    self.layernorm_mlp = nn.LayerNorm(self.embedding_dim)

    self.dropout = nn.Dropout(dropout)

  def forward(self,Z):
    with torch.autocast(enabled=False, device_type="cuda"):
      Q, K, V = self.Q(Z), self.K(Z), self.V(Z)
      attention = self.dropout(torch.matmul(self.softmax(torch.matmul(Q, K.transpose(-1, -2)) * self.head_dim ** -0.5),V))
    x = self.layernorm_att(Z + self.rearrange(attention))
    x = self.layernorm_mlp(x + self.dropout(self.mlp(x)))
    return x

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_shape, indices, heads=2, inner_dim=156):
        super(TransformerBlock, self).__init__()
        self.register_buffer("indices", indices)

        self.heads = heads
        self.im_h, self.im_w, self.im_d = img_shape

        self.embedding_dim = heads*inner_dim
        self.in_channels = in_channels

        self.embed = nn.Sequential(Rearrange('bd c t v -> bd t (c v)'),
                                   nn.Linear(in_channels*self.indices.shape[1]*1, self.embedding_dim))

        self.positional_encodings_inner = nn.Parameter(torch.randn(1, self.indices.shape[0], self.embedding_dim))
        
        self.T1 = Transformer(in_channels, head_dim=inner_dim*1, heads=heads//1)

        self.unembed = nn.Sequential(nn.Linear(self.embedding_dim,self.indices.shape[1]),
                                     Rearrange('bd t (c v) -> bd c t v',c=1))

    def forward(self, sino):
      x = sino[:,:,self.indices[:,:,0].long(),self.indices[:,:,1].long()]
      Z = self.embed(x)
      Z = Z + self.positional_encodings_inner
      Z = self.T1(Z)   
      out = self.unembed(Z)

      sino_out = sino[:,-1,:,:].unsqueeze(1)
      out_full = torch.zeros_like(sino_out)
      out_full[:,:,self.indices[:,:,0].long(),self.indices[:,:,1].long()] = out.float()*self.indices[:,:,2]
    
      return sino_out + out_full

class LPD(nn.Module):
    def __init__(self, n_iter, pet_projector, normalization, indices, return_all=False):
      super(LPD, self).__init__()
      self.n_iter = n_iter
      self.dual_shape = pet_projector.out_shape
      self.indices = indices

      self.primal_layers = nn.ModuleList([UnetP(i+1) for i in range(self.n_iter)])
      self.dual_layers = nn.ModuleList([TransformerBlock(i+1 if i==0 else i+2,1, self.dual_shape, indices) for i in range(self.n_iter)])
 
      self.fwd_op_layer = LinearSingleChannelOperator.apply
      self.adjoint_op_layer = AdjointLinearSingleChannelOperator.apply
      self.normalization = normalization
      self.proj = pet_projector

      self.to2DP = Rearrange('b c h s w -> (b s) c h w')
      self.to3DP = Rearrange('(b s) c h w -> b c h s w', s=self.dual_shape[-1])
      
      self.to2DD = Rearrange('b c h w s -> (b s) c h w')
      self.to3DD = Rearrange('(b s) c h w -> b c h w s', s=self.dual_shape[-1])

      self.return_all = return_all

    def forward(self, sino):
        sino = sino.cuda()
        h = self.to3DD(self.dual_layers[0](self.to2DD(sino)))
        f = self.to3DP(self.primal_layers[0](self.to2DP(self.adjoint_op_layer(h, self.proj)/self.normalization)))
        h_cats= [sino, h]
        f_cats = [f]
        for i in range(self.n_iter-1):
          h_cats.append(self.fwd_op_layer(f, self.proj))
          h = h + self.to3DD(self.dual_layers[i+1](self.to2DD(torch.cat(h_cats,dim=1))))
          
          f_cats.append(self.adjoint_op_layer(h, self.proj)/self.normalization)
          f = f + self.to3DP(self.primal_layers[i+1](self.to2DP(torch.cat(f_cats,dim=1))))

          h_cats.pop(), f_cats.pop()
          h_cats.append(h), f_cats.append(f)

        if self.return_all is True:
           return f_cats
        else:
          return f.squeeze(1)