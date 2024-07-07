import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
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

class UnetD(nn.Module):
    
    def __init__(self,n):
        
        super(UnetD,self).__init__()
        
        self.layerb = nn.Sequential(
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
                        nn.ConvTranspose2d(256,128,3,2,(0,1),(0,1)),
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
                        nn.ConvTranspose2d(128,64,3,2,(0,0),(0,0)),
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
                        nn.ConvTranspose2d(64,32,3,2,(0,1),(0,1)),
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
                        nn.Conv2d(32,1,1,1,0),
                        #nn.ReLU(inplace=True)
        )
        
          
    def forward(self,x):
        
        l1 = self.layerb(x)
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

class CrossAttentionBlock(nn.Module):
  def __init__(self, in_channels, primal_shape, dual_shape, indices, head_dim=96, heads=2, dropout=0.1, p=7):
    super(CrossAttentionBlock, self).__init__()
    self.head_dim = head_dim
    self.heads = heads
    self.register_buffer("indices", indices)

    self.im_h_d, self.im_w_d, self.im_d_d = dual_shape
    self.im_h_p, self.im_d_p, self.im_w_p = primal_shape

    self.embedding_dim = heads*head_dim
  
    self.embedd_p = nn.Sequential(Rearrange('bd c (p1 h) (p2 w) -> bd (p1 p2) (c h w)',h=p,w=p),
                                  nn.Linear(in_channels*p*p, self.embedding_dim))
    
    self.embedd_d = nn.Sequential(Rearrange('bd c t v -> bd t (c v)'),
                                   nn.Linear(in_channels*self.indices.shape[1]*1, self.embedding_dim))

    self.softmax = nn.Softmax(dim=-1)
    self.Q_p = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), Rearrange('b n (h d) -> b h n d', h = self.heads))
    self.K_p = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), Rearrange('b n (h d) -> b h n d', h = self.heads))
    self.V_p = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), Rearrange('b n (h d) -> b h n d', h = self.heads))

    self.Q_d = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), Rearrange('b n (h d) -> b h n d', h = self.heads))
    self.K_d = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), Rearrange('b n (h d) -> b h n d', h = self.heads))
    self.V_d = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), Rearrange('b n (h d) -> b h n d', h = self.heads))
    self.rearrange = Rearrange('b h n d -> b n (h d)')

    self.mlp_p = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.GELU(),
                               nn.Linear(self.embedding_dim, self.embedding_dim), nn.GELU(),
                               nn.Linear(self.embedding_dim, self.embedding_dim), nn.GELU())
    
    self.mlp_d = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.GELU(),
                               nn.Linear(self.embedding_dim, self.embedding_dim), nn.GELU(),
                               nn.Linear(self.embedding_dim, self.embedding_dim), nn.GELU())

    self.dropout = nn.Dropout(dropout)

    self.positional_encodings_p = nn.Parameter(torch.randn(1, self.indices.shape[0], self.embedding_dim))
    self.positional_encodings_d = nn.Parameter(torch.randn(1, self.indices.shape[0], self.embedding_dim))

    self.unembedd_p = nn.Sequential(nn.Linear(self.embedding_dim,p*p),
                                     Rearrange('bd (p1 p2) (h w c) -> bd c (p1 h) (p2 w)', c=1, h=p, p1=self.im_h_p // p))
    
    self.unembedd_d = nn.Sequential(nn.Linear(self.embedding_dim,self.indices.shape[1]),
                                     Rearrange('bd t (c v) -> bd c t v',c=1))
    
    self.layernorm_att_p = nn.LayerNorm(self.embedding_dim)
    self.layernorm_mlp_p = nn.LayerNorm(self.embedding_dim)
    
    self.layernorm_att_d = nn.LayerNorm(self.embedding_dim)
    self.layernorm_mlp_d = nn.LayerNorm(self.embedding_dim)

  def forward(self,x,y):
    xemb = self.embedd_p(x) + self.positional_encodings_p 
    yemb = self.embedd_d(y[:,:,self.indices[:,:,0].long(),self.indices[:,:,1].long()]) + self.positional_encodings_d

    Q_p, K_p, V_p = self.Q_p(xemb), self.K_p(xemb), self.V_p(xemb)
    Q_d, K_d, V_d = self.Q_p(yemb), self.K_p(yemb), self.V_p(yemb)
    with torch.autocast(enabled=False, device_type="cuda"):
      attention_p = self.dropout(torch.matmul(self.softmax(torch.matmul(Q_d, K_p.transpose(-1, -2)) * self.head_dim ** -0.5),V_p))
      attention_d = self.dropout(torch.matmul(self.softmax(torch.matmul(Q_p, K_d.transpose(-1, -2)) * self.head_dim ** -0.5),V_d))

    xemb = self.layernorm_att_p(xemb + self.rearrange(attention_p))
    yemb = self.layernorm_att_d(yemb + self.rearrange(attention_d))
    
    xemb = self.layernorm_mlp_p(xemb + self.dropout(self.mlp_p(xemb)))
    yemb = self.layernorm_mlp_d(yemb + self.dropout(self.mlp_d(yemb)))

    x_full = self.unembedd_p(xemb)
    y_c = self.unembedd_d(yemb)
    y_out = y[:,-1,:,:].unsqueeze(1)
    y_full = torch.zeros_like(y_out)
    y_full[:,:,self.indices[:,:,0].long(),self.indices[:,:,1].long()] = y_c*self.indices[:,:,2].half()

    x = x[:,-1,:,:].unsqueeze(1) + x_full
    y = y_out + y_full

    return x, y

class LPD(nn.Module):
    def __init__(self, n_iter, pet_projector, normalization, indices, return_all=False):
      super(LPD, self).__init__()
      self.n_iter = n_iter
      self.dual_shape = pet_projector.out_shape
      self.primal_shape = pet_projector.in_shape

      self.primal_layers = nn.ModuleList([UnetP(i+1) for i in range(self.n_iter)])
      self.dual_layers = nn.ModuleList([UnetD(i+1 if i==0 else i+2) for i in range(self.n_iter)])
      self.cabs = nn.ModuleList([CrossAttentionBlock(1, self.primal_shape, self.dual_shape, indices) for i in range(self.n_iter-1)])
 
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
          f_ca, h_ca = self.cabs[i](self.to2DP(f),self.to2DD(h))
          f = f + self.to3DP(f_ca)
          h = h + self.to3DD(h_ca)

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