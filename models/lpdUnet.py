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

class LPD(nn.Module):
    def __init__(self, n_iter, pet_projector, normalization, return_all=False):
      super(LPD, self).__init__()
      self.n_iter = n_iter
      self.dual_shape = pet_projector.out_shape

      self.primal_layers = nn.ModuleList([UnetP(i+1) for i in range(self.n_iter)])
      self.dual_layers = nn.ModuleList([UnetD(i+1 if i==0 else i+2) for i in range(self.n_iter)])
 
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