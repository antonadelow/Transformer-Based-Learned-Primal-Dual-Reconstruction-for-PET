import torch
import torch.nn as nn
from torchProjectionLayer import *
from models.restormer import *

class LPD(nn.Module):
    def __init__(self, n_iter, pet_projector, normalization, return_all=False):
      super(LPD, self).__init__()
      self.n_iter = n_iter
      self.dual_shape = pet_projector.out_shape
      self.primal_shape = pet_projector.in_shape

      self.primal_layers = nn.ModuleList([RestormerPrimal(i+1, self.primal_shape) for i in range(self.n_iter)])
      self.dual_layers = nn.ModuleList([RestormerDual(i+1 if i==0 else i+2, self.dual_shape) for i in range(self.n_iter)])
 
      self.fwd_op_layer = LinearSingleChannelOperator.apply
      self.adjoint_op_layer = AdjointLinearSingleChannelOperator.apply
      self.normalization = normalization
      self.proj = pet_projector
      
      self.return_all = return_all

    def forward(self, sino):
        h = self.dual_layers[0](sino)

        f = self.primal_layers[0](self.adjoint_op_layer(h, self.proj)/self.normalization)

        h_cats= [sino, h]
        f_cats = [f]
        for i in range(self.n_iter-1):
          h_cats.append(self.fwd_op_layer(f, self.proj))
          h = h + self.dual_layers[i+1](torch.cat(h_cats,dim=1))
          
          f_cats.append(self.adjoint_op_layer(h, self.proj)/self.normalization)
          f = f + self.primal_layers[i+1](torch.cat(f_cats,dim=1))

          h_cats.pop(), f_cats.pop()
          h_cats.append(h), f_cats.append(f)

        if self.return_all is True:
           return f_cats
        else:
          return f.squeeze(1)