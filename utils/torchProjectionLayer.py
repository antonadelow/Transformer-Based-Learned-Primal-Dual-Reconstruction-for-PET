import torch
import parallelproj
from array_api_compat import device


class LinearSingleChannelOperator(torch.autograd.Function):

     @staticmethod
     def forward(
         ctx, x: torch.Tensor, operator: parallelproj.LinearOperator
     ) -> torch.Tensor:
         
         ctx.set_materialize_grads(False)
         ctx.operator = operator

         batch_size = x.shape[0]
         channels = x.shape[1]
         y = torch.zeros(
             (batch_size,) + (channels,) + operator.out_shape, dtype=x.dtype, device=device(x)
         )

         for i in range(batch_size):
             for j in range(channels):
                y[i,j, ...] = operator(x[i,j, ...].detach())

         return y

     @staticmethod
     def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
         if grad_output is None:
             return None, None
         else:
             operator = ctx.operator

             batch_size = grad_output.shape[0]
             channels = grad_output.shape[1]
             x = torch.zeros(
                 (batch_size,) + (channels,) + operator.in_shape,
                 dtype=grad_output.dtype,
                 device=device(grad_output),
             )

             for i in range(batch_size):
                 for j in range(channels):
                    x[i,j, ...] = operator.adjoint(grad_output[i,j, ...].detach())

             return x, None

class AdjointLinearSingleChannelOperator(torch.autograd.Function):
    
     @staticmethod
     def forward(
         ctx, x: torch.Tensor, operator: parallelproj.LinearOperator
     ) -> torch.Tensor:

         ctx.set_materialize_grads(False)
         ctx.operator = operator

         batch_size = x.shape[0]
         channels = x.shape[1]
         y = torch.zeros(
             (batch_size,) + (channels,) + operator.in_shape, dtype=x.dtype, device=device(x)
         )

         for i in range(batch_size):
            for j in range(channels):
                y[i,j,...] = operator.adjoint(x[i,j, ...].detach())

         return y

     @staticmethod
     def backward(ctx, grad_output):
         if grad_output is None:
             return None, None
         else:
             operator = ctx.operator

             batch_size = grad_output.shape[0]
             channels = grad_output.shape[1]

             x = torch.zeros(
                 (batch_size,) + (channels,) + operator.out_shape,
                 dtype=grad_output.dtype,
                 device=device(grad_output),
             )

             for i in range(batch_size):
                for j in range(channels):

                 x[i,j, ...] = operator(grad_output[i,j, ...].detach())
            
             return x, None