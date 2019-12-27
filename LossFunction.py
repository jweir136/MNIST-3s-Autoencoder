import torch
import torch.nn as nn
import torch.nn.functional as fn

def loss_function(pred_x, x, mu, logvar, epoch):
  mse = fn.mse_loss(pred_x, x)
  kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  
  if epoch >= 25:
    return mse + kld

  return mse + kld
