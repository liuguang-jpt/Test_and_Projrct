import torch
import torch.nn as nn

def batch_norm(X,gamma,beta,moving_mean,moving_var,eps,momentum):
    if not torch.is_grad_enabled():
        X_hat=(X-moving_mean)/torch.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape)==2:
            mean=X.mean(dim=0)
            var=((X-mean)**2).mean(dim=0)
        else:
            mean=X.mean(dim=(0,2,3),keepdim=True)
            var=((X-mean)**2).mean(dim=(0,2,3),keepdim=True)
        X_hat=(X-mean)/torch.sqrt(var+eps)
        moving_mean=moving_mean*momentum+mean+(1-momentum)*mean
        moving_var=moving_var*momentum+var+(1-momentum)*var
    Y=gamma*X_hat+beta
    return Y,moving_mean.data,moving_var.data

class Batch_Norm(nn.Module):
    #num_feature 全连接层的输出数量或卷积层的输出通道数
    #num_dims 2表示全连接层，4表示卷积层
    def __init__(self,num_features,num_dims):
        super().__init__()
        if num_dims==2:
            shape=(1,num_features)
        else:
            shape=(1,num_features,1,1)
        self.gamma=nn.Parameter(torch.ones(shape))
        self.beta=nn.Parameter(torch.zeros(shape))

        self.moving_mean=nn.Parameter(torch.zeros(shape))
        self.moving_var=nn.Parameter(torch.ones(shape))

    def forward(self,X):
        if self.moving_mean.device!=X.device:
            self.moving_mean=self.moving_mean.to(X.device)
            self.moving_var=self.moving_var.to(X.device)
        Y,self.moving_mean,self.moving_var=batch_norm(X,self.gamma,self.beta,self.moving_mean,self.moving_var,eps=1e-5,momentum=0.9)
        return Y

nn.BatchNorm2d(9)