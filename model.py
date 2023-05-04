import numpy as np
import torch
import torch.nn as nn

class TaskModel(nn.Module):

    def __init__(self, w, v, c):
        super().__init__()
        self.w = w
        self.V = v
        self.c = c if c is not None else lambda x:0

    def forward(self, x):
        assert x.ndim > 1
        V = self.V(x) 
        return V @ self.w + self.c(x)
    
    def d_flow(self, t, x):
        return self.forward(x)


class MetaModel(nn.Module):

    def __init__(self, W_values, V, c=None) -> None:
        super().__init__()
        self.W = nn.Parameter(W_values)
        self.T, self.r = W_values.shape
        self.V = V
        self.c = c

        # transform = torch.eye(self.r)

    # def embed(self, x):
    #     return self.V(x)@self.transform.T
    
    # def forward(self, x):
    #     assert x.ndim > 1
    #     embedding = self.V(x)  # [b, d, r]
    #     W = self.W  # [T, r]
    #     c = self.c(x) if self.c is not None else 0
    #     predictions = embedding@W.T + c.unsqueeze(2).expand(-1, -1, self.T) # [b, d, T]
    #     return predictions
    def forward(self, x, task_index):
        assert x.ndim > 1
        embedding = self.V(x)  # [b, d, r]
        w = self.W[task_index]  
        c = self.c(x) if self.c is not None else 0
        predictions = embedding@w + c
        return predictions
    
    def task_forward(self, x, task):
        assert x.ndim > 1
        embedding = self.V(x)  # [b, d, r]
        W = self.W  # [T, r]
        c = self.c(x) if self.c is not None else 0
        predictions = embedding@W.T + c.unsqueeze(2).expand(-1, -1, self.T) # [b, d, T]
        return predictions

    
    def estimate_transform(self, W_star, indices=None):
        sample_size, _ = W_star.shape
        estimator, residuals, rank, s = np.linalg.lstsq(
        self.W[:sample_size].detach(), W_star, rcond=None)
        return estimator

    def recalibrate(self, W_star):
    
        self.estimator = self.estimate_transform(W_star)
        
        self.W_hat = self.W.detach() @ self.estimator

        transform = np.linalg.inv(self.estimator)
        tensor = torch.tensor(transform, dtype=torch.float, requires_grad=False)
        layer = nn.Linear(self.r, self.r, bias=False)
        layer.weight.data = tensor
        self.V_hat = nn.Sequential(self.V, layer)
        return self.V_hat, self.W_hat
    
    def adapt(self, x_values, y_values):
        v_values = self.V_hat(x_values)
        c_values = self.c(x_values).detach()
        
        X = v_values.view(-1, self.r).detach() 
        Y = (y_values - c_values).view(-1)
        
        w_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return self.define_task_model(w_hat.squeeze())

    def define_task_model(self, w):
        task_model = TaskModel(torch.tensor(w, dtype=torch.float), self.V_hat, self.c)
        return task_model

class MAML(nn.Module):

    def __init__(self, net) -> None:
        super().__init__()
        self.net = net
    
    def forward(self, x):
        return self.net(x)
    
    # def 
