import numpy as np
import torch
import torch.nn as nn

class TaskLinearModel(nn.Module):

    def __init__(self, w, v, c=None):
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


class TaskLinearMetaModel(nn.Module):

    def __init__(self, V, c=None) -> None:
        super().__init__()
        self.V = V
        self.V_hat = V
        self.c = c
        self.c_hat = c

    # def forward(self, x, task_index):
    #     assert x.ndim > 1
    #     embedding = self.V(x)  # [b, d, r]
    #     w = self.W[task_index]  
    #     c = self.c(x) if self.c is not None else 0
    #     predictions = embedding@w + c
    #     return predictions
    
    # def task_forward(self, x, task):
    #     assert x.ndim > 1
    #     embedding = self.V(x)  # [b, d, r]
    #     W = self.W  # [T, r]
    #     c = self.c(x) if self.c is not None else 0
    #     predictions = embedding@W.T + c.unsqueeze(2).expand(-1, -1, self.T) # [b, d, T]
    #     return predictions

    
    def estimate_transform(self, W_star, indices=None):
        calibration_size, _ = W_star.shape
        # w_values = []
        # for t in range(calibration_size):
        #     w = self.task_models[t].w
        #     w_values.append(w.detach().numpy())
        W_regression = self.W[:calibration_size].detach().numpy()
        estimator, residuals, rank, s = np.linalg.lstsq(
        W_regression, W_star, rcond=None)
        return estimator

    def recalibrate(self, W_star):
    
        self.estimator = self.estimate_transform(W_star)
        
        # for model in self.define_task_models:
        #     model.w = self.estimator.T @ model.w
        self.W_hat = self.W.detach() @ torch.tensor(self.estimator, dtype=torch.float)

        transform = np.linalg.inv(self.estimator)
        tensor = torch.tensor(transform, dtype=torch.float, requires_grad=False)
        layer = nn.Linear(self.r, self.r, bias=False)
        layer.weight.data = tensor
        self.V_hat = nn.Sequential(self.V, layer)
        self.c_hat = nn.Sequential(self.c, layer)
        return self.V_hat, self.W_hat
    
    def adapt(self, x_values, y_values):
        v_values = self.V_hat(x_values)
        c_values = self.c(x_values).detach()
        
        X = v_values.view(-1, self.r).detach() 
        Y = (y_values - c_values).view(-1)
        
        w_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        model = TaskLinearModel(w_hat.squeeze(), self.V_hat, self.c_hat)
        return model


    def define_task_models(self, W):
        self.W = nn.Parameter(W)
        self.T, self.r = W.shape
        # self.W = W
        self.task_models = []
        for t in range(self.T):
            w = self.W[t]
            model = TaskLinearModel(w, self.V_hat, self.c_hat)
            self.task_models.append(model)
        return self.task_models


class MAML(nn.Module):

    def __init__(self, net) -> None:
        super().__init__()
        self.net = net
    
    def forward(self, x):
        return self.net(x)
    
    # def 
