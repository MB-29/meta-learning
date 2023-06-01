import numpy as np
import torch
import torch.nn as nn
import learn2learn.algorithms as l2la

from models.metamodel import MetaModel

loss_function = nn.MSELoss()

class TaskLinearModel(nn.Module):

    def __init__(self, w, v, c=None):
        super().__init__()
        self.w = w
        self.V = v
        self.c = c 

    def forward(self, x):
        assert x.ndim > 1
        V = self.V(x) 
        y = V @ self.w
        y = y if self.c is None else y+self.c(x)
        return y
    
    def d_flow(self, t, x):
        return self.forward(x)


# class OneStepModel(nn.Module):

    # def __init__(self, net, targets, alpha):
    #     self.net = net
    #     self.targets = targets≤
    #     self.alpha = alpha

    # def one_step_forward(self, x):
    #     predictions = self.net(x)
    #     task_loss = nn.MSELoss()(predictions, self.targets)
    #     task_loss.backward(create_graph=True)
    #     for tensor in self.net.parameters():
    #         tensor.data += self.alpha*tensor.grad
    #     return self.net(self)

class TaskLinearMetaModel(MetaModel):

    def __init__(self, V, c=None, W=None) -> None:
        super().__init__()
        self.V = V
        self.V_hat = V
        self.c = c 
        self.c_hat = self.c

        # W = W if W is not None else 

    def estimate_transform(self, W_star, indices=None):
        calibration_size, _ = W_star.shape
        W_regression = self.W[:calibration_size].detach().numpy()
        estimator, residuals, rank, s = np.linalg.lstsq(
        W_regression, W_star, rcond=None)
        return estimator

    def calibrate(self, W_star):
    
        self.estimator = self.estimate_transform(W_star)
        self.W_hat = self.W.detach() @ torch.tensor(self.estimator, dtype=torch.float)

        transform = np.linalg.inv(self.estimator + 1e-7*np.eye(self.r))
        tensor = torch.tensor(transform, dtype=torch.float, requires_grad=False)
        layer = nn.Linear(self.r, self.r, bias=False)
        layer.weight.data = tensor
        self.V_hat = nn.Sequential(self.V, layer)
        self.c_hat = nn.Sequential(self.c, layer) if self.c is not None else None
        return self.V_hat, self.W_hat
    
    # def adapt_task_model(self, points, targets, n_steps=None):
    #     v_values = self.V_hat(points)
    #     c_values = self.c_hat(points).detach() if self.c is not None else torch.zeros_like(targets)
        
    #     X = v_values.view(-1, self.r).detach() 
    #     Y = (targets - c_values).view(-1)
        
    #     w_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    #     w = torch.tensor(w_hat, dtype=torch.float)
    #     model = TaskLinearModel(w, self.V_hat, self.c_hat)
    #     return model
    def adapt_task_model(self, points, targets, n_steps=None):
        v_values = self.V_hat(points)
        c_values = self.c_hat(points).detach() if self.c is not None else torch.zeros_like(targets)
        
        X = v_values.view(-1, self.r).detach() 
        Y = (targets - c_values).view(-1)
        
        w_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        w = torch.tensor(w_hat, dtype=torch.float)
        model = TaskLinearModel(w, self.V_hat, self.c_hat)
        return model
    # def adapt_task_model(self, points, targets, n_steps, lr=0.01):
    #     # learner = self.net    
    #     w = torch.randn(self.r, requires_grad=True)
    #     model = TaskLinearModel(w, self.V, self.c)
        
    #     # model = TaskCoDA(self.mnet, )
    #     # for parameter in learner.parameters():
    #     #     parameter.requires_grad = False
    #     # print(f'adapting {learner.module[-1].weight}')
    #     # print(f'adapt {n_steps} steps')
    #     # print(f'targets {targets[:10]}')
    #     adapter = torch.optim.Adam([w], lr=lr)
    #     for adaptation_step in range(n_steps):
    #         predictions = model(points)
    #         train_error = loss_function(predictions.squeeze(), targets.squeeze()) + 00.1*torch.norm(w)**2
    #         adapter.zero_grad()
    #         train_error.backward()
    #         # print(train_error)
    #         adapter.step()
    #         # print(train_error)
    #         # print(f'adapt, step{adaptation_step}')
    #     return model
    

    def define_task_models(self, W=None):
        W = W if W is not None else torch.randn(self.T, self.r)
        self.W = nn.Parameter(W)
        self.T, self.r = W.shape
        # self.W = W
        self.task_models = []
        for t in range(self.T):
            w = self.W[t]
            model = TaskLinearModel(w, self.V_hat, self.c_hat)
            self.task_models.append(model)
        return self.task_models
    
    def get_training_task_model(self, task_index, task_points=None, task_targets=None):
        return self.task_models[task_index]


