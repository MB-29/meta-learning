import numpy as np
import torch
import torch.nn as nn

import learn2learn.algorithms as l2la

loss_function = nn.MSELoss()

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


class OneStepModel(nn.Module):

    def __init__(self, net, targets, alpha):
        self.net = net
        self.targets = targets
        self.alpha = alpha

    def one_step_forward(self, x):
        predictions = self.net(x)
        task_loss = nn.MSELoss()(predictions, self.targets)
        task_loss.backward(create_graph=True)
        for tensor in self.net.parameters():
            tensor.data += self.alpha*tensor.grad
        return self.net(self)

class TaskLinearMetaModel(nn.Module):

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

    def recalibrate(self, W_star):
    
        self.estimator = self.estimate_transform(W_star)
        self.W_hat = self.W.detach() @ torch.tensor(self.estimator, dtype=torch.float)

        transform = np.linalg.inv(self.estimator)
        tensor = torch.tensor(transform, dtype=torch.float, requires_grad=False)
        layer = nn.Linear(self.r, self.r, bias=False)
        layer.weight.data = tensor
        self.V_hat = nn.Sequential(self.V, layer)
        self.c_hat = nn.Sequential(self.c, layer) if self.c is not None else None
        return self.V_hat, self.W_hat
    
    def adapt_task_model(self, points, targets, n_steps=None):
        v_values = self.V_hat(points)
        c_values = self.c(points).detach() if self.c is not None else torch.zeros_like(targets)
        
        X = v_values.view(-1, self.r).detach() 
        Y = (targets - c_values).view(-1)
        
        w_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        w = torch.tensor(w_hat, dtype=torch.float)
        model = TaskLinearModel(w, self.V_hat, self.c_hat)
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
    
    def get_training_task_model(self, task_index, task_points=None, task_targets=None):
        return self.task_models[task_index]
    

class MAML(nn.Module):

    def __init__(self, net, lr, first_order=False, n_adaptation_steps=1) -> None:
        super().__init__()
        self.lr = lr
        self.n_adaptation_steps = n_adaptation_steps
        self.learner = l2la.MAML(net, lr, first_order=False)
        
    
    def get_training_task_model(self, task_index, task_points, task_targets, n_adaptation_steps=None):
        n_steps = self.n_adaptation_steps if n_adaptation_steps is not None else 1
        return self.adapt_task_model(task_points, task_targets, n_steps)
    
    def adapt_task_model(self, points, targets, n_steps):
        learner = self.learner.clone()
        # print(f'adapt {n_steps} steps')
        # print(f'targets {targets[:10]}')
        for adaptation_step in range(n_steps):
            train_error = loss_function(learner(points).squeeze(), targets.squeeze())
            # print(f'adapt, step{adaptation_step}')
            learner.adapt(train_error)
        return learner
    # def define_task_models(self, training_sources, training_targets):
    #     self.T = len(training_sources)
    #     # self.W = W
    #     self.task_models = []
    #     for t in range(self.T):
    #         task_targets = training_targets[t]
    #         self.task_models = []
    #         model = OneStepModel(self.net, task_targets)
    #         self.task_models.append(model)
    #     return self.task_models