import numpy as np
import torch
import torch.nn as nn
import learn2learn.algorithms as l2la
import matplotlib.pyplot as plt


from models.metamodel import MetaModel
    

loss_function = nn.MSELoss()

class MAML(MetaModel):

    def __init__(self, T_train, net, lr, n_inner=1) -> None:
        super().__init__(T_train)
        self.lr = lr
        self.n_inner = n_inner
        self.inner_learner = l2la.MAML(net, lr, first_order=False)
        
    
    def parametrizer(self, task_index, dataset):
        task_dataset = dataset[task_index]
        points, targets = task_dataset
        learner = self.gradient_steps(points, targets, self.n_inner)
        return learner
    
    def gradient_steps(self, points, targets, n_inner):
        inner_learner = self.inner_learner.clone()
        # print(f'adapt {n_steps} steps')
        # print(f'targets {targets[:10]}')
        train_error_values = []
        for adaptation_step in range(n_inner):
            train_error = loss_function(inner_learner(points).squeeze(), targets.squeeze())
            # print(f'adapt, step{adaptation_step}')
            inner_learner.adapt(train_error)
            train_error_values.append(train_error.item())
        # plt.plot
        # if plot:
        #     plt.plot(train_error_values)
        #     plt.show()
        return inner_learner
    
    def adapt_task_model(self, data, n_steps):
        points, targets = data
        learner = self.gradient_steps(points, targets, n_steps)
        return learner

class BodyHead(nn.Module):
    def __init__(self, body, head, **kwargs) -> None:
        super().__init__()
        self.body = body
        self.head = head
    
    def forward(self, x):
        feature = self.body(x)
        return self.head(feature)

class ANIL(MAML):
    def __init__(self, T_train, body, head, lr, **kwargs) -> None:
        super().__init__(T_train, head, lr, **kwargs)
        self.body = body
        self.r = self.body[-1].weight.shape[0]

    def parametrizer(self, task_index, dataset):
        # print(f'body {self.body[0].weight}')
        task_dataset = dataset[task_index]
        points, targets = task_dataset
        features = self.body(points)
        # print(f'head {self.inner_learner.module.weight}')
        head =  self.gradient_steps(features, targets, self.n_inner)
        # print(f'adapted head {head.module.weight}')
        return BodyHead(self.body, head)

    def adapt_task_model(self, data, n_steps):
        points, targets = data
        features = self.body(points)
        head = self.gradient_steps(features, targets, n_steps)
        return BodyHead(self.body, head)
    
    def adapt_heads(self, dataset):
        T = len(dataset)
        W = torch.zeros(T, self.r)
        for task_index in range(T):
            data = dataset[task_index]
            task_model = self.adapt_task_model(data, n_steps=10)
            W[task_index] = task_model.head.module.weight
        self.W = W


