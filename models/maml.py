import numpy as np
import torch
import torch.nn as nn
import learn2learn.algorithms as l2la

from models.metamodel import MetaModel
    

loss_function = nn.MSELoss()

class MAML(MetaModel):

    def __init__(self, T_train, net, lr, n_adaptation_steps=1) -> None:
        super().__init__(T_train)
        self.lr = lr
        self.n_adaptation_steps = n_adaptation_steps
        self.learner = l2la.MAML(net, lr, first_order=False)
        
    
    def parametrizer(self, task_index, dataset):
        task_dataset = dataset[task_index]
        points, targets = task_dataset
        return self.gradient_step(points, targets, self.n_adaptation_steps)
    
    def gradient_step(self, points, targets, n_steps):
        learner = self.learner.clone()
        # print(f'adapt {n_steps} steps')
        # print(f'targets {targets[:10]}')
        for adaptation_step in range(n_steps):
            train_error = loss_function(learner(points).squeeze(), targets.squeeze())
            # print(f'adapt, step{adaptation_step}')
            learner.adapt(train_error)
        return learner
    
    def adapt_task_model(self, data, n_steps):
        points, targets = data
        return self.gradient_step(points, targets, n_steps)


class ANIL(MAML):
    def __init__(self, T_train, net, lr, n_adaptation_steps=1) -> None:
        super().__init__(T_train, net, lr, n_adaptation_steps)