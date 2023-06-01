import numpy as np
import torch
import torch.nn as nn
import learn2learn.algorithms as l2la

from models.metamodel import MetaModel
    

loss_function = nn.MSELoss()

class MAML(MetaModel):

    def __init__(self, meta_dataset, net, lr, first_order=False, n_adaptation_steps=1) -> None:
        super().__init__(meta_dataset)
        self.lr = lr
        self.n_adaptation_steps = n_adaptation_steps
        self.learner = l2la.MAML(net, lr, first_order=False)
        
    
    def get_training_task_model(self, task_index, task_points, task_targets, n_adaptation_steps=None):
        return self.fit_step(task_points, task_targets, self.n_adaptation_steps)
    
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
