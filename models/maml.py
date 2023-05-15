import numpy as np
import torch
import torch.nn as nn
import learn2learn.algorithms as l2la

from models.metamodel import MetaModel
    

loss_function = nn.MSELoss()

class MAML(MetaModel):

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
