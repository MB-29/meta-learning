import torch.nn as nn

class MetaModel(nn.Module):

    def __init__(self):
        super().__init__()

    def get_training_task_model(self, task_index, task_points, task_targets):
        raise NotImplementedError
    
    def adapt_task_model(self, points, targets, n_steps):
        raise NotImplementedError


