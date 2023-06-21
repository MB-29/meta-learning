import torch
import torch.nn as nn

class MetaModel(nn.Module):

    def __init__(self, T_train):
        super().__init__()
        self.T_train = T_train

    # def get_training_task_model(self, task_index, task_points, task_targets):
    #     raise NotImplementedError
    
    def adapt_task_model(self, data, **kwargs):
        raise NotImplementedError

    def parametrizer(self, task_index, dataset):
        raise NotImplementedError
    
    def regularization(self):
        return 0
    
    # def training_parametrizer(self, task_index):
    #     task_dataset = self.meta_dataset[task_index]
    #     return self.parametrizer(task_dataset)

    # def save(self, path):
    #     # path = f'output/models/dipole/{metamodel_name}_ngrad-{n_gradient}.dat'
    #     with open(path, 'wb') as file:
    #         torch.save(self, file)
    # def load(self, path):
    #     # path = f'output/models/dipole/{metamodel_name}_ngrad-{n_gradient}.dat'
    #     with open(path, 'rb') as file:
    #         self = torch.load(file)