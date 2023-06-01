import torch.nn as nn

class MetaModel(nn.Module):

    def __init__(self, meta_dataset):
        super().__init__()
        self.meta_dataset = meta_dataset

    def get_training_task_model(self, task_index, task_points, task_targets):
        raise NotImplementedError
    
    def adapt_task_model(self, data, **kwargs):
        raise NotImplementedError

    def parametrizer(self, dataset):
        raise NotImplementedError
    
    def training_parametrizer(self, task_index):
        task_dataset = self.meta_dataset[task_index]
        return self.parametrizer(task_dataset)

