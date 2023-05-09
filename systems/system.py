
import torch.nn as nn

class System(nn.Module):

    def __init__(self, W_train, d) -> None:
        super().__init__()
        self.W_train = W_train
        self.T, self.r_star = W_train.shape
        self.d = d

    def V_star(self, x):
        raise NotImplementedError

    def c_star(self, x):
        raise NotImplementedError
    
    def define_environment(self, w):
        def environment(x):
            return self.V_star(x)@w + self.c_star(x)
        return environment

    def generate_data(self, W, n_samples):
        raise NotImplementedError
    
    def generate_V_data(self):
        raise NotImplementedError
    
    def generate_training_data(self):
        return self.generate_data(self.W_train, self.training_task_n_samples)
    
    def generate_test_data(self):
        return self.generate_data(self.W_test, self.test_task_n_samples)
    
    def test_model(self, model):
        raise NotImplementedError
    
    def loss(self, meta_model, data):
        raise NotImplementedError

    def plot_system(self, w):
        raise NotImplementedError

