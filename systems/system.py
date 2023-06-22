import torch
import torch.nn as nn

from control import actuate

class System(nn.Module):

    def __init__(self, W_train, d) -> None:
        super().__init__()
        self.W_train = W_train
        self.T, self.r_star = W_train.shape
        self.d = d

    def V_star(self, x):
        raise NotImplementedError

    def c_star(self, x):
        return 0
    

    def generate_data(self, W, n_samples):
        raise NotImplementedError
    
    def generate_V_data(self):
        return self.V_star(self.grid)
    
    def generate_training_data(self):
        return self.generate_data(self.W_train, self.training_task_n_samples)
    
    def generate_test_data(self):
        return self.generate_data(self.W_test, self.test_task_n_samples)
    
    def test_model(self, model):
        raise NotImplementedError
    
    def loss(self, meta_model, data):
        raise NotImplementedError

    def generate_data(self, W, n_samples):
        T, r = W.shape
        dataset = []
        for task_index in range(T):
            w = W[task_index]
            environment = self.define_environment(w)
            task_dataset = self.generate_task_dataset(environment)
            dataset.append(task_dataset)
        return dataset
    
    def plot_system(self, w):
        raise NotImplementedError

class StaticSystem(System):

    def define_environment(self, w):
        def environment(x):
            return self.V_star(x)@w + self.c_star(x)
        return environment

    def generate_task_dataset(self, environment):
        task_targets = environment(self.grid).float()
        task_dataset = (self.grid, task_targets)
        return task_dataset
    
class ActuatedSystem(System):

    def define_environment(self, w):
        raise NotImplementedError
    
    def generate_task_dataset(self, environment):
        state_values = actuate(environment, self.u_values)
        task_targets = torch.tensor(self.u_values).float()
        task_points = self.extract_points(state_values)
        task_dataset = (task_points, task_targets) 
        return task_dataset
    

