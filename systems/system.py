import numpy as np 
import torch
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
        return 0
    

    def generate_data(self, W, n_trajectories):
        raise NotImplementedError
    
    def generate_V_data(self):
        return self.V_star(self.grid)
    
    def generate_training_data(self):
        dataset = self.generate_data(self.W_train, self.training_task_n_trajectories)
        return dataset
    
    def generate_test_data(self):
        return self.generate_data(self.W_test, self.test_task_n_trajectories)
    
    def test_model(self, model):
        raise NotImplementedError
    
    def loss(self, meta_model, data):
        raise NotImplementedError

    def generate_data(self, W, n_trajectories):
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
    
    def extract_points(self, state_values):
        raise NotImplementedError

    def extract_data(self, state_values, u_values):
        points = self.extract_points(state_values)
        targets = torch.tensor(u_values).float()

        return (points, targets)
    
    def generate_task_dataset(self, environment):
        task_points = torch.zeros((self.Nt*self.n_trajectories, self.d))
        task_targets = torch.zeros((self.Nt*self.n_trajectories))
        for trajectory_index in range(self.n_trajectories): 
            U = self.U_values[trajectory_index]
            state_values = environment.actuate(U, x0=self.x0_values[trajectory_index])
            task_targets[trajectory_index*self.Nt:(trajectory_index+1)*self.Nt] = torch.tensor(U).float().squeeze()
            task_points[trajectory_index*self.Nt:(trajectory_index+1)*self.Nt] = self.extract_points(state_values)
        task_dataset = (task_points, task_targets) 
        return task_dataset
    

