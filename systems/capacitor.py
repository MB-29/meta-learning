from os import listdir
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

from systems.system import StaticSystem

def derive_field(potential_map):

    def field_map(x):
        x_ = x.clone().detach().requires_grad_(True)
        potential = potential_map(x_)
        grad_outputs = torch.ones_like(potential)
        grad = torch.autograd.grad(
            potential,
            x_,
            grad_outputs
        )[0]
        return grad
    return field_map


class Capacitor(StaticSystem):

    d, m, r = 2, 1, 4


    x1_values = torch.linspace(-4, 4, 300)
    x2_values = torch.linspace(-2, 2, 200)
    grid_x1, grid_x2 = torch.meshgrid(
        x1_values, x2_values, indexing='xy')
    grid = torch.cat([
        grid_x1.reshape(-1, 1),
        grid_x2.reshape(-1, 1),
    ], 1)


    def __init__(self, data_path, sigma=0) -> None:
        super().__init__(sigma=sigma)
        self.data_path = data_path
        
    
    def predict(self, model):
        return model(self.grid)
    

    def plot_potential(self, potential_map):
        potential = potential_map(self.grid).detach().numpy()
        plt.pcolormesh(self.grid_x1,
                       self.grid_x2,
                       potential.reshape((200, 300)),
                       cmap='gray'
                       )
    def plot_potential_values(self, potential_values):
        plt.pcolormesh(self.grid_x1,
                       self.grid_x2,
                       potential_values.reshape((200, 300)),
                       cmap='gray'
                       )
        
    def plot_field(self, potential_map, **kwargs):
        field_map = derive_field(potential_map)
        field = field_map(self.grid)
        vector_x = field[:, 0].reshape(
            200, 300).detach().numpy()
        vector_y = field[:, 1].reshape(
            200, 300).detach().numpy()
        magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
        linewidth = magnitude / magnitude.max()
        plt.streamplot(
            np.linspace(-4, 4, 300),
            np.linspace(-2, 2, 200),
            # self.x2_values,
            vector_x,
            vector_y,
            # color='black',
            # linewidth=linewidth*5,
            arrowsize=.8,
            # density=.9,
            **kwargs)


    def generate_data(self, mode):
        data_path = f'{self.data_path}/{mode}'
        task_datasets = []
        context_values = []
        for file_name  in listdir(data_path):
            if file_name.split('.')[1] != 'pkl':
                continue
            file_path = f'{data_path}/{file_name}'
            print(file_path)
            with open(file_path, 'rb') as file:
                environment_data = pickle.load(file)
                solution = environment_data['solution']
                context = environment_data['context']
            points = self.grid
            targets = torch.tensor(solution).view(-1).float()
            noisy_targets = targets + self.sigma * torch.randn_like(targets)
            dataset = (points, noisy_targets)
            context_values.append(context)
            task_datasets.append(dataset)
            # print(f'context {context}')
        W =  np.array(context_values)

        return task_datasets, W
    

    def generate_training_data(self):
        meta_dataset, W = self.generate_data('train')
        self.W_train = W
        self.T, self.r_star = self.W_train.shape
        return meta_dataset
    
    def generate_test_data(self):
        dataset, W = self.generate_data('test')
        self.W_test = W
        self.T_test = self.W_test.shape[0]
        return dataset