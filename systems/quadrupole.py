import numpy as np
import torch
import matplotlib.pyplot as plt

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


class Quadrupole(StaticSystem):

    d, m, r = 3, 1, 4

    q1_values_train = np.array([1., 2.])
    q2_values_train = np.array([1., 2.])
    q3_values_train = np.array([1., 2.])
    q4_values_train = np.array([1., 2.])
    parameter_grid_train = np.meshgrid(
        q1_values_train,
        q2_values_train,
        q3_values_train,
        q4_values_train,
        )
    W_train = np.vstack(list(map(np.ravel, parameter_grid_train))).T

    training_task_n_trajectories = 1
    
    # q1_values_test = np.array([1., 2.])
    # q2_values_test = np.array([1., 2.])
    # q3_values_test = np.array([1., 2.])
    # q4_values_test = np.array([1., 2.])
    # parameter_grid_test = np.meshgrid(
    #     q1_values_test,
    #     q2_values_test,
    #     q3_values_test,
    #     q4_values_test,
    #     )
    # W_test = np.vstack(list(map(np.ravel, parameter_grid_test))).T
    # test_task_n_trajectories = 1
    W_test = np.array([
        [1.2, 0.9, 1.3, -1.5],
        # [0.6, 1.2, 1., 0.9],
        [1., -1., 1., -1.],
    ])


    n_points = 20
    x1_values = torch.linspace(-0.5, 0.5, n_points)
    x2_values = torch.linspace(-0.5, 0.5, n_points)
    x3_values = torch.linspace(-0.5, 0.5, n_points)
    grid_x1, grid_x2, grid_x3 = torch.meshgrid(
        x1_values, x2_values, x3_values, indexing='ij')
    plane_x1, plane_x2, = torch.meshgrid(
        x1_values, x2_values, indexing='ij')
    grid = torch.cat([
        grid_x1.reshape(-1, 1),
        grid_x2.reshape(-1, 1),
        grid_x3.reshape(-1, 1),
    ], 1)
    plane = torch.cat([
        plane_x1.reshape(-1, 1),
        plane_x2.reshape(-1, 1),
        torch.zeros_like(plane_x1).reshape(-1, 1),
        # torch.zeros_like(grid_x3).reshape(-1, 1),
    ], 1)

    # xA = torch.tensor([1., 0.])
    # xB = torch.tensor([0., 1.])
    # xC = torch.tensor([-1., 0.])
    # xD = torch.tensor([0., -1.])

    xA = torch.tensor([1., 0., 0.])
    xB = torch.tensor([0., 1., 0.])
    xC = torch.tensor([-1., 0., 0.])
    xD = torch.tensor([0., -1., 0.])


    def __init__(self, sigma=0) -> None:
        super().__init__(sigma=sigma)

    
    def V_star(self, x):
        rA = torch.linalg.norm(x - self.xA, dim=1)
        rB = torch.linalg.norm(x - self.xB, dim=1)
        rC = torch.linalg.norm(x - self.xC, dim=1)
        rD = torch.linalg.norm(x - self.xD, dim=1)
        v = torch.stack((1/rA, 1/rB, 1/rC, 1/rD), dim=1)
        return v

    def c_star(self, x):
        return 0
        
    # def generate_data(self, W, n_trajectories):
    #     T, r = W.shape
    #     data = np.zeros((T, self.n_points**2))
    #     for task_index in range(T):
    #         w = W[task_index]
    #         environment = self.define_environment(w)
    #         potential_values = environment(self.grid)
    #         data[task_index] = potential_values
    #     return data
    
    def predict(self, model):
        return model(self.grid)
    
    # def loss(self, meta_model, data):
    #     V = meta_model.V
    #     # for t in 
    #     T, batch_size = data.shape
    #     loss = 0
    #     for t in range(T):
    #         task_data = torch.tensor(data[t], dtype=torch.float)
    #         # task_model = meta_model.define_task_model(w)
    #         model = meta_model.task_models[t]
    #         predictions = model(self.grid)
    #         task_loss = self.mse_loss(predictions, task_data)
    #         loss += task_loss
    #     return loss
        # meta_predictions = meta_model(self.grid)
        # predictions = meta_model(self.grid)
        # loss = self.loss_function(data, predictions)

    def plot_potential(self, potential_map):
        potential = potential_map(self.grid).detach().numpy()
        plt.pcolormesh(self.grid_x1,
                       self.grid_x2,
                       potential.reshape((self.n_points, self.n_points)),
                       cmap='gray'
                       )
        
    def plot_field(self, potential_map, **kwargs):
        field_map = derive_field(potential_map)
        field = field_map(self.plane)
        vector_x = field[:, 0].reshape(
            self.n_points, self.n_points).detach().numpy()
        vector_y = field[:, 1].reshape(
            self.n_points, self.n_points).detach().numpy()
        magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
        linewidth = magnitude / magnitude.max()
        plt.streamplot(
            self.plane_x1.numpy().T,
            self.plane_x2.numpy().T,
            vector_x.T,
            vector_y.T,
            # color='black',
            linewidth=linewidth*5,
            arrowsize=.8,
            density=.5,
            **kwargs)
