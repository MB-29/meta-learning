import numpy as np
import torch
import matplotlib.pyplot as plt

from systems.system import StaticSystem

def derive_field(potential_map):

    def field_map(x):
        # x[:, 1] = torch.abs(x[:, 1])
        x_ = x.clone().detach().requires_grad_(True)
        potential = potential_map(x_)
        grad_outputs = torch.ones_like(potential)
        grad = torch.autograd.grad(
            potential,
            x_,
            grad_outputs
        )[0]
        # grad_ = grad.clone()
        # grad_[:, 1] = -grad[:, 1]
        # x2 = x_[:, 1].unsqueeze(1).expand(-1, 2)
        # symetrized_grad = torch.where(x2 < 0, grad_, grad)
        return grad
    return field_map


class Charges(StaticSystem):

    d, m, r = 2, 1, 2

    q1_values_train = np.arange(1, 6)
    q2_values_train = np.arange(1, 6)
    # q1_values_train = np.arange(-5, 5)
    # q2_values_train = np.arange(-5, 5)
    parameter_grid_train = np.meshgrid(
        q1_values_train,
        q2_values_train,
        )
    W_train = np.vstack(list(map(np.ravel, parameter_grid_train))).T

    training_task_n_trajectories = 1
    
    W_test = np.array([
        [1.5, 4.5,],
        [2.5, .5],
    ])
    # W_test = np.random.randn(10, 4)

    n_points = 20
    x1_values = torch.linspace(-1, 1, n_points)
    x2_values = torch.linspace(0, 1, n_points)
    x2_values_full  = torch.linspace(-1, 1, n_points)
    grid_x1, grid_x2 = torch.meshgrid(
        x1_values, x2_values, indexing='ij')
    grid = torch.cat([
        grid_x1.reshape(-1, 1),
        grid_x2.reshape(-1, 1),
    ], 1)
    grid_x1_, grid_x2_ = torch.meshgrid(
        x1_values, x2_values_full, indexing='ij')
    grid_full = torch.cat([
        grid_x1_.reshape(-1, 1),
        grid_x2_.reshape(-1, 1),
    ], 1)
    # norm2 = (full_grid**2).sum(dim=1).unsqueeze(1).expand(-1, 2)

    # grid = torch.where(norm2 > 3e-2, full_grid, 0)

    # xA = torch.tensor([1., 0.])
    # xB = torch.tensor([0., 1.])
    # xC = torch.tensor([-1., 0.])
    # xD = torch.tensor([0., -1.])

    xA = torch.tensor([-1.1, 0.,])
    xB = torch.tensor([1.1, 0.])

    # xA = torch.tensor([-0.1, -0.5,])
    # xB = torch.tensor([0.1, -0.5])


#
    def __init__(self, sigma=0) -> None:
        super().__init__(sigma=sigma)

    
    def V_star(self, x):
        rA = torch.linalg.norm(x - self.xA, dim=1) 
        rB = torch.linalg.norm(x - self.xB, dim=1) 
        v = torch.stack((1/rA, 1/rB), dim=1)
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
        field = field_map(self.grid)
        vector_x = field[:, 0].reshape(
            self.n_points, self.n_points).detach().numpy()
        vector_y = field[:, 1].reshape(
            self.n_points, self.n_points).detach().numpy()
        magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
        # strength = np.where(magnitude < 1e1, magnitude, 1e1)
        strength = magnitude
        linewidth = strength / strength.max()
        plt.streamplot(
            self.grid_x1.numpy().T,
            self.grid_x2.numpy().T,
            vector_x.T,
            vector_y.T,
            # color='black',
            # linewidth=linewidth*5,
            arrowsize=1.,
            density=.5,
            **kwargs)
