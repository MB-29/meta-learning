import numpy as np
import torch
import matplotlib.pyplot as plt

from systems.system import System

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


class Dipole(System):

    d, m, r = 2, 1, 2

    U_values = np.array([1., 2., 3.])
    p_values = np.array([0.1, 0.5, 1.0])
    W_train = np.dstack(np.meshgrid(U_values, p_values)).reshape(-1, 2)

    training_task_n_samples = 1

    n_points = 20
    x1_values = torch.linspace(-1, 1, n_points)
    x2_values = torch.linspace(0.1, 1, n_points)
    grid_x1, grid_x2 = torch.meshgrid(
        x1_values, x2_values, indexing='ij')
    grid = torch.cat([
        grid_x1.reshape(-1, 1),
        grid_x2.reshape(-1, 1),
    ], 1)

    mse_loss = torch.nn.MSELoss()

    def __init__(self) -> None:
        super().__init__(self.W_train, self.d)

        self.test_data = None
        
    
    def V_star(self, x):
        x1, x2 = torch.unbind(x, dim=1)
        r2 = x1**2 + x2**2
        v = torch.stack((x1, x1/r2), dim=1)
        return v

    def c_star(self, x):
        return 0
        
    def generate_data(self, W, n_samples):
        data = np.zeros((self.T, self.n_points**2))
        for task_index in range(self.T):
            w = W[task_index]
            environment = self.define_environment(w)
            potential = environment(self.grid)
            data[task_index] = potential
        return data
    
    def generate_test_data(self):
        return self.V_star(self.grid)
    
    def loss(self, meta_model, data):
        V, W = meta_model.V, meta_model.W
        # for t in 
        T, batch_size = data.shape
        assert W.shape[0] == T
        loss = 0
        for t in range(T):
            w = W[t]
            task_data = torch.tensor(data[t], dtype=torch.float)
            # task_model = meta_model.define_task_model(w)
            predictions = meta_model(self.grid, t)
            task_loss = self.mse_loss(predictions, task_data)
            loss += task_loss
        return loss
        # meta_predictions = meta_model(self.grid)
        # predictions = meta_model(self.grid)
        # loss = self.loss_function(data, predictions)

    def plot_potential(self, potential_map):
        potential = potential_map(self.grid)
        plt.pcolormesh(self.grid_x1,
                       self.grid_x2,
                       potential.reshape((self.n_points, self.n_points)),
                       cmap='gray'
                       )
        
    def plot_field(self, potential_map):
        field_map = derive_field(potential_map)
        field = field_map(self.grid)
        vector_x = field[:, 0].reshape(
            self.n_points, self.n_points).detach().numpy()
        vector_y = field[:, 1].reshape(
            self.n_points, self.n_points).detach().numpy()
        magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
        linewidth = magnitude / magnitude.max()
        plt.streamplot(
            self.grid_x1.numpy().T,
            self.grid_x2.numpy().T,
            vector_x.T,
            vector_y.T,
            color='black',
            linewidth=linewidth*2,
            arrowsize=.8,
            density=.8)
