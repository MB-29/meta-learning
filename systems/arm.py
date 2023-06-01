import numpy as np
import torch
import matplotlib.pyplot as plt

from systems.system import System



class Arm(System):

    d, m, r = 6, 1, 3

    I2_values_train = np.array([1., 2.0])
    mll_values_train = np.array([1.5, 2.5])
    mgl_values_train = np.array([10., 11.])
    parameter_grid_train = np.meshgrid(I2_values_train, mll_values_train, mgl_values_train)
    W_train = np.dstack(parameter_grid_train).reshape(-1, 3)

    training_task_n_samples = 1

    I2_values_test = np.array([1.5, 2.5])
    mll_values_test = np.array([1.0, 2.0])
    mgl_values_test = np.array([9., 12.])
    parameter_grid_test = np.meshgrid(I2_values_test, mll_values_test, mgl_values_test)
    W_test = np.dstack(parameter_grid_train).reshape(-1, 3)

    test_task_n_samples = 1

    n_points = 5
    phi_values = torch.linspace(-np.pi/5, np.pi/5, n_points)
    d_phi_values = torch.linspace(0, 1., n_points)
    dd_phi_values = torch.linspace(0, 1, n_points)
    grid_phi1, grid_phi2, grid_d_phi1, grid_d_phi2, grid_dd_phi1, grid_dd_phi2 = torch.meshgrid(
            phi_values,
            phi_values,
            d_phi_values,
            d_phi_values,
            dd_phi_values,
            dd_phi_values,
        )
    grid = torch.cat([
        torch.cos(grid_phi1).reshape(-1, 1),
        torch.sin(grid_phi1).reshape(-1, 1),
        grid_d_phi1.reshape(-1, 1),
        grid_dd_phi1.reshape(-1, 1),
        torch.cos(grid_phi2).reshape(-1, 1),
        torch.sin(grid_phi2).reshape(-1, 1),
        grid_d_phi2.reshape(-1, 1),
        grid_dd_phi2.reshape(-1, 1),
    ], 1)


    def __init__(self) -> None:
        super().__init__(self.W_train, self.d)

        self.test_data = None
        
    
    def V_star(self, x):
        cphi1, sphi1, d_phi1, dd_phi1, cphi2, sphi2, d_phi2, dd_phi2 = torch.unbind(x, dim=1)
        dd_phi = dd_phi1 + dd_phi2
        s12 = cphi1*sphi2 + cphi2*sphi1
        v = torch.stack((dd_phi, cphi2*dd_phi1+sphi2*d_phi1**2, s12), dim=1)
        return v

    def c_star(self, x):
        return 0
        
    def generate_data(self, W, n_samples):
        T, r = W.shape
        data = np.zeros((T, self.grid.shape[0]))
        for task_index in range(T):
            w = W[task_index]
            environment = self.define_environment(w)
            task_values = environment(self.grid)
            data[task_index] = task_values
        return data
    
    def generate_V_data(self):
        return self.V_star(self.grid)
    
    def predict(self, model):
        return model(self.grid)
    
