
import numpy as np
import torch 

from systems.system import StaticSystem

class StaticCartpole(StaticSystem):

    d, m, r = 5, 1, 2

    M_values_train = np.array([1., 2.0])
    ml_values_train = np.array([.2, .5])
    # mll_values_train = np.array([1.5, 2.5])
    # mgl_values_train = np.array([10., 11.])
    parameter_grid_train = np.meshgrid(M_values_train, ml_values_train)
    W_train = np.dstack(parameter_grid_train).reshape(-1, 2)

    training_task_n_trajectories = 1

    M_values_test = np.array([1.5, 2.5])
    ml_values_test = np.array([.3, .6])
    # mll_values_test = np.array([1.0, 2.0])
    # mgl_values_test = np.array([9., 12.])
    parameter_grid_test = np.meshgrid(M_values_test, ml_values_test)
    W_test = np.dstack(parameter_grid_test).reshape(-1, 2)

    test_task_n_trajectories = 1

    n_points = 5
    dd_z_values = torch.linspace(-1, 1, n_points)
    phi_values = torch.linspace(-np.pi, np.pi, n_points)
    d_phi_values = torch.linspace(0, 1., n_points)
    dd_phi_values = torch.linspace(0, 1, n_points)
    grid_dd_z, grid_phi, grid_d_phi, grid_dd_phi, = torch.meshgrid(
            dd_z_values,
            phi_values,
            d_phi_values,
            dd_phi_values,
        )
    grid = torch.cat([
        grid_dd_z.reshape(-1, 1),
        torch.cos(grid_phi).reshape(-1, 1),
        torch.sin(grid_phi).reshape(-1, 1),
        grid_d_phi.reshape(-1, 1),
        grid_dd_phi.reshape(-1, 1),
    ], 1)


    def __init__(self) -> None:
        super().__init__(self.W_train, self.d)

        self.test_data = None
        
    
    def V_star(self, x):
        dd_y, cphi, sphi, d_phi, dd_phi  = torch.unbind(x, dim=1)
        v = torch.stack((dd_y, dd_phi*cphi - d_phi**2*sphi), dim=1)
        return v

    def c_star(self, x):
        return 0
        
    def predict(self, model):
        return model(self.grid)