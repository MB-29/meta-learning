from systems.system import System

import numpy as np
import torch
import matplotlib.pyplot as plt

from systems.system import StaticSystem, ActuatedSystem
from robotics.cartpole import Cartpole



class StaticCartpole(StaticSystem):

    d, m, r = 5, 1, 2

    M_values_train = np.array([1., 2.0])
    ml_values_train = np.array([.2, .5])
    # mll_values_train = np.array([1.5, 2.5])
    # mgl_values_train = np.array([10., 11.])
    parameter_grid_train = np.meshgrid(M_values_train, ml_values_train)
    W_train = np.dstack(parameter_grid_train).reshape(-1, 2)

    training_task_n_samples = 1

    M_values_test = np.array([1.5, 2.5])
    ml_values_test = np.array([.3, .6])
    # mll_values_test = np.array([1.0, 2.0])
    # mgl_values_test = np.array([9., 12.])
    parameter_grid_test = np.meshgrid(M_values_test, ml_values_test)
    W_test = np.dstack(parameter_grid_test).reshape(-1, 2)

    test_task_n_samples = 1

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
    
class ActuatedCartpole(ActuatedSystem):

    d, m, r = 5, 1, 2

    M_values_train = np.array([1., 2.0])
    ml_values_train = np.array([.2, .5])
    # mll_values_train = np.array([1.5, 2.5])
    # mgl_values_train = np.array([10., 11.])
    parameter_grid_train = np.meshgrid(M_values_train, ml_values_train)
    W_train = np.dstack(parameter_grid_train).reshape(-1, 2)

    training_task_n_samples = 1

    M_values_test = np.array([1.5, 2.5])
    ml_values_test = np.array([.3, .6])
    # mll_values_test = np.array([1.0, 2.0])
    # mgl_values_test = np.array([9., 12.])
    parameter_grid_test = np.meshgrid(M_values_test, ml_values_test
    
    )
    W_test = np.dstack(parameter_grid_test).reshape(-1, 2)

    test_task_n_samples = 1

    dt = 0.02
    Nt = 200
    t_values = dt*np.arange(Nt)
    gamma = 10
    u_values = gamma*np.sign(np.sin(2*np.pi*t_values/(Nt/4*dt))).reshape(-1, 1)


    def __init__(self) -> None:
        super().__init__(self.W_train, self.d)

        self.test_data = None

    def V_star(self, x):
        dd_y, cphi, sphi, d_phi, dd_phi  = torch.unbind(x, dim=1)
        v = torch.stack((dd_y, dd_phi*cphi - d_phi**2*sphi), dim=1)
        return v


    def define_environment(self, w):
        M, ml = w
        cartpole = Cartpole(ml, M-ml, 1., alpha=0., beta=0., dt=self.dt)
        return cartpole
    
    def extract_points(self, state_values):
    
        velocity_values = (1/self.dt)*np.diff(state_values, axis=0)
        y, d_y, phi, d_phi = state_values[:-1].T
        d_y, dd_y, d_phi, dd_phi = velocity_values.T
        cphi, sphi = np.cos(phi), np.sin(phi)
        x_values = np.stack((dd_y, cphi, sphi, d_phi, dd_phi), axis=1)
        points = torch.tensor(x_values).float()
        return points