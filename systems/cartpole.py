from systems.system import System

import numpy as np
import torch
import matplotlib.pyplot as plt

from systems.system import ActuatedSystem
from robotics.cartpole import Cartpole

    
class ActuatedCartpole(ActuatedSystem):

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
    parameter_grid_test = np.meshgrid(M_values_test, ml_values_test
    
    )
    W_test = np.dstack(parameter_grid_test).reshape(-1, 2)

    test_task_n_trajectories = 1

    dt = 0.02
    Nt = 200
    t_values = dt*np.arange(Nt)
    gamma = 5
    n_trajectories = 8
    U_values = np.zeros((n_trajectories, Nt, 1))
    for traj_index in range(n_trajectories):
        period = 100*(traj_index//2+1)*dt
        phase = traj_index//4 * np.pi/2 - np.pi
        U_values[traj_index] = gamma*(np.sin(2*np.pi*t_values/period + phase)).reshape(-1, 1)
        # U_values[2*traj_index+1] = -gamma*(np.sin(2*np.pi*t_values/(100(traj_index+1)*dt))).reshape(-1, 1)
    # U_values[1] = -gamma*(np.sin(2*np.pi*t_values/(75*dt))).reshape(-1, 1)
    # U_values[2] = gamma*(np.cos(2*np.pi*t_values/(50*dt))).reshape(-1, 1)
    # U_values[3] = -gamma*(np.cos(2*np.pi*t_values/(10*dt))).reshape(-1, 1)
    
    x0_values = np.zeros((n_trajectories, 4))
    x0_values[::2, 2] = np.pi
    # U_values[200:] = gamma*(np.sin(2*np.pi*t_values[200:]/(Nt/2*dt))).reshape(-1, 1)


    def __init__(self, beta=0) -> None:
        super().__init__(self.W_train, self.d)
        self.beta = beta
        self.test_data = None

    def V_star(self, x):
        dd_y, cphi, sphi, d_phi, dd_phi  = torch.unbind(x, dim=1)
        v = torch.stack((dd_y, dd_phi*cphi - d_phi**2*sphi), dim=1)
        return v


    def define_environment(self, w):
        M, ml = w
        cartpole = Cartpole(ml, M-ml, 1., beta=self.beta, alpha = 0., dt=self.dt)
        return cartpole

    
    def extract_points(self, state_values):
    
        velocity_values = (1/self.dt)*np.diff(state_values, axis=0)
        y, d_y, phi, d_phi = state_values[:-1].T
        d_y, dd_y, d_phi, dd_phi = velocity_values.T
        cphi, sphi = np.cos(phi), np.sin(phi)
        x_values = np.stack((dd_y, cphi, sphi, d_phi, dd_phi), axis=1)
        points = torch.tensor(x_values).float()
        return points
    

class DampedActuatedCartpole(ActuatedCartpole):

    d, m, r = 6, 1, 3

    def __init__(self) -> None:
        super().__init__(beta=0.1)

    def extract_points(self, state_values):
    
        velocity_values = (1/self.dt)*np.diff(state_values, axis=0)
        y, d_y, phi, d_phi = state_values[:-1].T
        d_y, dd_y, d_phi, dd_phi = velocity_values.T
        cphi, sphi = np.cos(phi), np.sin(phi)
        x_values = np.stack((d_y, dd_y, cphi, sphi, d_phi, dd_phi), axis=1)
        points = torch.tensor(x_values).float()
        return points
    