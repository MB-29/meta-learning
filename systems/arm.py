from systems.system import System

import numpy as np
import torch
import matplotlib.pyplot as plt

from systems.system import ActuatedSystem
from robotics.arm import Arm

    
class ActuatedArm(ActuatedSystem):

    d, m, r = 8, 1, 3

    I2_values_train = np.array([1/3., 2.0/3])
    ml_values_train = np.array( [1., 2.])
    mll_values_train = np.array([1., 2.5])
    parameter_grid_train = np.meshgrid(I2_values_train, mll_values_train, ml_values_train)
    W_train = np.vstack(list(map(np.ravel, parameter_grid_train))).T

    training_task_n_trajectories = 1

    I2_values_test = np.array([1.5/3, 2.5/3.])
    ml_values_test = np.array([1.2, 1.7])
    mll_values_test = np.array([1.0, 2.0])
    parameter_grid_test = np.meshgrid(I2_values_test, mll_values_test, ml_values_test)
    W_test = np.vstack(list(map(np.ravel, parameter_grid_test))).T


    test_task_n_trajectories = 1

    dt = 0.02
    Nt = 200
    t_values = dt*np.arange(Nt)
    gamma = 5
    n_trajectories = 5
    U_values = np.zeros((n_trajectories, Nt, 1))
    x0_values = np.zeros((n_trajectories, 4))
    for traj_index in range(n_trajectories):
        period = 100*(traj_index//8+1)*dt
        phase = traj_index//4 * np.pi/2 - np.pi
        # magnitude = gamma/4 * (traj_index%4 +1) 
        # print(magnitude)
        U_values[traj_index] = gamma*(np.sin(2*np.pi*t_values/period + phase)).reshape(-1, 1)
        x0_values[traj_index] = np.random.randn(4)
        # x0_values[traj_index, 0] = traj_index * np.pi
        # x0_values[traj_index, 1] = 5*(2*traj_index//2-1)
        # x0_values[traj_index, 2] = (traj_index-1) * np.pi/2
        # U_values[2*traj_index+1] = -gamma*(np.sin(2*np.pi*t_values/(100(traj_index+1)*dt))).reshape(-1, 1)
    # U_values[1] = -gamma*(np.sin(2*np.pi*t_values/(75*dt))).reshape(-1, 1)
    # U_values[2] = gamma*(np.cos(2*np.pi*t_values/(50*dt))).reshape(-1, 1)
    # U_values[3] = -gamma*(np.cos(2*np.pi*t_values/(10*dt))).reshape(-1, 1)
    
    # x0_values[:, 2] = 1
    # x0_values[::2, 2] = np.pi
    # U_values[200:] = gamma*(np.sin(2*np.pi*t_values[200:]/(Nt/2*dt))).reshape(-1, 1)


    def __init__(self, alpha=.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.test_data = None

    def V_star(self, x):
        dd_y, cphi, sphi, d_phi, dd_phi  = torch.unbind(x, dim=1)
        v = torch.stack((dd_y, dd_phi*cphi - d_phi**2*sphi), dim=1)
        return v


    def define_environment(self, w):
        # print(f'w = {w}')
        I2, mll, ml = w
        l2 = 3*I2/ml
        l1 = mll/ml
        m2 = ml/l2
        arm = Arm(1., m2, l1, l2, self.alpha)
        return arm

    
    def extract_points(self, state_values):
    
        velocity_values = (1/self.dt)*np.diff(state_values, axis=0)
        phi1, d_phi1, phi2, d_phi2 = state_values[:-1].T
        d_phi1, dd_phi1, d_phi2, dd_phi2 = velocity_values.T
        cphi1, sphi1 = np.cos(phi1), np.sin(phi1)
        cphi2, sphi2 = np.cos(phi2), np.sin(phi2)
        x_values = np.stack((cphi1, sphi1, d_phi1, dd_phi1, cphi2, sphi2, d_phi2, dd_phi2), axis=1)
        points = torch.tensor(x_values).float()
        return points
    
