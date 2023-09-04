import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from systems.system import ActuatedSystem
from robotics.cartpole import Cartpole
from controller import actuate
    
class ActuatedCartpole(ActuatedSystem):

    d, m, r = 5, 1, 2

    M_values_train = np.array([1., 2.0])
    m_values_train = np.array([.2, .5])
    # mll_values_train = np.array([1.5, 2.5])
    # mgl_values_train = np.array([10., 11.])
    parameter_grid_train = np.meshgrid(M_values_train, m_values_train)
    W_train = np.dstack(parameter_grid_train).reshape(-1, 2)

    training_task_n_trajectories = 1

    M_values_test = np.array([1.5, 2.5])
    m_values_test = np.array([.3, .6])
    # mll_values_test = np.array([1.0, 2.0])
    # mgl_values_test = np.array([9., 12.])
    parameter_grid_test = np.meshgrid(M_values_test, m_values_test
    
    )
    W_test = np.dstack(parameter_grid_test).reshape(-1, 2)

    test_task_n_trajectories = 1

    
    Nt = 200
    n_trajectories = 8
    x0_values = np.zeros((n_trajectories, 4))
    x0_values[::2, 2] = np.pi

    # self.U_values[200:] = gamma*(np.sin(2*np.pi*t_values[200:]/(Nt/2*dt))).reshape(-1, 1)


    def __init__(self, dt = 0.02, beta=0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dt = dt
        self.beta = beta
        self.test_data = None


        t_values = dt*np.arange(self.Nt)
        gamma = 5
        self.U_values = np.zeros((self.n_trajectories, self.Nt, 1))
        for traj_index in range(self.n_trajectories):
            period = 100*(traj_index//2+1)*dt
            phase = traj_index//4 * np.pi/2 - np.pi
            self.U_values[traj_index] = gamma*(np.sin(2*np.pi*t_values/period + phase)).reshape(-1, 1)
            # self.U_values[2*traj_index+1] = -gamma*(np.sin(2*np.pi*t_values/(100(traj_index+1)*dt))).reshape(-1, 1)
        # self.U_values[1] = -gamma*(np.sin(2*np.pi*t_values/(75*dt))).reshape(-1, 1)
        # self.U_values[2] = gamma*(np.cos(2*np.pi*t_values/(50*dt))).reshape(-1, 1)
        # self.U_values[3] = -gamma*(np.cos(2*np.pi*t_values/(10*dt))).reshape(-1, 1)
        

    def V_star(self, x):
        dd_y, cphi, sphi, d_phi, dd_phi  = torch.unbind(x, dim=1)
        v = torch.stack((dd_y, dd_phi*cphi - d_phi**2*sphi), dim=1)
        return v

    def define_environment(self, w):
        M, m = w
        cartpole = Cartpole(m, M-m, 1., beta=self.beta, alpha = 0., dt=self.dt)
        return cartpole

    
    def extract_points(self, state_values):
        if state_values.shape[0] <= 1:
            return torch.zeros(1, self.d)
        velocity_values = (1/self.dt)*np.diff(state_values, axis=0)
        y, d_y, phi, d_phi = state_values[:-1].T
        d_y, dd_y, d_phi, dd_phi = velocity_values.T
        cphi, sphi = np.cos(phi), np.sin(phi)
        x_values = np.stack((dd_y, cphi, sphi, d_phi, dd_phi), axis=1)
        points = torch.tensor(x_values).float()
        return points
    

class DampedActuatedCartpole(ActuatedCartpole):

    d, m, r = 6, 1, 3

    def __init__(self, dt = 0.02, **kwargs) -> None:
        super().__init__(dt=dt, beta=0.1, **kwargs)

    def extract_points(self, state_values):
        if state_values.shape[0] <= 1:
            return torch.zeros(1, self.d)
        velocity_values = (1/self.dt)*np.diff(state_values, axis=0)
        y, d_y, phi, d_phi = state_values[:-1].T
        d_y, dd_y, d_phi, dd_phi = velocity_values.T
        cphi, sphi = np.cos(phi), np.sin(phi)
        x_values = np.stack((d_y, dd_y, cphi, sphi, d_phi, dd_phi), axis=1)
        points = torch.tensor(x_values).float()
        return points

class Upkie(DampedActuatedCartpole):

    def __init__(self, dt=1/200, **kwargs) -> None:
        super().__init__(dt, **kwargs)

    def generate_training_data(self):
        with open(f'data/upkie/train/dataset.pkl', 'rb') as file:
            dataset = pickle.load(file)
        self.T = len(dataset)
        return dataset
    
    def generate_test_data(self):
        with open(f'data/upkie/test/dataset.pkl', 'rb') as file:
            dataset = pickle.load(file)
        self.T_test = len(dataset)
        return dataset
    
    def generate_cartpole_test_data(self):
        # with open('output/u.pkl', 'rb') as file:
        with open('output.pkl', 'rb') as file:
            u_values = pickle.load(file).reshape(-1, 1)
        total_mass, mass = 1.5, .5
        w = np.array([total_mass, mass])
        cartpole = self.define_environment(w)
        x0 = np.array([0, 0, np.pi, 0])
        state_values = actuate(cartpole, u_values, x0=x0, plot=False)
        task_targets = torch.tensor(u_values).float().squeeze()
        task_points = self.extract_points(state_values)

        return [(task_points, task_targets)]


    
# class WheelUpkie(DampedActuatedCartpole):

#     def __init__(self, dt=1/200, **kwargs) -> None:
#         super().__init__(dt, **kwargs)

#     def generate_training_data(self):
#         with open(f'data/velocity/train/dataset.pkl', 'rb') as file:
#             dataset = pickle.load(file)
#         self.T = len(dataset)
#         return dataset
    
#     def generate_test_data(self):
#         with open(f'data/velocity/test/dataset.pkl', 'rb') as file:
#             dataset = pickle.load(file)
#         self.T_test = len(dataset)
#         return dataset
    