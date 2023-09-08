import pickle
import numpy as np
import torch

from systems.system import ActuatedSystem
from controller import actuate

class Upkie(ActuatedSystem):

    d, m, r = 6, 1, 1

    def __init__(self, dt=1/200, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dt = dt

    def generate_training_data(self):
        with open(f'data/upkie/train/dataset.pkl', 'rb') as file:
            data = pickle.load(file)
        dataset = data['dataset']
        contexts = data['contexts']
        self.T = len(dataset)
        self.W_train = contexts
        return dataset
    
    def generate_test_data(self):
        with open(f'data/upkie/test/dataset.pkl', 'rb') as file:
            data = pickle.load(file)
        dataset = data['dataset']
        contexts = data['contexts']
        self.T_test = len(dataset)
        self.W_test = contexts
        return dataset
    
    # def generate_cartpole_test_data(self):
    #     # with open('output/u.pkl', 'rb') as file:
    #     with open('output.pkl', 'rb') as file:
    #         u_values = pickle.load(file).reshape(-1, 1)
    #     total_mass, mass = 1.5, .5
    #     w = np.array([total_mass, mass])
    #     cartpole = self.define_environment(w)
    #     x0 = np.array([0, 0, np.pi, 0])
    #     state_values = actuate(cartpole, u_values, x0=x0, plot=False)
    #     task_targets = torch.tensor(u_values).float().squeeze()
    #     task_points = self.extract_points(state_values)

    #     return [(task_points, task_targets)]
    
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