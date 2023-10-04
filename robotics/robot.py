import numpy as np
import matplotlib.pyplot as plt

class Robot:

    def __init__(self, x0, d, m, dt, sigma=0):
        """
        :param x0: Initial state
        :type x0: array of size d
        :param d: state space dimension
        :type d: int
        :param m: action space dimension
        :type m: int
        :param dt: time step
        :type dt: float
        :param gamma: input amplitude
        :type gamma: float
        """
        self.d = d
        self.m = m
        self.dt = dt
        self.x0 = x0.copy()

        self.x = x0.copy()
        
        self.sigma = sigma
    
    def forward_dynamics(self, x, u):
        """Flow of the dynamics.

        :param x: state
        :type x: array of size d
        :param u: action
        :type u: array of size m
        :param t: time
        :type t: float, optional
        :raises NotImplementedError: _description_
        """
        raise NotImplementedError

    def step(self, u):
        """Compute the state increment as a function of the action and time.
        """
        x_dot = self.forward_dynamics(self.x, u)
        dx = x_dot * self.dt
        # noise =  self.sigma * np.random.randn(self.d)
        # dx += noise
        # print(dx)
        self.x += dx
        self.x = np.clip(self.x, self.x_min, self.x_max)
        return dx

    def reset(self, x0=None):
        x0 = self.x0 if x0 is None else x0
        self.x = x0.copy()

    def observe_state(self):
        noise = self.sigma*np.random.randn(*self.x.shape)
        return self.x.copy() + noise
    
    def d_dynamics(self, z):
        raise NotImplementedError
    def inverse_dynamics(self, x, x_dot):
        raise NotImplementedError
    
    def plot_system(self, x, u):
        raise NotImplementedError
    
    def set_dynamic_parameters(self, parameters):
        raise NotImplementedError
    def get_dynamic_parameters(self):
        raise NotImplementedError

    
    def plan_inverse_dynamics(self, x_target_values):
        T = x_target_values.shape[0]-1
        u_ff_values = np.zeros((T, 1))
        for t in range(T):
            x = x_target_values[t]
            x_ = x_target_values[t+1]
            x_dot = (x_ - x)/self.dt
            u_ff = self.inverse_dynamics(x, x_dot)
            u_ff_values[t] = u_ff.sum()
        return u_ff_values
    
        

