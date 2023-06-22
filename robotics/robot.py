import numpy as np

class Robot:

    def __init__(self, x0, d, m, dt):
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
        self.x += dx
        self.x = np.clip(self.x, self.x_min, self.x_max)
        return dx

    def reset(self):
        self.x = self.x0.copy()

    
    def d_dynamics(self, x, u):
        raise NotImplementedError
    
    def plot_system(self, x, u):
        raise NotImplementedError
