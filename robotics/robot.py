import numpy as np
import matplotlib.pyplot as plt

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

    def actuate(self, u_values, x0=None):
        T = u_values.shape[0]
        self.reset()
        self.x0 = x0 if x0 is not None else self.x0
        x_values = np.zeros((T+1, self.d))
        for t in range(T):
            x = self.x.copy()
            x_values[t] = x
            # u = np.random.choice([-1.0, 1.0], size=1)
            u = u_values[t]
            self.step(u)

            # self.plot_system(x, u, t)
            # plt.pause(0.1)
            # plt.close()
        x_values[T] = self.x.copy()
        # z_values = np.concatenate((x_values, u_values), axis=1)
        return x_values
    
    def plan_inverse_dynamics(self, model, x_target_values):
        T = x_target_values.shape[0]-1
        u_ff_values = np.zeros((T, 1))
        for t in range(T):
            x = x_target_values[t]
            x_ = x_target_values[t+1]
            x_dot = (x_ - x)/self.dt
            u_ff = model(x, x_dot)
            u_ff_values[t] = u_ff
        return u_ff_values
    
    def control_loop(self, x_target_values, u_ff_values, plot=None):
        self.reset()
        state_values = np.zeros_like(x_target_values)
        u_values = np.zeros_like(u_ff_values)
        T = x_target_values.shape[0]-1
        target_tip_positions = self.compute_tip_positions(x_target_values)
        for t in range(T):
            x = self.x.copy()
            state_values[t] = x
            u_ff = u_ff_values[t]
            x_target = x_target_values[t]

            obs = self.observe(x)
            obs_target = self.observe(x_target)
            e = obs - obs_target
            u_b = self.K @ e
            
            target_tip = target_tip_positions[t, :] 
            tip  = self.compute_tip_positions(x.reshape(1, -1))
            residual = target_tip[0] - tip[0, 0]
            u_b = 10*np.array([residual])

            u = u_ff + u_b
            u = u_ff 
            self.step(u)

            u_values[t] = u

            if plot is None:
                continue
            u_target_values = plot['u_target_values']
            self.plot_system(x_target, u_target_values[t], t, alpha=0.2)
            # self.plot_system(x, u, t)
            plt.scatter(*target_tip_positions[t], color="red")
            plt.pause(0.1)
            plt.close()
        state_values[-1] = self.x.copy()
        # z_values = np.concatenate((state_values[:-1], u_values), axis=1)
        return state_values, u_values