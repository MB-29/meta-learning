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
        # self.x = np.clip(self.x, self.x_min, self.x_max)
        return dx

    def reset(self, x0=None):
        x0 = self.x0 if x0 is None else x0
        self.x = x0.copy()

    
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

    def actuate(self, u_values, x0=None, plot=False, noise_size=0):
        # print(x0)
        T = u_values.shape[0]
        self.reset(x0)
        x_values = np.zeros((T+1, self.d))
        for t in range(T):
            x = self.x.copy()
            noise = noise_size*np.random.randn(*x.shape)
            x_values[t] = x + noise
            # x_values[t] = x 
            # u = np.random.choice([-1.0, 1.0], size=1) 
            u = u_values[t]
            self.step(u)

            if not plot:
                continue
            self.plot_system(x, u, t)
            plt.pause(0.1) ; plt.close()
        x_values[T] = self.x.copy()
        # z_values = np.concatenate((x_values, u_values), axis=1)
        return x_values
    
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
    
        

    def control_loop(self, u_ff_values, x_target_values, parameter_changes=None, x0=None, plot=None):
        self.reset(x0)
        T = u_ff_values.shape[0]
        state_values = np.zeros((T+1, self.d))
        u_values = np.zeros_like(u_ff_values)

        parameter_updates = T * [None]
        if parameter_changes is not None:
            for t, value in parameter_changes.items():
                parameter_updates[t] = value

        # target_tip_positions = self.compute_tip_positions(x_target_values)
        for t in range(T):


            x = self.x.copy()
            state_values[t] = x
            u_ff = u_ff_values[t]
            x_target = x_target_values[t]

            obs = self.compute_tip_positions(x.reshape(1, -1))
            obs_target = self.compute_tip_positions(x_target.reshape(1, -1))
            e_p = (obs_target - obs).squeeze()
            u_p = np.array([0.001, 0.0035]) @ e_p
            u_p = self.K @ e_p

            e_d = x_target[1::2] - x[1::2] 
            u_d = self.K @ e_d
            
            # target_tip = target_tip_positions[t, :] 
            # tip  = self.compute_tip_positions(x.reshape(1, -1))
            # residual = target_tip[0] - tip[0, 0]
            # u_p = 10*np.array([residual])

            u = u_ff 
            u = u_ff + u_p + u_d
            # u = u_ff + u_p 

            self.step(u)

            u_values[t] = u

            dynamic_parameters = parameter_updates[t]
            if dynamic_parameters is not None:
                self.set_dynamic_parameters(dynamic_parameters)

            if plot is None:
                continue
            # x_target_values = plot['x_target_values']
            # u_target_values = plot['u_target_values']
            # self.plot_system(x_target, u_target_values[t], t, alpha=0.2)
            self.plot_system(x, u, t)
            # plt.scatter(*target_tip_positions[t], color="red")
            plt.pause(0.1)
            plt.close() 
        state_values[-1] = self.x.copy()
        # z_values = np.concatenate((state_values[:-1], u_values), axis=1)
        return state_values, u_values
    
