import numpy as np
import matplotlib.pyplot as plt

class Controller:

    def __init__(self, metamodel, state_target_values, data, data_extractor, shots) -> None:
        self.metamodel = metamodel
        self.data = data
        self.state_target_values = state_target_values
        self.data_extractor = data_extractor
        self.shots = shots
        self.target_points = data_extractor(state_target_values)


    def adapt(self, state_values, u_values):
        new_data = self.data_extractor(state_values, u_values)
        adaptation_data = new_data if state_values.shape[0] >= self.shots else self.data
        # adaptation_data = self.data
        self.adapted_model = self.metamodel.adapt_task_model(adaptation_data, n_steps=100)
    
    def control(self, state_values, t):
        predictions = self.adapted_model(self.target_points)
        u_ff_values = predictions.detach().numpy().reshape(-1, 1)
        return u_ff_values[t]


def actuate(robot, u_values, x0=None, plot=False, noise_size=0):
    # print(x0)
    T = u_values.shape[0]
    robot.reset(x0)
    x_values = np.zeros((T+1, robot.d))
    for t in range(T):
        x = robot.x.copy()
        noise = noise_size*np.random.randn(*x.shape)
        x_values[t] = x + noise
        # x_values[t] = x 
        # u = np.random.choice([-1.0, 1.0], size=1) 
        u = u_values[t]
        robot.step(u)

        if not plot:
            continue
        robot.plot_system(x, u, t)
        plt.pause(0.1) ; plt.close()
    x_values[T] = robot.x.copy()
    # z_values = np.concatenate((x_values, u_values), axis=1)
    return x_values

def control_loop(robot, u_ff_values, x_target_values, parameter_changes=None, x0=None, plot=None):
    robot.reset(x0)
    T = u_ff_values.shape[0]
    state_values = np.zeros((T+1, robot.d))
    u_values = np.zeros_like(u_ff_values)

    parameter_updates = T * [None]
    if parameter_changes is not None:
        for t, value in parameter_changes.items():
            parameter_updates[t] = value

    # target_tip_positions = robot.compute_tip_positions(x_target_values)
    for t in range(T):


        x = robot.x.copy()
        state_values[t] = x
        u_ff = u_ff_values[t]
        x_target = x_target_values[t]

        obs = robot.compute_tip_positions(x.reshape(1, -1))
        obs_target = robot.compute_tip_positions(x_target.reshape(1, -1))
        e_p = (obs_target - obs).squeeze()
        u_p = np.array([0.001, 0.0035]) @ e_p
        u_p = robot.K @ e_p

        e_d = x_target[1::2] - x[1::2] 
        u_d = robot.K @ e_d
        
        # target_tip = target_tip_positions[t, :] 
        # tip  = robot.compute_tip_positions(x.reshape(1, -1))
        # residual = target_tip[0] - tip[0, 0]
        # u_p = 10*np.array([residual])

        u = u_ff 
        u = u_ff + u_p + u_d
        # u = u_ff + u_p 

        robot.step(u)

        u_values[t] = u

        dynamic_parameters = parameter_updates[t]
        if dynamic_parameters is not None:
            robot.set_dynamic_parameters(dynamic_parameters)

        if plot is None:
            continue
        # x_target_values = plot['x_target_values']
        # u_target_values = plot['u_target_values']
        # robot.plot_system(x_target, u_target_values[t], t, alpha=0.2)
        robot.plot_system(x, u, t)
        # plt.scatter(*target_tip_positions[t], color="red")
        plt.pause(0.1)
        plt.close() 
    state_values[-1] = robot.x.copy()
    # z_values = np.concatenate((state_values[:-1], u_values), axis=1)
    return state_values, u_values
    


def adaptive_control(robot, controller, T, x0=None, plot=None):
    robot.reset(x0)
    state_values = np.zeros((T+1, robot.d))
    u_values = np.zeros((T+1, robot.m))
    # target_tip_positions = robot.compute_tip_positions(x_target_values)
    for t in range(T):
        x = robot.x.copy()
        state_values[t] = x
        # if t==0:
        controller.adapt(state_values[t-10:t+1], u_values[t-10:t])
        u = controller.control(state_values, t)
        # print(u)

        robot.step(u)

        if t > T/2:
            robot.mass = 1.2

        u_values[t] = u

        if plot is None:
            continue
        x_target_values = plot['x_target_values']
        u_target_values = plot['u_target_values']
        robot.plot_system(x_target_values[t], u_target_values[t], t, alpha=0.2)
        robot.plot_system(x, u, t)
        # plt.scatter(*target_tip_positions[t], color="red")
        plt.pause(0.1)
        plt.close()
    state_values[-1] = robot.x.copy()
    # z_values = np.concatenate((state_values[:-1], u_values), axis=1)
    return state_values, u_values