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