import numpy as np
import matplotlib.pyplot as plt

def actuate(environment, u_values):
    T = u_values.shape[0]
    environment.reset()
    x_values = np.zeros((T+1, environment.d))
    for t in range(T):
        x = environment.x.copy()
        x_values[t] = x
        # u = np.random.choice([-1.0, 1.0], size=1)
        u = u_values[t]
        environment.step(u)

        # environment.plot_system(x, u, t)
        # plt.pause(0.1)
        # plt.close()
    x_values[T] = environment.x.copy()
    # z_values = np.concatenate((x_values, u_values), axis=1)
    return x_values