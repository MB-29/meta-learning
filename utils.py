import numpy as np
import torch
from scipy.integrate import odeint
from torchdiffeq import odeint as d_odeint

# def define_flow(V, c, w, diff):

#     if not diff:

    
#     else:

        

# integrator_choice = {True: d_odeint, False: odeint}

# def integrate_trajectory(x0, V, c, w, time_values):
#     if x0.ndim > 1:
#         def d_flow(time, x):
#             x_dot = V(x)@w + c(x)
#             return x_dot

#         return d_odeint(d_flow, x0, time_values)
#     else:
#         def flow(x, time):
#             x = torch.tensor(x).unsqueeze(0)
#             x_dot = V(x)@w + c(x)
#             return x_dot.squeeze()
#     # solution = odeint(flow, x0, time_values, args=(beta, delta))
#         solution = odeint(flow, x0, time_values)
#     return solution
def integrate_trajectory(x0, model, time_values):
    if x0.ndim > 1:
        def d_flow(time, x):
            x_dot = model(x)
            return x_dot

        return d_odeint(d_flow, x0, time_values)
    else:
        def flow(x, time):
            x = torch.tensor(x).unsqueeze(0)
            x_dot = model(x)
            return x_dot.squeeze()
    # solution = odeint(flow, x0, time_values, args=(beta, delta))
        solution = odeint(flow, x0, time_values)
    return solution

def build_grid(a_values, b_values):
    length = len(a_values)
    assert len(b_values == length)
    grid = np.zeros((length**2, 2))
    for i in range(length):
        for j in range(length):
            a, b = a_values[i], b_values[j]
            index = length*i + j
            grid[index] = a, b
    W_train = torch.tensor(W_train, dtype=torch.float)
    training_task_samples = 4