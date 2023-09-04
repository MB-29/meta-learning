import numpy as np
import matplotlib.pyplot as plt


from robotics.cartpole import Cartpole
import torch

from systems import DampedActuatedCartpole
from scripts.train.damped_cartpole import metamodel_choice
from scripts.plot.layout import color_choice
from interpret import estimate_context_transform
from controller import actuate

np.random.seed(5)
torch.manual_seed(5)

affine = True

mass_values = np.array(
    [[0.8, 0.2],
     [1.8, 0.2],
     [0.5, 0.5],
     [1.5, 0.5]])

sigma = 0
sigma = 1e-3
system = DampedActuatedCartpole(sigma=sigma)
meta_dataset = system.generate_training_data()
dt = 0.02
sigma = .0001
alpha, beta = 0., 0.1
gamma = 5
Mass, mass = 1.5, .6
Mass, mass = 1.2, .25
Mass, mass = 1.4, .7
Mass, mass = 3.4, 1.2
Mass, mass = 1.7, .4
Mass, mass = 1., .2

# Mass, mass = .9, .2
l = 1
robot = Cartpole(mass, Mass, l, alpha, beta, sigma=sigma)


# d = robot.d
T = 200
def law(t):
    magnitude = gamma/3 + (t/(T*dt))*gamma/3
    magnitude = lambda u: gamma * np.tanh((2+t/(T*dt))*u)
    # period = dt*(100 - 20*np.exp(-(t-dt*T/2)**2))
    period = 50*dt + (t/(T*dt))*50*dt
    return magnitude(np.sin(2*np.pi*t/(period)))
# def law(t):
#     magnitude = gamma
#     # period = dt*(100 - 20*np.exp(-(t-dt*T/2)**2))
#     period = 100*dt
#     return magnitude*np.sin(2*np.pi*t/(period))
t_values = dt*np.arange(T)
u_target_values = law(t_values).reshape(-1, 1)
x_target_values = actuate(robot, u_target_values)
points = system.extract_points(x_target_values)

plot = {'u_target_values': u_target_values, 'x_target_values': x_target_values}
plot = None


n_gradient = 50_000
max_shots = 100
fig = plt.figure(figsize=(4.2, 3))
fig.set_tight_layout(True)
# for model_index, metamodel_name in enumerate(['tldr']):
shot_values = np.arange(0, max_shots)  
# for model_index, metamodel_name in enumerate(['tldr', 'anil', 'coda']):
for model_index, metamodel_name in enumerate(['tldr', 'anil']):
# for model_index, metamodel_name in enumerate(['tldr', 'anil', 'maml', 'coda']):

    metamodel = metamodel_choice[metamodel_name]
    path = f'output/models/damped_cartpole/{metamodel_name}_{n_gradient}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)


    context_values = metamodel.get_context_values(meta_dataset, n_steps=50).detach().numpy()
    supervised_contexts = mass_values
    learned_contexts = context_values
    context_estimator = estimate_context_transform(learned_contexts, supervised_contexts, affine=affine)
    for shots in shot_values:
        test_dataset = system.extract_data(x_target_values, u_target_values)
        test_points, test_targets = test_dataset
        adaptation_points, adaptation_targets = test_points[:shots], test_targets[:shots]
        adaptation_dataset = (adaptation_points, adaptation_targets)
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=100)
        w = adapted_model.get_context()
                # print(w)
        # print(f'w = {w}')
        w_bar = w.squeeze().numpy()
        if affine:
            w_bar = np.append(w, 1.0)
        w_hat = context_estimator.T @ w_bar
        w_star = np.array([Mass, mass])
        error = np.linalg.norm(w_hat-w_star)
        mape = np.mean(np.abs(w_hat - w_star) / np.abs(w_star))
        # parameter_error = np.linalg.norm(w_hat - w_star)
        plt.scatter(shots, mape, marker='x',
                    color=color_choice[metamodel_name])
        # print(f'w_hat = {w_hat}, error = {mape}')
plt.yscale('log')
plt.show()