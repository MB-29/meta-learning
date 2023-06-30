import numpy as np
import matplotlib.pyplot as plt


from robotics.cartpole import Cartpole
import torch

from systems import ActuatedCartpole
from scripts.train.cartpole import metamodel_choice
from plots.layout import color_choice

np.random.seed(5)
torch.manual_seed(5)

system = ActuatedCartpole()

dt = 0.02
sigma, alpha, beta = 0, 0., 0.
gamma = 5
Mass, mass = 1.5, .6
Mass, mass = 1.2, .25
# Mass, mass = 1.4, .7
# Mass, mass = .9, .2
l = 1
robot = Cartpole(mass, Mass, l, alpha, beta)


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
x_target_values = robot.actuate(u_target_values)
points = system.extract_points(x_target_values)

plot = {'u_target_values': u_target_values}
plot = None



n_gradient = 50_000
shots = 30
fig = plt.figure(figsize=(4, 4))
fig.set_tight_layout(True)

# for model_index, metamodel_name in enumerate(['tldr']):
for model_index, metamodel_name in enumerate(['tldr', 'anil']):
# for model_index, metamodel_name in enumerate(['tldr', 'anil', 'maml', 'coda']):

    metamodel = metamodel_choice[metamodel_name]
    path = f'output/models/cartpole/{metamodel_name}_{n_gradient}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)
    print(f'metamodel {metamodel_name}')

    test_dataset = system.extract_data(x_target_values, u_target_values)
    test_points, test_targets = test_dataset
    adaptation_points, adaptation_targets = test_points[:shots], test_targets[:shots]
    adaptation_dataset = (adaptation_points, adaptation_targets)
    adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=30)

    # model = format_model(adapted_model)
    # u_ff_values = plan_inverse_dynamics(robot, model, x_target_values)

    u_ff_values = adapted_model(points).detach().numpy().reshape(-1, 1)
    # model = robot.inverse_dynamics
# for model_name, model in models.items():

    x_values, u_values = robot.control_loop(x_target_values, u_ff_values, plot)

    color = color_choice[metamodel_name]

    plt.subplot(2, 1, 1)
    plt.plot(u_ff_values.squeeze(), label=metamodel_name, color=color, lw=2.5, alpha=.9)

    plt.subplot(2, 1, 2)
    tip_height = x_values[:, 0] + l*np.sin(x_values[:, 2])
    tip_height = -l*np.cos(x_values[:, 2])
    plt.plot(tip_height, label=metamodel_name, color=color, lw=2.5, alpha=.8)

    error_values = robot.evaluate_tracking(x_values, x_target_values)
    print(f'total error {error_values.sum()}')
    # plt.subplot(2, 1, 3)

    # plt.plot(error_values, label=metamodel_name, color=color)


dynamics_model = robot.inverse_dynamics
u_ff_values = robot.plan_inverse_dynamics(dynamics_model, x_target_values)
x_values, u_values = robot.control_loop(x_target_values, u_ff_values, plot)
plt.subplot(2, 1, 1)
plt.plot(u_ff_values.squeeze(), ls='--', color='black')
plt.ylabel(r'input')
plt.xticks([])
plt.subplot(2, 1, 2)
plt.ylabel(r'tip height')
plt.xlabel(r'time')
tip_height = x_values[:, 0] + l*np.sin(x_values[:, 2])
tip_height = -l*np.cos(x_values[:, 2])
plt.plot(tip_height, label=metamodel_name, color='black', ls='--', lw=2.5, alpha=.8)
error_values = robot.evaluate_tracking(x_values, x_target_values)
print(f'total error {error_values.sum()}')
# plt.subplot(2, 1, 3)
# plt.plot(error_values, ls='--', color='black')

# plt.plot(error_values, label='target')

# plt.legend()
plt.savefig('output/plots/tracking_cartpole.pdf')
plt.show()