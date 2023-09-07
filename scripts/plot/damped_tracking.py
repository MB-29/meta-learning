import numpy as np
import matplotlib.pyplot as plt


from robotics.cartpole import Cartpole
import torch

from systems import DampedActuatedCartpole
from scripts.train.damped_cartpole import metamodel_choice
from scripts.plot.layout import color_choice
from controller import control_loop, actuate

np.random.seed(5)
torch.manual_seed(5)

system = DampedActuatedCartpole()

dt = 0.02
sigma = .0001
alpha, beta = 0., 0.1
gamma = 5
Mass, mass = 1.5, .6
Mass, mass = 1.2, .25
Mass, mass = 1.4, .7
Mass, mass = 1.9, .5
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
shots = 10
fig = plt.figure(figsize=(4.5, 3))
fig.set_tight_layout(True)
# for model_index, metamodel_name in enumerate(['tldr']):
for model_index, metamodel_name in enumerate(['tldr', 'anil']):
# for model_index, metamodel_name in enumerate(['tldr', 'anil', 'maml', 'coda']):

    metamodel = metamodel_choice[metamodel_name]
    path = f'output/models/damped_cartpole/{metamodel_name}_{n_gradient}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)

    test_dataset = system.extract_data(x_target_values, u_target_values)
    test_points, test_targets = test_dataset
    adaptation_points, adaptation_targets = test_points[:shots], test_targets[:shots]
    adaptation_dataset = (adaptation_points, adaptation_targets)
    adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=100)

    # model = format_model(adapted_model)
    # u_ff_values = plan_inverse_dynamics(robot, model, x_target_values)

    u_ff_values = adapted_model(points).detach().numpy().reshape(-1, 1)
    # model = robot.inverse_dynamics
# for model_name, model in models.items():

    x_values, u_values = control_loop(robot, u_ff_values, x_target_values, plot=plot)

    color = color_choice[metamodel_name]

    # plt.subplot(2, 1, 1)
    # plt.plot(u_ff_values.squeeze(), label=metamodel_name, color=color, lw=2.5, alpha=.9)

    # plt.subplot(2, 1, 2)
    tip_height = x_values[:, 0] + l*np.sin(x_values[:, 2])
    tip_height = -l*np.cos(x_values[:, 2])
    plt.plot(tip_height, label=metamodel_name, color=color, lw=2.5, alpha=.8)

    error_values = robot.evaluate_tracking(x_values, x_target_values)
    print(f'model {metamodel_name}, total error {error_values.sum()}')
    # plt.subplot(2, 1, 3)

    # plt.plot(error_values, label=metamodel_name, color=color)

# dynamics_model = robot.inverse_dynamics
u_ff_values = robot.plan_inverse_dynamics(x_target_values)
x_values, u_values = control_loop(robot, u_ff_values, x_target_values, plot=plot)

# plt.subplot(2, 1, 1)
# plt.plot(u_ff_values.squeeze(), lw=2.5, color='indigo', alpha=.8)

# plt.subplot(2, 1, 2)
tip_height = x_values[:, 0] + l*np.sin(x_values[:, 2])
tip_height = -l*np.cos(x_values[:, 2])
plt.plot(tip_height, color='indigo', lw=2.5, alpha=.8, label='analytic')
# plt.gca().get_yaxis().set_label_coords(-0.1,0.5)

error_values = robot.evaluate_tracking(x_values, x_target_values)
print(f'analytic model, total error {error_values.sum()}')

# plt.subplot(2, 1, 1)
# plt.plot(u_target_values.squeeze(), lw=2.5, ls='--', color='black')
# plt.ylabel(r'input')
# plt.xticks([])

# plt.subplot(2, 1, 2)
plt.ylim((-1.1, 1.1))
plt.yticks((-1, 1))
plt.ylabel(r'tip height')
plt.xlabel(r'time')
tip_height = x_target_values[:, 0] + l*np.sin(x_target_values[:, 2])
tip_height = -l*np.cos(x_target_values[:, 2])
plt.plot(tip_height, color='black', ls='--', lw=2.5, label='target')
# plt.gca().get_yaxis().set_label_coords(-0.2,0.5)
error_values = robot.evaluate_tracking(x_values, x_target_values)
# plt.suptitle(r'damped cartpole inverse dynamics tracking')
plt.suptitle(r'cartpole')
# plt.subplot(2, 1, 3)
# plt.plot(error_values, ls='--', color='black')

# plt.plot(error_values, label='target')
fig.align_labels()

# plt.legend(loc='center left')
plt.savefig('output/plots/tracking_damped_cartpole.pdf')
plt.show()