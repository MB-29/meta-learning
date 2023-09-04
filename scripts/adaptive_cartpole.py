import numpy as np
import matplotlib.pyplot as plt
import pickle

from robotics.cartpole import Cartpole
import torch

from systems import DampedActuatedCartpole
from scripts.train.damped_cartpole import metamodel_choice
from scripts.plot.layout import color_choice
from controller import Controller, adaptive_control, actuate

np.random.seed(5)
torch.manual_seed(5)

system = DampedActuatedCartpole()

dt = 0.02
sigma, alpha, beta = 0, 0., 0.1
gamma = 5
Mass, mass = 1., .5
l = 1
robot = Cartpole(mass, Mass, l, alpha, beta)


with open('output/u.pkl', 'rb') as file:
    u = pickle.load(file)
# d = robot.d
T = 100
x0 = np.array([0, 0, np.pi, 0])
# u_target_values = np.zeros((T, 1))
u_target_values = u.reshape(-1, 1)
# u_target_values[100:] = 1.08
x_target_values = actuate(robot, u_target_values, x0=x0)
points = system.extract_points(x_target_values)

plot = {'u_target_values': u_target_values, 'x_target_values': x_target_values}
plot = None



n_gradient = 50_000
shots = 30
fig = plt.figure(figsize=(8, 6))
fig.set_tight_layout(True)

test_dataset = system.extract_data(x_target_values, u_target_values)
test_points, test_targets = test_dataset
adaptation_points, adaptation_targets = test_points[:shots], test_targets[:shots]
adaptation_dataset = (adaptation_points, adaptation_targets)

# for model_index, metamodel_name in enumerate(['tldr']):
# for model_index, metamodel_name in enumerate(['tldr', 'anil', 'maml']):
for model_index, metamodel_name in enumerate(['tldr', 'maml']):

    metamodel = metamodel_choice[metamodel_name]
    path = f'output/models/damped_cartpole/{metamodel_name}_{n_gradient}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)

    controller = Controller(
        metamodel,
        x_target_values,
        adaptation_dataset,
        system.extract_data,
        shots=shots
        )
   
    # controller.adapt(x_target_values, u_target_values)
    # adapted_model = control.
    # u_ff_values = controller.adapted_model(points).detach().numpy().reshape(-1, 1)

    # model = format_model(adapted_model)
    # u_values = plan_inverse_dynamics(robot, model, x_target_values)

    # model = robot.inverse_dynamics
# for model_name, model in models.items():
    # controller.adapt(x_target_values, u_target_values)
    state_values, u_values = adaptive_control(robot, controller, T, x0=x0)
    # u_ff_values = controller.adapted_model(points).detach().numpy().reshape(-1, 1)

    # x_values, u_values = control_loop(robot, u_values, x_target_values, x0=x0, plot=plot)

    color = color_choice[metamodel_name]

    plt.subplot(2, 1, 1)
    plt.plot(u_values.squeeze(), label=metamodel_name, color=color, lw=2.5, alpha=.9)

    plt.subplot(2, 1, 2)
    tip_abscissa = state_values[:, 0] + l*np.sin(state_values[:, 2])
    tip_abscissa = -l*np.cos(state_values[:, 2])
    plt.plot(tip_abscissa, label=metamodel_name, color=color, lw=2.5, alpha=.8)

    error_values = robot.evaluate_tracking(state_values, x_target_values)
    print(f'model {metamodel_name}, total error {error_values.sum()}')
    # plt.subplot(2, 1, 3)

    # plt.plot(error_values, label=metamodel_name, color=color)

dynamics_model = robot.inverse_dynamics
u_values = robot.plan_inverse_dynamics(x_target_values)
state_values, u_values = control_loop(robot, u_values, x_target_values, x0=x0, plot=plot)
plt.subplot(2, 1, 1)
plt.plot(u_values.squeeze(), lw=2.5, color='indigo', alpha=.8)
plt.subplot(2, 1, 2)
tip_abscissa = state_values[:, 0] + l*np.sin(state_values[:, 2])
# tip_abscissa = -l*np.cos(state_values[:, 2])
plt.plot(tip_abscissa, color='indigo', lw=2.5, alpha=.8, label='analytic')
error_values = robot.evaluate_tracking(state_values, x_target_values)
print(f'analytic model, total error {error_values.sum()}')
# plt.plot(error_values, ls='--', color='indigo', label='analytic')

plt.subplot(2, 1, 1)
plt.plot(u_target_values.squeeze(), lw=2.5, ls='--', color='black')
plt.ylabel(r'input')
plt.xticks([])
plt.subplot(2, 1, 2)
plt.ylabel(r'tip height')
plt.xlabel(r'time')
tip_abscissa = x_target_values[:, 0] + l*np.sin(x_target_values[:, 2])
tip_abscissa = -l*np.cos(x_target_values[:, 2])
plt.plot(tip_abscissa, color='black', ls='--', lw=2.5, label='target')
plt.suptitle(r'damped cartpole inverse dynamics tracking')
# plt.subplot(2, 1, 3)

# plt.plot(error_values, label='target')

plt.legend(loc='center left')
# plt.savefig('output/plots/tracking_damped_cartpole.pdf')
plt.show()