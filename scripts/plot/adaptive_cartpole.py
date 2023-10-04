import numpy as np
import matplotlib.pyplot as plt
import pickle

from robotics.cartpole import Cartpole
import torch

from systems import DampedActuatedCartpole
from scripts.train.damped_cartpole import metamodel_choice
from scripts.plot.layout import color_choice
from controller import Controller, adaptive_control, actuate, control_loop
from interpret import estimate_context_transform
from models import CAMEL
# np.random.seed(5)
# torch.manual_seed(5)

system = DampedActuatedCartpole()
# mass_values = np.array(
#     [[0.8, 0.2],
#      [1.8, 0.2],
#      [0.5, 0.5],
#      [1.5, 0.5]])
mass_values = system.W_train @ np.array([[1.0, 0.], [-1., 1]])
meta_dataset = system.generate_training_data()

n_obs = 1_200
sigma, alpha, beta = 0, 0., 0.1
gamma = 4
Mass, mass = 1.5, .3
l = 1
robot = Cartpole(mass, Mass, l, alpha=alpha, beta=beta, sigma=sigma)
dt = robot.dt

t_values = dt*np.arange(n_obs)
magnitude_values = np.where(t_values < 1_000*dt, gamma, gamma)


def law(t):
    # magnitude = magnitude_values[t]
    # magnitude = gamma - (t/(1_500*dt) - 0.5)**2*gamma
    # magnitude = lambda u: gamma *(u**3)
    # period = dt*(100 - 20*np.exp(-(t-dt*n_obs/2)**2))
    period = 500*dt - (t/(n_obs*dt))*00*dt
    # period = 500*dt
    return (np.sin(2*np.pi*t/(period)))


# d = robot.d
# T = 100
# x0 = np.array([0, 0, np.pi, 0])
# u_target_values = np.zeros((T, 1))
u_target_values = (magnitude_values*law(t_values)).reshape(-1, 1)

# u_target_values = u.reshape(-1, 1)
# u_target_values[100:] = 1.08
x_target_values = actuate(robot, u_target_values)
points = system.extract_points(x_target_values)

plot = None
# plot = {'u_target_values': u_target_values, 'x_target_values': x_target_values}


n_gradient = 50_000
shots = 200
min_shots = 30
fig = plt.figure(figsize=(3, 2.8))
# fig.set_tight_layout(True)
fig.subplots_adjust(left=0.2, wspace=0.1, hspace=0.1)


u_values = robot.plan_inverse_dynamics(x_target_values)
state_values, u_values = control_loop(
    robot, u_values, x_target_values, plot=plot)
# plt.subplot(2, 1, 1)
# plt.plot(u_values.squeeze(), lw=2.5, color='slategrey', alpha=1.)
plt.subplot(2, 1, 1)
tip_abscissa = state_values[:, 0] + l*np.sin(state_values[:, 2])
# tip_abscissa = -l*np.cos(state_values[:, 2])
# plt.plot(tip_abscissa, color='slategrey', lw=2.5, alpha=1., label='analytic')
error_values = robot.evaluate_tracking(state_values, x_target_values)
print(f'analytic model, total error {error_values.sum()}')

test_dataset = system.extract_data(x_target_values, u_target_values)
test_points, test_targets = test_dataset
adaptation_points, adaptation_targets = test_points[:shots], test_targets[:shots]
adaptation_dataset = (adaptation_points, adaptation_targets)


plot = {'u_target_values': u_target_values, 'x_target_values': x_target_values}
plot = None

analytic_model = CAMEL(system.T, system.r, system.V_star)
metamodel_choice['analytic'] = analytic_model

# for model_index, metamodel_name in enumerate(['tldr', 'anil', 'maml']):
# for model_index, metamodel_name in enumerate(['tl$dr', 'maml']):
# for model_index, metamodel_name in enumerate(['tldr', 'coda', 'anil']):
# for model_index, metamodel_name in enumerate(['tldr_50000_T_9']):
# for model_index, metamodel_name in enumerate(['tldr_r_1']):
# for model_index, metamodel_name in enumerate(['tldr', 'coda', 'anil']):
# for model_index, metamodel_name in enumerate(['tldr', 'coda']):
for model_index, metamodel_name in enumerate(['tldr', 'anil', 'analytic']):
    # for model_index, metamodel_name in enumerate(['tldr']):
    architecture = metamodel_name.split('_')[0]
    if metamodel_name == 'analytic':
        metamodel = analytic_model
        interpreter = np.eye(system.r+1, system.r)

    else:

        metamodel = metamodel_choice[architecture]
        path = f'output/models/damped_cartpole/{metamodel_name}.ckpt'
        checkpoint = torch.load(path)
        metamodel.load_state_dict(checkpoint)
        architecture = metamodel_name.split('_')[0]

        context_values = metamodel.get_context_values(
            meta_dataset, n_steps=50).detach().numpy()
        supervised_contexts = mass_values
        learned_contexts = context_values
        interpreter = estimate_context_transform(
            learned_contexts, supervised_contexts, affine=True)
    

    controller = Controller(
        metamodel,
        x_target_values,
        adaptation_dataset,
        system.extract_data,
        min_shots=min_shots
    )

    state_values, u_values, estimates = adaptive_control(
        robot, controller, n_obs, interpreter=interpreter, plot=plot)

    color = color_choice[architecture]

    # plt.subplot(2, 1, 1)
    # plt.plot(u_values.squeeze(), label=metamodel_name, color=color, lw=2.5, alpha=.9)

    plt.subplot(2, 1, 1)
    tip_abscissa = state_values[:, 0] + l*np.sin(state_values[:, 2])
    # tip_abscissa = -l*np.cos(state_values[:, 2])
    plt.plot(tip_abscissa, label=architecture, color=color, lw=2.5, alpha=.8)

    plt.subplot(2, 1, 2)
    error_values = robot.evaluate_tracking(state_values, x_target_values)
    print(f'model {architecture}, total error {error_values.sum()}')

    plt.plot(estimates[:, 0], color=color, lw=2.5, alpha=.8)

    # plt.plot(estimates[:, 1], ls='-.', color=color)


plt.subplot(2, 1, 1)
plt.title(r'cartpole')
plt.ylabel(r'tip position')
tip_abscissa = x_target_values[:, 0] + l*np.sin(x_target_values[:, 2])
plt.xticks([])
# tip_abscissa = -l*np.cos(x_target_values[:, 2])
plt.plot(tip_abscissa, color='black', ls='--', lw=2.5, label='target')
plt.subplot(2, 1, 2)
plt.xticks([0, 1000])
plt.yticks([1.5, 3])
plt.xlabel(r'time')
plt.gca().xaxis.set_label_coords(.4, -0.1)
plt.ylabel(r'mass')
plt.plot(300*[1.5], ls='--', color='black', lw=3.)
plt.plot(np.arange(300, n_obs), (n_obs-300)*[3], ls='--', color='black', lw=3.)
plt.ylim((1., 4.))

fig.align_ylabels()
# plt.plot(error_values, ls='--', color='indigo', label='analytic')


# plt.ylim((1.5, 2.5))
# plt.subplot(2, 1, 3)  

# plt.plot(error_values, label='target')

# plt.legend(loc='center left')
plt.savefig('output/plots/adaptive_cartpole.pdf')
plt.show()
