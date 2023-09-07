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

np.random.seed(5)
torch.manual_seed(5)

system = DampedActuatedCartpole()
mass_values = np.array(
    [[0.8, 0.2],
     [1.8, 0.2],
     [0.5, 0.5],
     [1.5, 0.5]])

meta_dataset = system.generate_training_data()

n_obs = 1_000
sigma, alpha, beta = 0, 0., 0.1
gamma = 2
Mass, mass = 1.9, .6
l = 1
robot = Cartpole(mass, Mass, l, alpha=alpha, beta=beta)
# dt = 1/200
dt = robot.dt

def law(t):
    magnitude = gamma/3 
    # magnitude = lambda u: gamma * np.tanh((2+t/(n_obs*dt))*u)
    # period = dt*(100 - 20*np.exp(-(t-dt*T/2)**2))
    period = 400*dt + (t/(n_obs*dt))*400*dt
    # period = 300*dt
    return magnitude*(np.sin(2*np.pi*t/(period)))

# d = robot.d
# T = 100
# x0 = np.array([0, 0, np.pi, 0])
# u_target_values = np.zeros((T, 1))
t_values = dt*np.arange(n_obs)
u_target_values = law(t_values).reshape(-1, 1)

# u_target_values = u.reshape(-1, 1)
# u_target_values[100:] = 1.08
x_target_values = actuate(robot, u_target_values)
points = system.extract_points(x_target_values)

plot = None
# plot = {'u_target_values': u_target_values, 'x_target_values': x_target_values}


n_gradient = 50_000
shots = 50
min_shots = 5
fig = plt.figure(figsize=(8, 6))
fig.set_tight_layout(True)

# plt.subplot(2, 1, 1)
# plt.plot(u_target_values.squeeze(), lw=2.5, ls='--', color='black')
# plt.ylabel(r'input')
# plt.xticks([])
plt.subplot(2, 1, 1)
plt.ylabel(r'tip height')
plt.xlabel(r'time')
tip_abscissa = x_target_values[:, 0] + l*np.sin(x_target_values[:, 2])
# tip_abscissa = -l*np.cos(x_target_values[:, 2])
plt.plot(tip_abscissa, color='black', ls='--', lw=2.5, label='target')
plt.suptitle(r'damped cartpole inverse dynamics tracking')
plt.subplot(2, 1, 2)
plt.plot(400*[1.9] + 600*[1.7], ls='--', color='black', lw=2.5)
plt.ylim((1, 2))


u_values = robot.plan_inverse_dynamics(x_target_values)
state_values, u_values = control_loop(robot, u_values, x_target_values, plot=plot)
# plt.subplot(2, 1, 1)
# plt.plot(u_values.squeeze(), lw=2.5, color='darkgray', alpha=1.)
plt.subplot(2, 1, 1)
tip_abscissa = state_values[:, 0] + l*np.sin(state_values[:, 2])
# tip_abscissa = -l*np.cos(state_values[:, 2])
plt.plot(tip_abscissa, color='darkgray', lw=2.5, alpha=1., label='analytic')
error_values = robot.evaluate_tracking(state_values, x_target_values)
print(f'analytic model, total error {error_values.sum()}')

test_dataset = system.extract_data(x_target_values, u_target_values)
test_points, test_targets = test_dataset
adaptation_points, adaptation_targets = test_points[:shots], test_targets[:shots]
adaptation_dataset = (adaptation_points, adaptation_targets)


plot = {'u_target_values': u_target_values, 'x_target_values': x_target_values}
plot = None


# for model_index, metamodel_name in enumerate(['tldr', 'anil', 'maml']):
# for model_index, metamodel_name in enumerate(['tl$dr', 'maml']):
# for model_index, metamodel_name in enumerate(['tldr']):
for model_index, metamodel_name in enumerate(['tldr', 'anil']):
# for model_index, metamodel_name in enumerate(['tldr', 'coda', 'anil']):

    metamodel = metamodel_choice[metamodel_name]
    path = f'output/models/damped_cartpole/{metamodel_name}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)
    architecture = metamodel_name.split('_')[0]

    context_values = metamodel.get_context_values(meta_dataset, n_steps=50).detach().numpy()
    supervised_contexts = mass_values
    learned_contexts = context_values
    interpreter = estimate_context_transform(learned_contexts, supervised_contexts, affine=True)

    controller = Controller(
        metamodel,
        x_target_values,
        adaptation_dataset,
        system.extract_data,
        min_shots=min_shots
        )
   
    state_values, u_values, estimates = adaptive_control(robot, controller, n_obs, interpreter=interpreter, plot=plot)

    color = color_choice[architecture]

    # plt.subplot(2, 1, 1)
    # plt.plot(u_values.squeeze(), label=metamodel_name, color=color, lw=2.5, alpha=.9)

    plt.subplot(2, 1, 1)
    tip_abscissa = state_values[:, 0] + l*np.sin(state_values[:, 2])
    # tip_abscissa = -l*np.cos(state_values[:, 2])
    plt.plot(tip_abscissa, label=metamodel_name, color=color, lw=2.5, alpha=.8)

    plt.subplot(2, 1, 2)
    error_values = robot.evaluate_tracking(state_values, x_target_values)
    print(f'model {metamodel_name}, total error {error_values.sum()}')

    plt.plot(estimates[:, 0], color=color, lw=2.5)
    # plt.plot(estimates[:, 1], ls='-.', color=color)


# plt.plot(error_values, ls='--', color='indigo', label='analytic')


# plt.ylim((1.5, 2.5))
# plt.subplot(2, 1, 3)

# plt.plot(error_values, label='target')

# plt.legend(loc='center left')
# plt.savefig('output/plots/tracking_damped_cartpole.pdf')
plt.show()