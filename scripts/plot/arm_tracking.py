import numpy as np
import matplotlib.pyplot as plt


from robotics.arm import Arm
import torch

from systems import ActuatedArm
from scripts.train.arm import metamodel_choice
from scripts.plot.layout import color_choice
from controller import control_loop, actuate
# np.random.seed(5)
# torch.manual_seed(5)

system = ActuatedArm()

sigma = 1e-4
# sigma = 0

# m2, l1, l2 = 3., 4.95, 1.11
# I2, mll, ml = system.W_test[-2]
# I2, mll, ml = .7, .7, .7
# I2, mll, ml = .3, 1., .8
# I2, mll, ml = .3, 1., .8
# l2 = 3*I2/ml
# l1 = mll/ml
# m2 = ml/l2
# arm = Arm(1., m2, l1, l2, self.alpha)
# l1, l2, m2 = 1.2, 1.4, 1.5
# l1, l2, m2 = 0.6, 0.7, 0.9

gamma = 5
I2, m2 = 0.38, 1.2  
shots = 20
T = 140

# gamma = 8
# I2, m2 = 1., 3.0
# shots = 20
# T = 200

robot = Arm(I2, m2, system.alpha)
dt = robot.dt

plot = False
# plot = True
# d = robot.d
def law(t): 
    # magnitude = gamma/2 + (t/(T*dt))*gamma/2
    magnitude = lambda u: gamma * np.tanh((2+t/(T*dt))*u)
    # period = dt*(100 - 20*np.exp(-(t-dt*T/2)**2))
    period = 100*dt + (t/(T*dt))*10*dt
    return magnitude(np.sin(2*np.pi*t/(period)))
# def law(t):
#     magnitude = gamma
#     # period = dt*(100 - 20*np.exp(-(t-dt*T/2)**2))
#     period = 100*dt
#     return magnitude*np.sin(2*np.pi*t/(period))
t_values = dt*np.arange(T)
u_target_values = law(t_values).reshape(-1, 1)
x_target_values = actuate(robot, u_target_values, plot=plot)
points = system.extract_points(x_target_values)

plot = None
# plot = {'u_target_values': u_target_values, 'x_target_values': x_target_values}

n_gradient = 50_000
n_gradient = 40_000
n_gradient = 60_000
n_gradient = 45_000
n_gradient = 35_000
# n_gradient = 200_000
fig = plt.figure(figsize=(3., 2.))
fig.set_tight_layout(True)

plt.axvspan(-1., 60, facecolor='black', alpha=0.3)


# for model_index, metamodel_name in enumerate(['tldr']):
# for model_index, metamodel_name in enumerate(['tldr', 'anil']):
# for model_index, metamodel_name in enumerate(['tldr', 'maml']):
for model_index, metamodel_name in enumerate(['tldr', 'anil']):

    metamodel = metamodel_choice[metamodel_name]
    path = f'output/models/arm/{metamodel_name}_{n_gradient}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)
    print(f'metamodel {metamodel_name}')
    color = color_choice[metamodel_name]

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
    # plt.subplot(2, 1, 1)
    # plt.plot(u_ff_values.squeeze(), label=metamodel_name, color=color, lw=2.5, alpha=.9)
    # plt.show()

    x_values, u_values = control_loop(robot, u_ff_values, x_target_values, plot=plot)


    # plt.subplot(2, 1, 2)
    # tip_height = x_values[:, 0] + l*np.sin(x_values[:, 2])
    tip_abscissa, tip_ordinate = robot.compute_tip_positions(x_values).T
    plt.plot(tip_ordinate, label=metamodel_name, color=color, lw=2.5, alpha=.8)

    error_values = robot.evaluate_tracking(x_values, x_target_values)
    print(f'agent {metamodel_name}, total error {error_values.sum()}')
    # plt.subplot(2, 1, 3)

    # plt.plot(error_values, label=metamodel_name, color=color)


dynamics_model = robot.inverse_dynamics
u_ff_values = robot.plan_inverse_dynamics(x_target_values)
x_values, u_values = control_loop(robot, u_ff_values, x_target_values, plot=plot)
# plt.subplot(2, 1, 1)
# plt.plot(u_ff_values.squeeze(), color='purple', lw=2.5, alpha=.7)
# plt.plot(u_target_values.squeeze(), ls='--', lw=2.5, color='black')
# plt.ylabel(r'input')
plt.xticks([0, 120], labels=[r'$0$', r'$100$'])
# plt.ylim((-2.1, -0.))
# plt.yticks([])
# plt.ylim((-1.5*gamma, 1.5*gamma))
# plt.subplot(2, 1, 2)
plt.ylabel(r'tip height')
plt.xlabel(r'time')
plt.gca().xaxis.set_label_coords(.35, -0.1)
plt.title('arm')
tip_abscissa, tip_ordinate = robot.compute_tip_positions(x_values).T
target_abscissa, target_ordinate = robot.compute_tip_positions(x_target_values).T
plt.plot(target_ordinate, ls='--', lw=2.5, color='black')
plt.plot(tip_ordinate, color='slategrey', lw=2.5, alpha=.7)  
error_values = robot.evaluate_tracking(x_values, x_target_values)
print(f'agent analytic, total error {error_values.sum()}')
# plt.suptitle(r'arm')
plt.text(0.1, -.6, 'adaptation', fontsize=12)

# plt.subplot(2, 1, 3)
# plt.plot(error_values, ls='--', color='black')

# # plt.plot(error_values, label='target')

# # plt.legend()
plt.savefig('output/plots/tracking_arm.pdf')
plt.show()