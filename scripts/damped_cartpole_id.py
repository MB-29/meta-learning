import numpy as np
import matplotlib.pyplot as plt


from robotics.cartpole import Cartpole
import torch

from models.tldr import TLDR
from systems import DampedActuatedCartpole
from scripts.train.damped_cartpole import metamodel_choice, V_net, meta_dataset, r
from scripts.plot.layout import color_choice

np.random.seed(5)
torch.manual_seed(5)

system = DampedActuatedCartpole()

dt = 0.02
sigma, alpha, beta = 0, 0., 0.1
gamma = 5
Mass, mass = 1.5, .6
Mass, mass = 1.2, .25
Mass, mass = 1.4, .7
Mass, mass = .9, .2
mass_values = np.array(
    [[0.8, 0.2],
     [1.8, 0.2],
     [0.5, 0.5],
     [1.5, 0.5]])
w_star = torch.tensor([Mass, mass])
# mass_values = np.array([[0.8, 0.2, 0.8**2],
#                         [1.8, 0.2, 1.8**2],
#                         [0.5, 0.5, 0.5**2],
#                         [1.5, 0.5, 1.5*2]])
# w_star = torch.tensor([Mass, mass, Mass**2])
l = 1
robot = Cartpole(mass, Mass, l, alpha, beta)


# d = robot.d
T = 200


def law(t):
    magnitude = gamma/3 + (t/(T*dt))*gamma/3
    def magnitude(u): return gamma * np.tanh((2+t/(T*dt))*u)
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

fig = plt.figure(figsize=(4, 4))
fig.set_tight_layout(True)

# for model_index, metamodel_name in enumerate(['tldr']):
# for model_index, metamodel_name in enumerate(['tldr', 'anil', 'maml', 'coda']):
metamodel_name = 'tldr'
metamodel_name = 'anil'
metamodel_name = 'maml'
# for metamodel_name in ['tldr', 'anil']:
for metamodel_name in ['tldr']:
    metamodel = metamodel_choice[metamodel_name]
    path = f'output/models/damped_cartpole/{metamodel_name}_{n_gradient}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)
    print(f'metamodel {metamodel_name}')

    if metamodel_name in ['maml', 'anil']:
        metamodel.adapt_heads(meta_dataset, n_steps=50)
    W_bar = metamodel.W.data
    T_train = system.T
    tldr = TLDR(T_train, r, V_net, W=W_bar)
    V_hat, W_hat = tldr.calibrate(mass_values)

    shot_values = np.arange(2, 20)
    for shots in shot_values:
        test_dataset = system.extract_data(x_target_values, u_target_values)
        test_points, test_targets = test_dataset
        adaptation_points, adaptation_targets = test_points[:shots], test_targets[:shots]
        adaptation_dataset = (adaptation_points, adaptation_targets)
        adapted_model = tldr.adapt_task_model(adaptation_dataset, n_steps=100)

        parameter_error = torch.norm(adapted_model.w - w_star)
        plt.scatter(shots, parameter_error, marker='x',
                    color=color_choice[metamodel_name])
plt.yscale('log')
plt.show()


estimator, residuals, rank, s = np.linalg.lstsq(
        W_bar, mass_values, rcond=None) 
print(residuals)
# model = format_model(adapted_model)
# u_ff_values = plan_inverse_dynamics(robot, model, x_target_values)
