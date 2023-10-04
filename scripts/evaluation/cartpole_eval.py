import numpy as np
import matplotlib.pyplot as plt


from robotics.cartpole import Cartpole
import torch

from systems import DampedActuatedCartpole, ActuatedArm
from scripts.train.damped_cartpole import metamodel_choice
# from scripts.train.arm import metamodel_choice
from scripts.plot.layout import color_choice
from controller import control_loop
from meta_training import test_model

np.random.seed(5)
torch.manual_seed(5)

system = DampedActuatedCartpole()
# system = ActuatedArm()

n_test = 20
W_test = (np.random.random((n_test, 2)) - 0.5)*np.array([1, .2]) + np.array([2., .3])
# W_test = (2*np.random.random((n_test, 2)) - 1)*np.array([.15, .1]) + np.array([.3, 1])

test_meta_dataset = system.generate_data(W_test)
max_shots = 50
shot_values = np.arange(1, max_shots, 5)
shot_values = [5, 10, 30, 50, 100, 200]
shot_values = [100]
results = {}
# for model_index, metamodel_name in enumerate(['tldr']):
for model_index, metamodel_name in enumerate(['tldr', 'coda', 'maml', 'anil']):
    adaptation_error_values = []
# for model_index, metamodel_name in enumerate(['tldr', 'anil', 'maml', 'coda']):
    architecture = metamodel_name.split('_')[0]
    metamodel = metamodel_choice[metamodel_name]
    path = f'output/models/damped_cartpole/{metamodel_name}.ckpt'
    # path = f'output/models/arm/{metamodel_name}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)
    for shots in shot_values:
        adaptation_indices = np.arange(0, shots)
        adaptation_error = test_model(metamodel, test_meta_dataset, adaptation_indices, n_steps=20)
        adaptation_error_values.append(adaptation_error.copy())
        results[architecture] = adaptation_error_values


# fig = plt.figure(figsize=(3, 2.5))
fig = plt.figure()
fig.set_tight_layout(True)

for model_name, adaptation_error_values in results.items():
    color = color_choice[model_name]
    # plt.plot(shot_values, adaptation_error_values, color=color)
    plt.plot(shot_values, np.mean(adaptation_error_values, axis=1), color=color)
    # plt.plot(shot_values, np.median(adaptation_error_values, axis=1), color=color, ls='--')
    min_values = np.array(adaptation_error_values).min(axis=1)
    max_values = np.array(adaptation_error_values).max(axis=1)
    q1_values = np.quantile(adaptation_error_values, 0.25, axis=1)
    q3_values = np.quantile(adaptation_error_values, 0.75, axis=1)
    plt.fill_between(shot_values, min_values, max_values, color=color, alpha=.4)
    lower, mean, upper = q1_values[-1], np.mean(adaptation_error_values, axis=1)[-1], q3_values[-1]
    print(f'model {model_name}, ({mean - lower:.2e}, {mean:.2e}, {upper-mean:.2e})')

plt.yscale('log')
plt.xlabel('shots')
plt.ylabel('test error')
# plt.ylim((1e-2, 1e1))
plt.show()