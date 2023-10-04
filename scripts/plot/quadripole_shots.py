import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import torch
import torch.nn as nn

from systems import Quadrupole
from meta_training import test_model
from scripts.train.quadrupole import metamodel_choice
from scripts.plot.layout import color_choice

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

np.random.seed(10)
torch.manual_seed(10)

system = Quadrupole()
test_dataset = system.generate_test_data()
T_test = len(test_dataset)
max_shots = 20
adaptation_indices = np.random.randint(len(system.grid), size=max_shots)
# adaptation_indices = np.arange(400)



loss_function = nn.MSELoss()
shot_values = np.arange(1, max_shots)
n_gradient = 5000


results = {}
# for model_index, model_name in enumerate(['maml', 'tldr']):
for model_index, (model_name, metamodel) in enumerate(metamodel_choice.items()):
    if model_name == 'coda':
        continue
    print(f'model {model_name}')
    path = f'output/models/quadrupole/{model_name}_{n_gradient}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)
    adaptation_error_values = []
    for shot_index, shots in enumerate(shot_values):
        shot_indices = adaptation_indices[:shots]
        adaptation_error = test_model(metamodel, test_dataset, shot_indices, n_steps=10)
        adaptation_error_values.append(adaptation_error.copy())
    results[model_name] = adaptation_error_values
    

fig = plt.figure(figsize=(3, 2.5))
# fig = plt.figure()
fig.set_tight_layout(True)

for model_name, adaptation_error_values in results.items():
    color = color_choice[model_name]
    plt.plot(shot_values, np.median(adaptation_error_values, axis=1), color=color)
    min_values = np.array(adaptation_error_values).min(axis=1)
    max_values = np.array(adaptation_error_values).max(axis=1)
    plt.fill_between(shot_values, min_values, max_values, color=color, alpha=.4)

plt.title('quadrupole')
plt.yscale('log')
plt.xlabel('shots')
plt.ylabel('test error')
# plt.ylim((1e-2, 1e1))
# plt.savefig(f'output/plots/dipole_shots-adaptation.pdf')
plt.show()
    


