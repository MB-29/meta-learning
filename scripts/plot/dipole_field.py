import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from tqdm import tqdm 
from matplotlib import rc

from systems import Dipole
from scripts.train.dipole import metamodel_choice

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

np.random.seed(10)
torch.manual_seed(10)

system = Dipole(sigma=1e-3)
test_dataset = system.generate_test_data()
T_test = len(test_dataset)
n_shots = 2
n_gradient = 5000
adaptation_indices = np.random.randint(400, size=n_shots)
# adaptation_indices = np.arange(400)

color_choice = {'maml': 'black', 'tldr': 'darkred', 'coda': 'darkblue', 'anil': 'purple'}
title_choice = {'maml': r'MAML', 'tldr': r'CAMEL', 'coda': 'darkblue', 'anil': 'purple'}



fig = plt.figure(figsize=(5, 4))
# fig = plt.figure()
fig.set_tight_layout(True)

T_display=2
# plt.suptitle(r'electric dipole moment, 2-shot adaptation at points \qquad'+ ' ')
# fig.text(0.83, 0.96, r'$\times$', color='red', fontsize=15)
for task_index in range(T_display):
    adaptation_indices = np.random.randint(400, size=n_shots)

    w = system.W_test[task_index]
    plt.subplot(T_display, 3, 3*task_index+1)
    potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
    system.plot_field(potential_map, color='black')
    adaptation_points = system.grid[adaptation_indices]
    plt.scatter(*adaptation_points.T, color="red", s=30, marker='x', label='adaptation\npoints')
    plt.yticks([0.1, 1.], labels=[r'$0$', r'$1$'])
    plt.ylabel(r'$y$', rotation=0)
    plt.xticks([-1., 0., 1.])
    ax = plt.gca()
    if task_index == 0:
        plt.title('target')
        # ax.text(-3, 0.5, r'new dipole 1')
        # ax.legend(loc='lower center', bbox_to_anchor=(-.7,0), fancybox=True)
    if task_index == 1:
        # ax.text(-3, 0.5, r'new dipole 2')
        plt.xlabel(r'$x$')
        ax = plt.gca()
        box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # ax.legend(loc='upper center', bbox_to_anchor=(.5,-.3), fancybox=True)


    for model_index, model_name in enumerate(['tldr', 'maml']):

        metamodel = metamodel_choice[model_name]
        path = f'output/models/dipole/{model_name}_{n_gradient}.ckpt'
        checkpoint = torch.load(path)
        metamodel.load_state_dict(checkpoint)


        plt.subplot(T_display, 3, 3*task_index + 2 + model_index)
        title = title_choice[model_name]
        w = system.W_test[task_index]
        # w = meta_model.W[task_index]
        # plt.title(fr'$U = {w[0]:.1f}$, $p = {w[1]:.1f}$')
        task_test_data = test_dataset[task_index]
        test_points, test_targets = task_test_data
        adaptation_dataset = (test_points[adaptation_indices], test_targets[adaptation_indices])
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=100)
        plt.yticks([0.5, 1.])
        plt.yticks([])

        system.plot_field(adapted_model, color='black')
        if task_index == 0:
            plt.title(title)
        if task_index == 1:
            plt.xlabel(r'$x$')  
            # plt.legend(loc='lower center')
# plt.gcf().text(0.6, .05, r'2-shot adaptation', fontsize=15, bbox=dict(facecolor='none', edgecolor='lightgrey'))
# plt.gcf().text(-0.01, .5, r'dipole 1', fontsize=15,)
# for task_index in range(T_display):
    # system.plot_potential(potential_map)
plt.savefig(f'output/plots/dipole_field.pdf')
plt.show()

