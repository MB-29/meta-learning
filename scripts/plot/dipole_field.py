import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm 
from matplotlib import rc

from systems import Dipole
from scripts.train.dipole import metamodel_choice
from interpret import estimate_context_transform

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

np.random.seed(10)
torch.manual_seed(10)

system = Dipole()
meta_dataset = system.generate_training_data()

# W_test = system.W_test
# test_dataset = system.generate_test_data(sigma=1e-1)

W_test = np.array([
    [1.5, 0.7],
    [6., 0.6]
])
test_dataset = system.generate_data(W_test, grid=system.grid, sigma=1e-1)
T_test = len(test_dataset)
n_shots = 10
n_gradient = 5000
adaptation_indices = np.random.randint(400, size=n_shots)
# adaptation_indices = np.arange(400)

color_choice = {'maml': 'black', 'tldr': 'darkred', 'coda': 'darkblue', 'anil': 'purple'}
title_choice = {'maml': r'ANIL', 'tldr': r'CAMEL', 'coda': 'darkblue', 'anil': 'ANIL'}




# fig = plt.figure(figsize=(5, 4))
fig = plt.figure(figsize=(5, 3.5))
# fig = plt.figure()
# fig.set_tight_layout(True)
fig.subplots_adjust(wspace=0.1, hspace=0.1)

T_display = 2
# T_display = 1
n_cols = 4
# plt.suptitle(r'electric dipole moment, 2-shot adaptation at points \qquad'+ ' ')
# fig.text(0.83, 0.96, r'$\times$', color='red', fontsize=15)
for task_index in range(T_display):
    adaptation_indices = np.random.randint(400, size=n_shots)
    adaptation_indices = np.array([135, 265])

    w = W_test[task_index]
    plt.subplot(T_display, n_cols, n_cols*task_index+1)
    potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
    system.plot_field(potential_map)
    adaptation_points = system.grid[adaptation_indices]
    # adaptation_points = torch.rand(n_shots, 2) + torch.tensor([0, 0.2])
    plt.scatter(*adaptation_points.T, color="red", s=30, marker='x', label='adaptation\npoints')
    # plt.yticks([0.1, 1.], labels=[r'$0$', r'$1$'])
    plt.yticks([])
    plt.ylabel(r'$x_2$', rotation=0)
    plt.gca().yaxis.set_label_coords(-0.15,.2)

    # plt.xticks([-1., 0., 1.])
    ax = plt.gca()
    if task_index == 0: 
        plt.title('target')
        # ax.text(-3, 0.5, r'new dipole 1')
        # ax.legend(loc='lower center', bbox_to_anchor=(-.7,0), fancybox=True)
        # plt.xticks([])
    if task_index == T_display-1:
        # ax.text(-3, 0.5, r'new dipole 2')
        plt.xlabel(r'$x_1$')
        ax = plt.gca()
        box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # ax.legend(loc='upper center', bbox_to_anchor=(.5,-.3), fancybox=True)
    plt.xticks([])


    # for model_index, model_name in enumerate(['tldr', 'anil']):
    # for model_index, model_name in enumerate(['tldr', 'maml']):
    for model_index, model_name in enumerate(['maml', 'tldr']):

        metamodel = metamodel_choice[model_name]
        path = f'output/models/dipole/{model_name}_{n_gradient}.ckpt'
        checkpoint = torch.load(path)
        metamodel.load_state_dict(checkpoint)


        plt.subplot(T_display, n_cols, n_cols*task_index + 2 + model_index)
        title = title_choice[model_name]
        # w = meta_model.W[task_index]
        # plt.title(fr'$U = {w[0]:.1f}$, $p = {w[1]:.1f}$')
        task_test_data = test_dataset[task_index]
        test_points, test_targets = task_test_data
        adaptation_dataset = (test_points[adaptation_indices], test_targets[adaptation_indices])
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=10)
        # plt.yticks([0.5, 1.])
        plt.yticks([])

        system.plot_field(adapted_model)
        if task_index == 0:
            plt.title(title)

        if task_index == T_display-1:
            plt.xlabel(r'$x_1$')  
        plt.xticks([])
            # plt.legend(loc='lower center')
    context_values = metamodel.get_context_values(meta_dataset, n_steps=50).detach().numpy()
    supervised_contexts = system.W_train
    learned_contexts = context_values
    weight_estimator = estimate_context_transform(supervised_contexts, learned_contexts, affine=True)
    w_hat = weight_estimator.T @ np.append(w, 1.0)
    zero_model = metamodel.create_task_model(torch.tensor(w_hat).float())
    plt.subplot(T_display, 4, 4*(task_index+1))
    system.plot_field(zero_model)
    if task_index == 0:
        plt.title(r'$\varphi$-CAMEL')  
    if task_index == T_display-1:
        plt.xlabel(r'$x_1$')  
    plt.xticks([])
    plt.yticks([])

# plt.gcf().text(0.6, .05, r'2-shot adaptation', fontsize=15, bbox=dict(facecolor='none', edgecolor='lightgrey'))
# plt.gcf().text(-0.01, .5, r'dipole 1', fontsize=15,)
# for task_index in range(T_display):
    # system.plot_potential(potential_map)
# plt.savefig(f'output/plots/dipole_field_1-task.pdf')
plt.savefig(f'output/plots/dipole_field.pdf')
plt.show()

