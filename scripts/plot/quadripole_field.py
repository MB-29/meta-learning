import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import rc

from systems import Quadrupole
from scripts.train.quadrupole import metamodel_choice
from scripts.plot.layout import title_choice
from interpret import estimate_context_transform

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

np.random.seed(1)
torch.manual_seed(2)
sigma = 1e-5
system = Quadrupole(sigma=sigma)
meta_dataset = system.generate_training_data()
test_dataset = system.generate_test_data()
T_test = len(test_dataset)
n_shots = 10
n_gradient = 5000



fig = plt.figure(figsize=(10, 5))
# fig = plt.figure()
fig.set_tight_layout(True)

adaptation_indices = np.random.randint(len(system.grid), size=n_shots)
adaptation_indices = np.concatenate([20 + k*50 +np.arange(7)*1_000 for k in range(6)])[::10]

n_cols = 5

T_display = system.T_test
# T_display = 1
for task_index in range(T_display):

    w = system.W_test[task_index]
    plt.subplot(T_display, n_cols, n_cols*task_index+1)
    potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
    system.plot_field(potential_map)
    adaptation_points = system.grid[adaptation_indices]
    plt.ylabel(r'$x_2$', rotation=0)
    plt.gca().yaxis.set_label_coords(-0.1,.52)

    plt.yticks([-0.5, 0, 0.5])
    plt.scatter(*adaptation_points[:, :2].T, color="red", s=30, marker='x')
    ax = plt.gca()
    if task_index == 0:
        plt.title('target')
    if task_index == T_display-1:
        plt.xlabel(r'$x_1$')
        ax = plt.gca()
        box = ax.get_position()

    plt.xticks([])
    plt.yticks([])
    # for model_index, model_name in enumerate(['anil', 'tldr']):
    for model_index, model_name in enumerate(['anil', 'maml', 'tldr']):


        metamodel = metamodel_choice[model_name]
        path = f'output/models/quadrupole/{model_name}_{n_gradient}.ckpt'
        checkpoint = torch.load(path)
        metamodel.load_state_dict(checkpoint)


        plt.subplot(T_display, n_cols, n_cols*task_index + 2 + model_index)
        title = title_choice[model_name]
        w = system.W_test[task_index]   
        # w = meta_model.W[task_index]
        # plt.title(fr'$U = {w[0]:.1f}$, $p = {w[1]:.1f}$')
        task_test_data = test_dataset[task_index]
        test_points, test_targets = task_test_data
        adaptation_dataset = (test_points[adaptation_indices], test_targets[adaptation_indices])
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=50)
        # plt.yticks([0.5, 1.])
        plt.yticks([])
        plt.xticks([])

        system.plot_field(adapted_model)
        if task_index == 0:
            plt.title(title)
        if task_index == T_display-1:
            plt.xlabel(r'$x_1$')  
    
    context_values = metamodel.get_context_values(meta_dataset, n_steps=50).detach().numpy()
    supervised_contexts = system.W_train
    learned_contexts = context_values
    weight_estimator = estimate_context_transform(supervised_contexts, learned_contexts, affine=True)
    w_hat = weight_estimator.T @ np.append(w, 1.0)
    zero_model = metamodel.create_task_model(torch.tensor(w_hat).float())
    plt.subplot(T_test, n_cols, n_cols*(task_index+1))
    system.plot_field(zero_model)
    if task_index == 0:
        plt.title(r'$\varphi$-CAMEL')  
    if task_index == 2-1:
        plt.xlabel(r'$x_1$')  
    plt.xticks([])
    plt.yticks([])
plt.savefig(f'output/plots/quadrupole_field.pdf')
plt.show()

