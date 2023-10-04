import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import rc

from systems import Charges
from charges_train import metamodel_choice
from scripts.plot.layout import color_choice, title_choice

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

np.random.seed(1)
torch.manual_seed(2)
sigma = 1e-5
system = Charges()
test_dataset = system.generate_test_data(sigma=sigma)
T_test = len(test_dataset)
n_shots = 50
n_gradient = 5000




fig = plt.figure(figsize=(5, 4))
# fig = plt.figure()
fig.set_tight_layout(True)
adaptation_indices = np.random.randint(len(system.grid), size=n_shots)
# adaptation_indices = (1 + 12*np.arange(30))[::13]
adaptation_indices = (1 + 12*np.arange(30))[::1]

T_display = system.T_test
for task_index in range(T_display):

    w = system.W_test[task_index]
    plt.subplot(T_display, 4, 4*task_index+1)
    potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
    system.plot_field(potential_map, color='black')
    adaptation_points = system.grid[adaptation_indices]
    plt.scatter(*adaptation_points.T, color='red', marker='x')
    plt.ylabel(r'$y$', rotation=0)
    # plt.yticks([-0.5, 0, 0.5])
    ax = plt.gca()
    if task_index == 0:
        plt.title('target')
    if task_index == 1:
        plt.xlabel(r'$x$')
        ax = plt.gca()
        box = ax.get_position()


    # for model_index, model_name in enumerate(['tldr', 'anil']):
    for model_index, model_name in enumerate(['tldr', 'coda', 'anil']):
    # for model_index, model_name in enumerate([]):

        metamodel = metamodel_choice[model_name]
        path = f'output/models/charges/{model_name}.ckpt'
        checkpoint = torch.load(path)
        metamodel.load_state_dict(checkpoint)


        plt.subplot(T_display, 4, 4*task_index + 2 + model_index)
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

        system.plot_field(adapted_model, color='black')
        if task_index == 0:
            plt.title(title)
        if task_index == 1:
            plt.xlabel(r'$x$')  
plt.show()

