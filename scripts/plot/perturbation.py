import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from capacitor import metamodel_choice
from systems import Capacitor
from scripts.plot.layout import title_choice, color_choice

np.random.seed(5)
torch.manual_seed(5)

fig = plt.figure(figsize=(5.5, 3.5))
# fig.set_tight_layout(True)
epsilon_values = [0.1, 0.2, 0.5, 1.0]
epsilon_values = [0.1, 0.5, 1.0]
task_index = 0
n_epsilon = 3
model_names = ['tldr']
# model_names = ['tldr', 'coda']
n_models = len(model_names)
data_path = f'data/capacitor'
epsilon_index = 0
for file_name in sorted(os.listdir(data_path)):
    epsilon_data_path = f'{data_path}/{file_name}'
    if not os.path.isdir(epsilon_data_path) or file_name.split('_')[0] != 'epsilon':
        continue
    if file_name.split('_')[1] == '02':
        continue
    epsilon = epsilon_values[epsilon_index]
    epsilon_index += 1
    print(f'data path: {epsilon_data_path}')

    system = Capacitor(epsilon_data_path)
    meta_dataset = system.generate_training_data()
    test_dataset = system.generate_test_data()
    # test_dataset = meta_dataset[:2]
    T_test = len(test_dataset)
    n_shots = 50


    adaptation_indices = np.random.choice(len(system.grid), size=n_shots)
    # adaptation_indices = np.arange(300, len(system.grid), 1000)
    adaptation_points = system.grid[adaptation_indices]

    w = system.W_test[task_index]
    plt.subplot(n_models + 2, n_epsilon, epsilon_index)
    potential_values = test_dataset[task_index][1]
    system.plot_potential_values(potential_values)
    # plt.ylabel(r'$y$', rotation=0)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    # plt.title('target')

    # plt.scatter(*adaptation_points.T, color="red", marker='x')

    for model_index, name in enumerate(model_names):

        metamodel = metamodel_choice[name]
        path = f'output/models/capacitor/{file_name}/{name}.ckpt'
        checkpoint = torch.load(path)
        metamodel.load_state_dict(checkpoint)

        plt.subplot(n_models + 2, n_epsilon, epsilon_index  + (model_index + 1) * n_epsilon)
        title = name
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
        # plt.text(-1.3, -2.7, fr'$\varepsilon={epsilon}$')

        system.plot_potential(adapted_model)
        # system.plot_field(adapted_model, color='black', density=.6)
        # if task_index == 0:
        # plt.title(title)
# plt.text(0.01, 0.8, fr'target', transform=plt.gcf().transFigure)
# plt.text(0.04, 0.5, fr'CAMEL', transform=plt.gcf().transFigure)
plt.subplot(3, 3, 1)
plt.ylabel('target')
plt.subplot(3, 3, 4)
plt.ylabel('CAMEL')

results = {
'tldr': {'mean': 0.01*np.array([1., 3, 10, 21]), 'err': 0.01*np.array([.5, 1, 3, 10])},
'coda': {'mean': 0.01*np.array([11., 9, 10, 20]), 'err': 0.01*np.array([1, 2, 2, 6])},
'anil': {'mean': 0.01*np.array([17., 15, 20, 20]), 'err': 0.01*np.array([3, 5, 7, 9])},
}
results = {
'anil': {'mean': 0.01*np.array([17.,  20, 20]), 'err': 0.01*np.array([3, 7, 9])},
'tldr': {'mean': 0.01*np.array([1., 3, 21]), 'err': 0.01*np.array([.5, 3, 10])},
'coda': {'mean': 0.01*np.array([11., 10, 20]), 'err': 0.01*np.array([2, 3, 6])},
}
x = [1/5, 2/5, 3/5, 4/5]
x = [1/4, 1/2, 3/4]
plt.subplot(4, 1, 4)
plt.grid(axis='y', which='major', alpha=.7)
# plt.grid(axis='y', which='minor', ls='--', alpha=.1)
for architecture, values in results.items():
    mean = values['mean']
    err = values['err']
    bottom = mean-err/2
    color = color_choice[architecture]
    # plt.scatter(x, bottom+err, alpha=.9, color=color)
    # alpha = .9 if 
    label = title_choice[architecture]
    plt.bar(x, 2*err, bottom=bottom, width=.04, alpha=.9, capstyle='round', edgecolor='black', color=color, label=label)
# plt.scatter([1/5, 2/5, 3/5, 4/5], estimation_values, marker='x')
plt.xlim((0.1, 0.9))
plt.ylim((2e-3, .5))
# plt.xscale('log')
plt.yscale('log')
# plt.xticks([0.1, 0.2, 0.5, 1.0], labels=[0.1, 0.2, 0.5, 1.0])
plt.xticks([1/4, 1/2, 3/4], labels=[r'$0.1$', r'$0.5$', r'$1$'])
plt.xlabel(r'$\varepsilon$')
plt.gca().xaxis.set_label_coords(0.95, -0.025)
plt.title('identification error')
plt.plot((.903), (2e-3), ls="", marker=">", ms=4, color="k", clip_on=False)
fig.subplots_adjust(wspace=0.1, hspace=0.2)

# plt.yticks([])
# plt.legend()
plt.savefig('output/plots/perturbation.pdf')
plt.show()

