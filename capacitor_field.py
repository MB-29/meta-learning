import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import rc

from systems import Capacitor
from capacitor import metamodel_choice, CAMEL, V_net, c_net, r
from scripts.plot.layout import color_choice, title_choice
from interpret import estimate_context_transform

np.random.seed(5)
torch.manual_seed(5)

data_path = 'data/capacitor/epsilon_1'
system = Capacitor(data_path)
meta_dataset = system.generate_training_data()
test_dataset = system.generate_test_data()
task_indices = np.arange(len(test_dataset))
# task_indices = np.array([0, 2])
task_indices = np.array([2, 4])
# test_dataset = system.generate_test_data()
# test_dataset = meta_dataset[:2]
T_test = len(task_indices)
n_shots = 30



# fig = plt.figure(figsize=(5, 3.5))
fig = plt.figure(figsize=(12, 7))
fig.set_tight_layout(True)
# fig.subplots_adjust(wspace=0.1, hspace=0.1)

# model_names = ['tldr', 'anil']
# model_names = ['tldr', 'anil', 'coda']
# model_names = ['anil']
model_names = ['tldr_1500', 'tldr_3000']
# model_names = ['tldr', 'anil']
model_names = ['tldr_6000', 'coda_3000']
model_names = ['tldr_10000', 'coda_10000', 'anil']
model_names = ['anil', 'coda', 'tldr']
# model_names = ['coda', 'tldr']

# test_dataset = test_dataset[3:]
# test_dataset = meta_dataset[-2:]
n_rows, n_cols = T_test, len(model_names)+2


# for task_index in range(len(test_dataset)):
for row_index in range(T_test):
    grid_indices = np.argwhere((system.grid_x1>-3) & (system.grid_x1<3) & (system.grid_x2>0) & (system.grid_x2<2))
    indices = np.ravel_multi_index((grid_indices[0], grid_indices[1]), (200, 300))
    adaptation_indices = np.random.choice(indices, size=n_shots)
    adaptation_indices = np.random.choice(len(system.grid), size=n_shots)
    # adaptation_indices = 100+np.arange(7)*9_000
    adaptation_indices = np.concatenate([20 + k*50 +np.arange(7)*9_000 for k in range(6)])[::1]
    # adaptation_indices = np.arange(300, len(system.grid), 1000)
    adaptation_points = system.grid[adaptation_indices]

    task_index = task_indices[row_index]


    w = system.W_test[task_index]
    plt.subplot(n_rows, n_cols, row_index*n_cols+1)
    potential_values = test_dataset[task_index][1]
    system.plot_potential_values(potential_values)
    plt.ylabel(r'$x_2$', rotation=0)
    plt.gca().yaxis.set_label_coords(-0.15,.52)
    ax = plt.gca()
    if row_index == 0:
        plt.title('target')
    if row_index == n_rows-1:
        plt.xlabel(r'$x_1$')  
    plt.xticks([])
    plt.yticks([])

    # plt.scatter(*adaptation_points.T, color="red", marker='x')

    for model_index, name in enumerate(model_names):
        architecture = name.split('_')[0]
        metamodel = metamodel_choice[architecture]
        path = f'output/models/capacitor/epsilon_1/{name}.ckpt'
        checkpoint = torch.load(path)
        metamodel.load_state_dict(checkpoint)

        plt.subplot(n_rows, n_cols, model_index+2 + row_index*n_cols)
        title = title_choice[architecture]
        w = system.W_test[task_index]
        # w = meta_model.W[task_index]
        # plt.title(fr'$U = {w[0]:.1f}$, $p = {w[1]:.1f}$')
        task_test_data = test_dataset[task_index]
        test_points, test_targets = task_test_data
        adaptation_dataset = (test_points[adaptation_indices], test_targets[adaptation_indices])
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=50)
        # plt.yticks([0.5, 1.])
        plt.xticks([])
        plt.yticks([])

        system.plot_potential(adapted_model)
        # system.plot_field(adapted_model, color='black')
        # if task_index == 0:
        if row_index == 0:
            # plt.xlabel(r'$x$')  
            plt.title(title)
        if row_index == n_rows-1:
            plt.xlabel(r'$x_1$')  
# plt.savefig(f'output/plots/capacitor_field.pdf')

    context_values = metamodel.get_context_values(meta_dataset, n_steps=50).detach().numpy()
    supervised_contexts = system.W_train
    learned_contexts = context_values
    weight_estimator = estimate_context_transform(supervised_contexts, learned_contexts, affine=True)
    w_hat = weight_estimator.T @ np.append(w, 1.0)
    zero_model = metamodel.create_task_model(torch.tensor(w_hat).float())
    plt.subplot(T_test, n_cols, n_cols*(row_index+1))
    system.plot_potential(zero_model)
    if row_index == 0:
        plt.title(r'$\varphi$-CAMEL')  
    if row_index == n_rows-1:
        plt.xlabel(r'$x_1$')  
    plt.xticks([])
    plt.yticks([])
    print(f'w_hat = {w_hat}')
    print(f'w_star = {system.W_test[task_index]}')
plt.savefig('output/plots/capacitor_field_full.pdf')
# plt.savefig('output/plots/capacitor_field.pdf')
plt.show()

