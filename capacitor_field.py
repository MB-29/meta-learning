import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import rc

from systems import Capacitor
from capacitor import metamodel_choice, TLDR, V_net, c_net, r
from scripts.plot.layout import color_choice, title_choice

np.random.seed(5)
torch.manual_seed(5)

data_path = 'data/capacitor/epsilon_1'
system = Capacitor(data_path)
meta_dataset = system.generate_training_data()
test_dataset = system.generate_test_data()
# test_dataset = meta_dataset[:2]
T_test = len(test_dataset)
n_shots = 30
# n_gradient = 1_500
# n_gradient = 5_000
# n_gradient = 10_000
# n_gradient = 3_000
# n_gradient = 5_000



# fig = plt.figure()
# fig.set_tight_layout(True)
# model_names = ['tldr', 'anil']
# model_names = ['tldr', 'anil', 'coda']
# model_names = ['anil']
model_names = ['tldr_1500', 'tldr_3000']
# model_names = ['tldr', 'anil']
# model_names = ['tldr', 'coda', 'anil']
model_names = ['tldr_6000', 'coda_3000']
model_names = ['tldr_10000', 'coda_10000', 'anil']

# test_dataset = test_dataset[3:]
# test_dataset = meta_dataset[-2:]
n_rows, n_cols = len(test_dataset), len(model_names)+1


for task_index in range(len(test_dataset)):
    grid_indices = np.argwhere((system.grid_x1>-3) & (system.grid_x1<3) & (system.grid_x2>0) & (system.grid_x2<2))
    indices = np.ravel_multi_index((grid_indices[0], grid_indices[1]), (200, 300))
    adaptation_indices = np.random.choice(indices, size=n_shots)
    adaptation_indices = np.random.choice(len(system.grid), size=n_shots)
    # adaptation_indices = 100+np.arange(7)*9_000
    adaptation_indices = np.concatenate([20 + k*50 +np.arange(7)*9_000 for k in range(6)])[::2]
    # adaptation_indices = np.arange(300, len(system.grid), 1000)
    adaptation_points = system.grid[adaptation_indices]

    w = system.W_test[task_index]
    plt.subplot(n_rows, n_cols, task_index*n_cols+1)
    potential_values = test_dataset[task_index][1]
    system.plot_potential_values(potential_values)
    plt.ylabel(r'$y$', rotation=0)
    ax = plt.gca()
    plt.title('ground truth')

    plt.scatter(*adaptation_points.T, color="red", marker='x')

    for model_index, name in enumerate(model_names):
        architecture = name.split('_')[0]
        metamodel = metamodel_choice[architecture]
        path = f'output/models/capacitor/epsilon_1/{name}.ckpt'
        checkpoint = torch.load(path)
        metamodel.load_state_dict(checkpoint)

        plt.subplot(n_rows, n_cols, model_index+2 + task_index*n_cols)
        title = title_choice[architecture]
        w = system.W_test[task_index]
        # w = meta_model.W[task_index]
        # plt.title(fr'$U = {w[0]:.1f}$, $p = {w[1]:.1f}$')
        task_test_data = test_dataset[task_index]
        test_points, test_targets = task_test_data
        adaptation_dataset = (test_points[adaptation_indices], test_targets[adaptation_indices])
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=50)
        # plt.yticks([0.5, 1.])
        plt.yticks([])

        system.plot_potential(adapted_model)
        # system.plot_field(adapted_model, color='black')
        # if task_index == 0:
        plt.title(title)
        # if task_index == 1:
            # plt.xlabel(r'$x$')  
# plt.savefig(f'output/plots/capacitor_field.pdf')

        # context_values = metamodel.get_context(meta_dataset, n_steps=50)
        # # T_train = system.T
        # # tldr = TLDR(T_train, r, V_net, c_net, W=context_values)
        # # context_estimator = tldr.estimate_context_transform(system.W_train)
        # context_estimator = estimate_context_transform(context_values, system.W_train)
        # # w = context_values[task_index].detach().numpy()
        # w = adapted_model.get_context()
        # # print(f'w = {w}')
        # w_bar = np.append(w, 1.0)
        # w_hat = context_estimator.T @ w_bar
        # # V_hat = tldr.calibrate(system.W_train)
        # # adapted_model = tldr.adapt_task_model(adaptation_dataset)
        # # print(f'adapted w = {adapted_model.w}')
        # print(f'w_hat = {w_hat}')
        # print(f'w_star = {system.W_test[task_index]}')
plt.show()

