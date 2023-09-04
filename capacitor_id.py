import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import rc

from systems import Capacitor
from capacitor import metamodel_choice, TLDR, V_net, r
from scripts.plot.layout import color_choice, title_choice
from meta_training import test_model
from interpret import estimate_context_transform


# np.random.seed(5)
# torch.manual_seed(5)

data_path = 'data/capacitor/epsilon_01'
system = Capacitor(data_path)
meta_dataset = system.generate_training_data()
test_dataset = system.generate_test_data()
# test_dataset = meta_dataset[:2]
T_test = len(test_dataset)
adaptation_indices = np.concatenate([20 + k*50 +np.arange(7)*9_000 for k in range(6)])
# adaptation_indices = np.arange(len(system.grid))

# grid_indices = np.argwhere((system.grid_x1>-3) & (system.grid_x1<3) & (system.grid_x2>0) & (system.grid_x2<2))
# adaptation_indices = np.ravel_multi_index((grid_indices[0], grid_indices[1]), (200, 300))
# n_gradient = 10_000


affine = False
# affine = True
# fig = plt.figure()
# fig.set_tight_layout(True)
model_names = ['tldr']
# model_names = ['tldr', 'coda', 'maml', 'anil']
# model_names = ['tldr_10000', 'coda_3000']
model_names = ['tldr_10000', 'coda_3000']
# model_names = ['tldr_5000', 'coda_3000']
model_names = ['tldr_10000', 'coda_3000']
model_names = ['tldr_6000', 'coda_3000']
model_names = ['tldr', 'coda', 'anil']
model_names = ['tldr', 'coda']

# model_names = ['tldr_10000']
# model_names = ['tldr', 'anil', 'coda']
n_gradient = 2_000
n_gradient = 3_000

supervision_values = np.arange(1, system.T)
results = {}
# for model_index, model_name in enumerate(['maml', 'tldr']):
for model_index, name in enumerate(model_names):
    architecture = name.split('_')[0]
    metamodel = metamodel_choice[architecture]
    print(f'model {name}')
    path = f'output/models/capacitor/epsilon_01/{name}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)
    context_error_values = []
    context_values = metamodel.get_context_values(meta_dataset, n_steps=50).detach().numpy()
    for n_supervision in supervision_values:
        context_error = np.zeros(system.T_test)
        supervised_contexts = system.W_train[:n_supervision]
        learned_contexts = context_values[:n_supervision]
        context_estimator = estimate_context_transform(learned_contexts, supervised_contexts, affine=affine)
        for task_index in range(system.T_test):
            task_test_data = test_dataset[task_index]
            test_points, test_targets = task_test_data
            # plt.scatter(*test_points[shot_indices].T, color="red", marker='x')
            # plt.pause(0.1)
            # plt.close()
            adaptation_dataset = (test_points[adaptation_indices], test_targets[adaptation_indices])
            adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=50)
            
            # w = context_values[task_index].detach().numpy()
            w = adapted_model.get_context()
            # print(w)
            # print(f'w = {w}')
            w_bar = w.squeeze().numpy()
            if affine:
                w_bar = np.append(w, 1.0)
            w_hat = context_estimator.T @ w_bar
            w_star = system.W_test[task_index]
            error = np.linalg.norm(w_hat-w_star)
            mape = np.mean(np.abs(w_hat - w_star) / np.abs(w_star))
            mape = np.mean(np.abs(w_hat - w_star) / 0.2)
            context_error[task_index] = mape
        context_error_values.append(context_error)
    results[architecture] = context_error_values


# fig = plt.figure(figsize=(3, 2.5))
fig = plt.figure()
fig.set_tight_layout(True)

for model_name, context_error_values in results.items():
    color = color_choice[model_name]
    # plt.plot(supervision_values, context_error_values, color=color)
    plt.plot(supervision_values, np.mean(context_error_values, axis=1), color=color, ls='--')
    plt.plot(supervision_values, np.median(context_error_values, axis=1), color=color)
    min_values = np.array(context_error_values).min(axis=1)
    max_values = np.array(context_error_values).max(axis=1)
    plt.fill_between(supervision_values, min_values, max_values, color=color, alpha=.4)

plt.yscale('log')
plt.xlabel('supervision')
plt.ylabel('parameter error')
# plt.ylim((1e-2, 1e1))
# plt.savefig(f'output/plots/dipole_shots-adaptation.pdf')
plt.show()
    
