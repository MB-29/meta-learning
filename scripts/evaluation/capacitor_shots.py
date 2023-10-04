import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import rc

from systems import Capacitor
from capacitor import metamodel_choice, CAMEL, V_net, r
from scripts.plot.layout import color_choice, title_choice
from meta_training import test_model, loss_function

from interpret import estimate_context_transform

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

# np.random.seed(5)
# torch.manual_seed(5)

data_path = 'data/capacitor/epsilon_1'
system = Capacitor(data_path)
meta_dataset = system.generate_training_data()
test_dataset = system.generate_test_data()
# test_dataset = meta_dataset[:2]
T_test = len(test_dataset)
max_shots = 42
shot_values = np.arange(1, max_shots)
adaptation_indices = np.random.randint(len(system.grid), size=max_shots)
adaptation_indices = np.concatenate([k*50 +np.arange(7)*9_000 for k in range(6)])[:max_shots]
adaptation_indices = np.concatenate([20 + k*50 +np.arange(7)*9_000 for k in range(6)])
pace_values = [40, 18, 10, 5, 3, 2, 1]
pace_values = [1]
shot_values = [len(adaptation_indices[::pace]) for pace in pace_values]
# grid_indices = np.argwhere((system.grid_x1>-3) & (system.grid_x1<3) & (system.grid_x2>0) & (system.grid_x2<2))
# adaptation_indices = np.ravel_multi_index((grid_indices[0], grid_indices[1]), (200, 300))
# n_gradient = 10_000



# fig = plt.figure()@
# fig.set_tight_layout(True)
model_names = ['tldr']
# model_names = ['tldr', 'coda', 'maml', 'anil']
model_names = ['tldr_10000', 'coda_3000']
# model_names = ['tldr_1500', 'coda_3000']
model_names = ['tldr_6000', 'coda_3000']
model_names = ['tldr_10000', 'coda_10000']
model_names = ['tldr_r_5', 'coda_r_5']
# model_names = ['tldr_4000', 'coda_4000', 'anil']
model_names = ['tldr', 'coda', 'anil']

# model_names = ['tldr', 'anil', 'coda']
n_gradient = 2_000
n_gradient = 3_000
n_gradient = 10_000


results = {}
# for model_index, model_name in enumerate(['maml', 'tldr']):
for model_index, name in enumerate(model_names):
    architecture = name.split('_')[0]
    metamodel = metamodel_choice[architecture]
    print(f'model {name}')
    path = f'output/models/capacitor/epsilon_1/{name}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)
    adaptation_error_values = []    
    for pace in pace_values:
        shot_indices = adaptation_indices[::pace]
        adaptation_error = test_model(metamodel, test_dataset, shot_indices, n_steps=20)
        adaptation_error_values.append(adaptation_error.copy())
    results[architecture] = adaptation_error_values

    context_values = metamodel.get_context_values(meta_dataset, n_steps=50).detach().numpy()
    supervised_contexts = system.W_train
    learned_contexts = context_values
    weight_estimator = estimate_context_transform(supervised_contexts, learned_contexts, affine=True)

    zero_adaptation_error_values = []
    for pace in pace_values:
        task_zero_adaptation_error_values = []
        for task_index, w in enumerate(system.W_test):
            w_hat = weight_estimator.T @ np.append(w, 1.0)
            zero_model = metamodel.create_task_model(torch.tensor(w_hat).float())
            targets = test_dataset[task_index][1]
            predictions = zero_model(system.grid)
            task_zero_adaptation_error_values.append(loss_function(predictions.squeeze(), targets.squeeze()).item())
        zero_adaptation_error_values.append(np.array(task_zero_adaptation_error_values))
    results[f'{architecture}_zero'] = zero_adaptation_error_values


# fig = plt.figure(figsize=(3, 2.5))
fig = plt.figure()
fig.set_tight_layout(True)  

for model_name, adaptation_error_values in results.items():
    color = color_choice[model_name.split('_')[0]]
    # plt.plot(shot_values, adaptation_error_values, color=color)
    plt.plot(shot_values, np.mean(adaptation_error_values, axis=1), color=color)
    plt.plot(shot_values, np.median(adaptation_error_values, axis=1), color=color, ls='--')
    q1_values = np.quantile(adaptation_error_values, 0.25, axis=1)
    q3_values = np.quantile(adaptation_error_values, 0.75, axis=1)
    plt.fill_between(shot_values, q1_values, q3_values, color=color, alpha=.4)
    lower, mean, upper = q1_values[-1], np.mean(adaptation_error_values, axis=1)[-1], q3_values[-1]
    print(f'model {model_name}, ({mean - lower:.2e}, {mean:.2e}, {upper-mean:.2e})')

plt.yscale('log')
plt.xlabel('shots')
plt.ylabel('test error')
# plt.ylim((1e-2, 1e1))
plt.show()
    
