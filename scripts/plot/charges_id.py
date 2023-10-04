import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import rc

from systems import Charges
from charges_train import metamodel_choice, CAMEL, V_net, r
from scripts.plot.layout import color_choice, title_choice
from meta_training import test_model
from interpret import estimate_context_transform


np.random.seed(10)
torch.manual_seed(10)
system = Charges()
sigma = 1e-5
# sigma = 0

W_test = system.W_test[:1]
W_test = np.array([
    [2., 4.5]
])
W_test = 5*np.random.random((8, 2)) + 1.
n_tasks = W_test.shape[0]

meta_dataset = system.generate_training_data()
test_dataset = system.generate_data(W_test, sigma=sigma)
# test_dataset = meta_dataset[:2]
T_test = len(test_dataset)
adaptation_indices = (1 + 12*np.arange(30))
# adaptation_indices = np.arange(len(system.grid))


affine = False
model_names = ['tldr']
# model_names = ['tldr', 'coda', 'maml', 'anil']
# model_names = ['tldr_10000', 'coda_3000']
model_names = ['tldr_10000', 'coda_3000']
# model_names = ['tldr_5000', 'coda_3000']
model_names = ['tldr_10000', 'coda_3000']
model_names = ['tldr_6000', 'coda_3000']
model_names = ['tldr', 'coda', 'anil']
# model_names = ['tldr', 'coda']
# model_names = ['tldr']

# model_names = ['tldr_10000']
# model_names = ['tldr', 'anil', 'coda']

W_train = np.array(system.W_train, dtype=np.float)
W_train += 1e-2*np.random.randn(*W_train.shape)
supervision_values = np.arange(1, system.T)
supervision_values = np.logspace(0, 2, 10, dtype=int)
# supervision_values = [1, 3, 8, 15, 35, 50, 80, 99]
results = {}
# for model_index, model_name in enumerate(['maml', 'tldr']):
for model_index, name in enumerate(model_names):
    architecture = name.split('_')[0]
    metamodel = metamodel_choice[architecture]
    print(f'model {name}')
    path = f'output/models/charges/{name}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)
    context_error_values = []
    context_values = metamodel.get_context_values(meta_dataset, n_steps=50).detach().numpy()
    for n_supervision in supervision_values:
        context_error = np.zeros(n_tasks)
        supervised_contexts = W_train[:n_supervision]
        learned_contexts = context_values[:n_supervision]
        context_estimator = estimate_context_transform(learned_contexts, supervised_contexts, affine=affine)
        for task_index in range(n_tasks):
            task_test_data = test_dataset[task_index]
            test_points, test_targets = task_test_data
            # plt.scatter(*test_points[shot_indices].T, color="red", marker='x')
            # plt.pause(0.1)
            # plt.close()
            adaptation_dataset = (test_points[adaptation_indices], test_targets[adaptation_indices])
            adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=5)
            
            # w = context_values[task_index].detach().numpy()
            w = adapted_model.get_context()
            # print(w)
            # print(f'w = {w}')
            w_bar = w.squeeze().numpy()
            if affine:
                w_bar = np.append(w, 1.0)
            w_hat = context_estimator.T @ w_bar
            w_star = W_test[task_index]
            error = np.linalg.norm(w_hat-w_star)
            mape = np.mean(np.abs(w_hat - w_star) / np.abs(w_star))
            # mape = np.mean(np.abs(w_hat - w_star) / 0.1)
            context_error[task_index] = mape
        context_error_values.append(context_error)
    results[architecture] = context_error_values


fig = plt.figure(figsize=(3., 2.2))
# fig = plt.figure()
fig.set_tight_layout(True)

for model_name, context_error_values in results.items():
    color = color_choice[model_name]
    # plt.plot(supervision_values, context_error_values, color=color)
    # plt.scatter(supervision_values, np.median(context_error_values, axis=1), color=color)
    # min_values = np.array(context_error_values).min(axis=1)
    # max_values = np.array(context_error_values).max(axis=1)
    label = title_choice[model_name]
    mean_values = np.mean(context_error_values, axis=1)
    median_values = np.median(context_error_values, axis=1)
    q1_values = np.quantile(context_error_values, 0.25, axis=1)
    q2_values = np.quantile(context_error_values, 0.75, axis=1)
    plt.scatter(supervision_values, median_values, color=color, ls='--')
    plt.plot(supervision_values, median_values, color=color, label=label)
    plt.fill_between(supervision_values, q1_values, q2_values, color=color, alpha=.2)

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$T$')
# plt.gca().xaxis.set_label_coords(.7, -0.13)
# plt.ylabel('mean absolute\npercentage error')
# plt.ylabel('identification\nrelative error')
plt.title('identification error')
plt.grid(axis='y', which='major', alpha=.5)
plt.grid(axis='x', which='major', alpha=.5)
# plt.ylabel(r'$ \Vert \varphi - \hat{\varphi} \Vert$')
plt.ylim((3e-3, 2e0))
# plt.legend(loc='right')

plt.savefig(f'output/plots/charges_id.pdf')
plt.show()
    
