import numpy as np
import matplotlib.pyplot as plt


import torch

from systems import Charges
from charges_train import metamodel_choice
# from scripts.train.arm import metamodel_choice
from scripts.plot.layout import color_choice
from meta_training import test_model, loss_function

from interpret import estimate_context_transform

np.random.seed(5)
torch.manual_seed(5)

system = Charges()
# system = ActuatedArm()

n_test = 30
W_test = 5*np.random.random((30, 2)) + 1
# W_test = (2*np.random.random((n_test, 2)) - 1)*np.array([.15, .1]) + np.array([.3, 1])

meta_dataset = system.generate_training_data()
test_meta_dataset = system.generate_data(W_test)

adaptation_indices = (1 + 12*np.arange(30))
pace_values = [13, 9, 6, 3, 2, 1]
# pace_values = [13, 9, 6]
shot_values = [len(adaptation_indices[::pace]) for pace in pace_values]
results = {}
# for model_index, metamodel_name in enumerate(['tldr']):
# for model_index, metamodel_name in enumerate(['tldr', 'coda', 'anil', 'maml']):
for model_index, metamodel_name in enumerate(['tldr', 'coda', 'anil']):
    adaptation_error_values = []
    print(f'model {metamodel_name}')
# for model_index, metamodel_name in enumerate(['tldr', 'anil', 'maml', 'coda']):
    architecture = metamodel_name.split('_')[0]
    metamodel = metamodel_choice[metamodel_name]
    path = f'output/models/charges/{metamodel_name}.ckpt'
    # path = f'output/models/arm/{metamodel_name}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)
    for pace in pace_values:
        shot_indices = adaptation_indices[::pace]
        adaptation_error = test_model(metamodel, test_meta_dataset, shot_indices, n_steps=100)
        adaptation_error_values.append(adaptation_error.copy()/50)
    results[architecture] = adaptation_error_values

    context_values = metamodel.get_context_values(meta_dataset, n_steps=50).detach().numpy()
    supervised_contexts = system.W_train
    learned_contexts = context_values
    weight_estimator = estimate_context_transform(supervised_contexts, learned_contexts, affine=True)

    zero_adaptation_error_values = []
    for pace in pace_values:
        task_zero_adaptation_error_values = []
        for task_index, w in enumerate(W_test):
            w_hat = weight_estimator.T @ np.append(w, 1.0)
            zero_model = metamodel.create_task_model(torch.tensor(w_hat).float())
            targets = test_meta_dataset[task_index][1]
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
    # plt.plot(shot_values, np.median(adaptation_error_values, axis=1), color=color, ls='--')
    q3_values = np.quantile(adaptation_error_values, 0.75, axis=1)
    q1_values = np.quantile(adaptation_error_values, 0.25, axis=1)
    plt.fill_between(shot_values, q1_values, q3_values, color=color, alpha=.4)

    lower, mean, upper = q1_values[-1], np.mean(adaptation_error_values, axis=1)[-1], q3_values[-1]
    print(f'model {model_name}, ({mean - lower:.2e}, {mean:.2e}, {upper-mean:.2e})')

plt.yscale('log')
plt.xlabel('shots')
plt.ylabel('test error')
# plt.ylim((1e-2, 1e1))
plt.show()