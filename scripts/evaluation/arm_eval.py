import numpy as np
import matplotlib.pyplot as plt


from robotics.cartpole import Cartpole
import torch

np.random.seed(5)
torch.manual_seed(5)

from systems import DampedActuatedCartpole, ActuatedArm
from scripts.train.arm import metamodel_choice
from scripts.plot.layout import color_choice
from meta_training import test_model, loss_function

from interpret import estimate_context_transform

system = ActuatedArm()

n_test = 50
# W_test = (np.random.random((n_test, 2)) - 0.5)*np.array([1, .2]) + np.array([2., .3])
W_test = (2*np.random.random((n_test, 2)) - 1)*np.array([.05, .1]) + np.array([.35, 1])
W_test = (2*np.random.random((n_test, 2)) - 1)*np.array([.1, .2]) + np.array([.3, 1.])
# W_test = system.W_test
# W_test = system.W_train 

meta_dataset = system.generate_training_data()
test_meta_dataset = system.generate_data(W_test, sigma=0.0001)
max_shots = 100
shot_values = np.arange(1, max_shots, 5)
shot_values = [5, 10, 30, 50, 100, 200]
shot_values = [100]
# shot_values = [50]
results = {}
# for model_index, metamodel_name in enumerate(['tldr']):
# for model_index, metamodel_name in enumerate(['tldr', 'coda', 'maml', 'anil']):
for model_index, metamodel_name in enumerate(['tldr', 'coda', 'anil']):
    adaptation_error_values = []
# for model_index, metamodel_name in enumerate(['tldr', 'anil', 'maml', 'coda']):
    architecture = metamodel_name.split('_')[0]
    metamodel = metamodel_choice[metamodel_name]
    path = f'output/models/arm/{metamodel_name}.ckpt'
    checkpoint = torch.load(path)
    metamodel.load_state_dict(checkpoint)
    for shots in shot_values:
        adaptation_indices = np.arange(0, shots)
        adaptation_error = test_model(metamodel, test_meta_dataset, adaptation_indices, n_steps=10)
        adaptation_error_values.append(adaptation_error.copy())
        results[architecture] = adaptation_error_values
    
    context_values = metamodel.get_context_values(meta_dataset, n_steps=50).detach().numpy()
    supervised_contexts = system.W_train
    learned_contexts = context_values
    weight_estimator = estimate_context_transform(supervised_contexts, learned_contexts, affine=True)

    zero_adaptation_error_values = []
    task_zero_adaptation_error_values = []
    for task_index, w in enumerate(system.W_test):
        w_hat = weight_estimator.T @ np.append(w, 1.0)
        zero_model = metamodel.create_task_model(torch.tensor(w_hat).float())
        points, targets = test_meta_dataset[task_index]
        predictions = zero_model(points)
        task_zero_adaptation_error_values.append(loss_function(predictions.squeeze(), targets.squeeze()).item())
    zero_adaptation_error_values.append(np.array(task_zero_adaptation_error_values))
    results[f'{architecture}_zero'] = zero_adaptation_error_values


# test_data = test_meta_dataset[5]
# test_points, test_targets = test_data
# adaptation_data = test_points[:100], test_targets[:100]

# model = metamodel.adapt_task_model(adaptation_data)
# predictions = model(test_points).detach().numpy()

# plt.plot(predictions)
# plt.plot(test_targets, ls='--')
# plt.show()



# fig = plt.figure(figsize=(3, 2.5))
fig = plt.figure()
fig.set_tight_layout(True)

for model_name, adaptation_error_values in results.items():
    color = color_choice[model_name.split('_')[0]]
    # plt.plot(shot_values, adaptation_error_values, color=color)
    plt.plot(shot_values, np.mean(adaptation_error_values, axis=1), color=color)
    # plt.plot(shot_values, np.median(adaptation_error_values, axis=1), color=color, ls='--')
    min_values = np.array(adaptation_error_values).min(axis=1)
    max_values = np.array(adaptation_error_values).max(axis=1)
    q1_values = np.quantile(adaptation_error_values, 0.25, axis=1)
    q3_values = np.quantile(adaptation_error_values, 0.75, axis=1)
    plt.fill_between(shot_values, min_values, max_values, color=color, alpha=.4)
    lower, mean, upper = q1_values[-1], np.mean(adaptation_error_values, axis=1)[-1], q3_values[-1]
    print(f'model {model_name}, ({mean - lower:.2e}, {mean:.2e}, {upper-mean:.2e})')

plt.yscale('log')
plt.xlabel('shots')
plt.ylabel('test error')
# plt.ylim((1e-2, 1e1))
plt.show()