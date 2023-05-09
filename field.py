import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm 

from model import TaskLinearMetaModel, TaskLinearModel, MAML
from systems.dipole import Dipole

# np.random.seed(5)
# torch.manual_seed(5)

system = Dipole()
W_train = system.W_train
T_train, r = W_train.shape
W_test = system.W_test
T_test = W_test.shape[0]
V_target = system.generate_V_data()

adaptation_indices = np.random.randint(400, size=100)

# for index in range(system.T):
#     plt.subplot(3, 3, index+1)
#     w = system.W_train[index]
#     plt.title(f'U = {w[0]}, p = {w[1]}')
#     # system.plot_potential(w)
#     potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
#     system.plot_field(potential_map)
# plt.show()

training_data = system.generate_training_data()
test_data = system.generate_test_data()

n_adapt_steps = 1

V_net = torch.nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, r)
)
# c_net = nn.Linear(2, 2)
W_values = torch.abs(torch.randn(T_train, r))
# torch.randn(T, 2, requires_grad=True)
meta_model = TaskLinearMetaModel(V_net, c=None)
training_task_models = meta_model.define_task_models(W_values)

net = torch.nn.Sequential(
    nn.Linear(2, 8),
    nn.Tanh(),
    nn.Linear(8, 2),
    # nn.Tanh(),
    # nn.Linear(16, 2),
    nn.Tanh(),
    nn.Linear(2, 1)
)
meta_model = MAML(net, lr=0.005, first_order=False)



n_gradient = 500
test_interval = n_gradient // 100
W_test_values, V_test_values = [], []
adaptation_error_values = []
loss_values = np.zeros(n_gradient)
optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.005)
# optimizer_weights = torch.optim.Adam(W_values, lr=0.01)
loss_function = nn.MSELoss()
for step in tqdm(range(n_gradient)):
    loss = 0
    for task_index in range(T_train):
        task_targets = torch.tensor(training_data[task_index], dtype=torch.float)
        task_points = system.grid
        model = meta_model.get_training_task_model(task_index, task_points, task_targets)
        predictions = system.predict(model)
        loss += loss_function(predictions, task_targets)

    loss_values[step] = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % test_interval != 0:
        continue
    # print(f'step {step}')

    adaptation_error = np.zeros(T_test)
    adaptation_points = system.grid[adaptation_indices]
    for test_task_index in range(T_test):
        task_targets =  torch.tensor(test_data[test_task_index], dtype=torch.float)
        adaptation_targets = task_targets[adaptation_indices]
        adapted_model = meta_model.adapt_task_model(adaptation_points, adaptation_targets, 100)
        # print(f'adapted w = {adapted_model.w}')
        predictions = adapted_model(system.grid)
        task_adaptation_error = loss_function(predictions, task_targets)
        adaptation_error[test_task_index] = task_adaptation_error
    adaptation_error_values.append(adaptation_error)

    if not isinstance(meta_model, TaskLinearMetaModel):
        continue

    V_hat, W_hat = meta_model.recalibrate(W_train[:2])
    # V_hat, W_hat = meta_model.recalibrate(W_train)
    W_error = torch.norm(W_hat - W_train)
    W_test_values.append(W_error)

    V_predictions = V_hat(system.grid)
    V_error = loss_function(V_predictions, V_target)
    V_test_values.append(V_error.data)



plt.subplot(2, 1, 1)
plt.yscale('log')
plt.plot(loss_values)
plt.subplot(2, 1, 2)
# plt.yscale('log')
plt.plot(np.array(adaptation_error_values))
# plt.yscale('log')
plt.show()
# plt.subplot(3, 1, 1)
# plt.plot(loss_values)
# plt.title('loss')
# plt.yscale('log')
# plt.subplot(3, 1, 2)
# plt.title('W test')
# plt.plot(W_test_values)
# plt.yscale('log')
# plt.subplot(3, 1, 3)
# plt.title('V test')
# plt.plot(V_test_values)
# plt.yscale('log')
# plt.show()

# v_hat = meta_model.V_hat

for index in range(T_test):
    plt.subplot(2, T_test, index+1)
    w = W_test[index]
    # w = meta_model.W[index]
    plt.title(f'U = {w[0]:3d}, p = {w[1]:3d}')
    # system.plot_potential(w)
    task_targets =  torch.tensor(test_data[index], dtype=torch.float)
    adaptation_points = system.grid[adaptation_indices]
    adaptation_targets = task_targets[adaptation_indices]
    adapted_model = meta_model.adapt_task_model(adaptation_points, adaptation_targets)
    print(f'w {w}')
    print(f'adapted w {adapted_model.w}')
    print(f'adaptation targets {adaptation_targets}')

    system.plot_field(adapted_model)
    plt.scatter(*adaptation_points.T, color="red", s=1, marker='x')

    plt.subplot(2, T_test, index+1+T_test)
    potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
    system.plot_field(potential_map)

plt.show()

