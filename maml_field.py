import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm 

from systems.dipole import Dipole
# from learn2learn.algorithms import MAML
from model import MAML



np.random.seed(5)
torch.manual_seed(5)

system = Dipole()
W_train = system.W_train
T, r = W_train.shape
V_target = system.generate_test_data()

W_test = system.W_test
T_test = W_test.shape[0]
V_target = system.generate_test_data()

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

m = 1
net = torch.nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    # nn.Linear(32, 32),
    # nn.Tanh(),
    nn.Linear(16, 2),
    nn.Tanh(),
    nn.Linear(2, 1)
)
meta_model = MAML(net, lr=0.001, first_order=False)
# c_net = nn.Linear(2, 2)
# torch.randn(T, 2, requires_grad=True)

# W_calibration = 

n_adaptation = 1
n_adapt_steps = 1

n_gradient = 500
test_interval = n_gradient // 100
W_test_values, V_test_values = [], []
adaptation_error_values = []
loss_values = np.zeros(n_gradient)
optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.005)
# optimizer_weights = torch.optim.Adam(W_values, lr=0.01)
loss_function = nn.MSELoss()
for step in tqdm(range(n_gradient)):
    optimizer.zero_grad()
    loss = 0
    for task_index in range(7):
        # learner = meta_model.clone()
        task_points = system.grid
        task_targets = torch.tensor(training_data[task_index], dtype=torch.float)
        # for adaptation_step in range(n_adaptation):     
        #     train_error = loss_function(learner(task_points), task_targets)
        #     learner.adapt(train_error)
        learner = meta_model.get_training_task_model(task_index, task_points, task_targets)
        task_predictions = learner(system.grid)
        task_loss = loss_function(task_predictions.squeeze(), task_targets)
        # task_loss.backward()
        loss += task_loss
    loss.backward()
        # print(f'step {step}, loss {loss}, task loss {task_loss}')

    loss_values[step] = loss


    # loss.backward()
    optimizer.step()

    if step%test_interval != 0:
        continue
    # T_test = 2
    adaptation_error = np.zeros(T_test)
    for test_task_index in range(T_test):
        task_points = system.grid
        adaptation_task_targets = torch.tensor(test_data[test_task_index], dtype=torch.float)
        # for adaptation_step in range(n_adaptation):     
        #     train_error = loss_function(learner(task_points), adaptation_task_targets)
        # #     learner.adapt(train_error)
        # learner = meta_model.adapt_task_model(system.grid, adaptation_task_targets, 3)
        adaptation_points = system.grid
        adapted_model = meta_model.adapt_task_model(adaptation_points, adaptation_task_targets, 50)
        # # predictions = adapted_model(system.grid)
        task_predictions = adapted_model(system.grid)
        adapted_task_loss = loss_function(task_predictions.squeeze(), adaptation_task_targets)
        # learner = meta_model.clone()
        # predictions = learner(system.grid)
        # task_adaptation_error = loss_function(predictions, adaptation_task_targets)
        adaptation_error[test_task_index] = adapted_task_loss
    adaptation_error_values.append(adaptation_error)


plt.subplot(2, 1, 1)
plt.plot(loss_values)
# plt.yscale('log')
plt.subplot(2, 1, 2)
plt.plot(np.array(adaptation_error_values))
# plt.yscale('log')
plt.show()

