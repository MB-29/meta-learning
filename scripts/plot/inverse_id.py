import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm 
from matplotlib import rc

from models import CAMEL, TaskLinearModel, MAML, CoDA
from systems.static_arm import Arm

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

np.random.seed(10)
torch.manual_seed(10)

system = Arm()
W_train = system.W_train
T_train, r = W_train.shape
W_test = system.W_test
T_test = W_test.shape[0]
V_target = system.generate_V_data()


training_data = system.generate_training_data()
test_data = system.generate_test_data()


shots = 400
adaptation_indices = np.random.randint(400, size=shots)
adaptation_points = system.grid[adaptation_indices]

# adaptation_indices = np.arange(400)

# for index in range(system.T):
#     plt.subplot(3, 3, index+1)
#     w = system.W_train[index]
#     plt.title(f'U = {w[0]}, p = {w[1]}')
#     # system.plot_potential(w)
#     potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
#     system.plot_field(potential_map)
#     # system.plot_potential(potential_map)
# plt.show()

V_net = torch.nn.Sequential(
    nn.Linear(8, 8),
    nn.Tanh(),
    nn.Linear(8, 8),
    nn.Tanh(),
    nn.Linear(8, r)
)
# c_net = nn.Linear(2, 2)
W_values = torch.abs(torch.randn(T_train, r))
# torch.randn(T, 2, requires_grad=True)
metamodel = CAMEL(V_net, c=None)
training_task_models = metamodel.define_task_models(W_values)



n_gradient = 5000
test_interval = max(n_gradient // 100, 1)
W_test_values, V_test_values = [], []
adaptation_error_values = []
loss_values = np.zeros(n_gradient)
optimizer = torch.optim.Adam(metamodel.parameters(), lr=0.005)
# optimizer_weights = torch.optim.Adam(W_values, lr=0.01)
loss_function = nn.MSELoss()
for step in tqdm(range(n_gradient)):
    optimizer.zero_grad()
    
    loss = 0
    # print(f'training {metamodel.net[-1].weight}')
    # print(f'training net {net[-1].weight}')
    # print(f'training W {metamodel.W[:10]}')

    for task_index in range(T_train):
        task_points = system.grid
        task_targets = torch.tensor(training_data[task_index], dtype=torch.float)
        task_model = metamodel.get_training_task_model(task_index, task_points, task_targets)
        task_predictions = system.predict(task_model).squeeze()
        # task_predictions = metamodel.task_models[0](system.grid).squeeze()
        # print(f'prediction {task_predictions[:10]}')
        # task_predictions = model(system.grid)
        task_loss = loss_function(task_predictions, task_targets)
        # print(f'task {task_index}, loss {task_loss}')
        loss += task_loss
        # print(f'task {task_index}, loss {loss}')
        # print(f'task {task_index}, model {task_model.net[-1].weight}')
    # loss += metamodel.W.norm()
    loss /= T_train
    loss.backward()
    loss_values[step] = loss.item()
    # loss.backward(retain_graph=True)
    optimizer.step()

    if step % test_interval != 0:
        continue
    # print(f'step {step}')

    adaptation_error = np.zeros(T_test)
    for test_task_index in range(T_test):
        task_targets =  torch.tensor(test_data[test_task_index], dtype=torch.float)
        adaptation_targets = task_targets[adaptation_indices]
        # print(f'test task {test_task_index}')
        # print(f'task {test_task_index},  model {metamodel.learner.module[0].weight}')
        adapted_model = metamodel.adapt_task_model(adaptation_points, adaptation_targets, 50)
        # print(f'predict')
        # print(f'task {test_task_index}, adapted model {adapted_model.module[0].weight}')
        # print(f'task {test_task_index}, net {net[0].weight}')
        # print(f'targets {adaptation_targets}')

        # print(f'adapted w = {adapted_model.w}')
        predictions = system.predict(adapted_model).squeeze()
        task_adaptation_error = loss_function(predictions, task_targets)
        adaptation_error[test_task_index] = task_adaptation_error
    adaptation_error_values.append(adaptation_error)


    V_hat, W_hat = metamodel.calibrate(W_train[:5])
    # V_hat, W_hat = metamodel.calibrate(W_train)
    W_error = torch.norm(W_hat - W_train) / T_train
    W_test_values.append(W_error)

    V_predictions = V_hat(system.grid)
    V_error = loss_function(V_predictions, V_target)
    V_test_values.append(V_error.data)

fig = plt.figure(figsize=(3, 2.5))
# fig = plt.figure()
fig.set_tight_layout(True)
# plt.subplot(2, 1, 1)
# plt.yscale('log')
# plt.plot(loss_values)
# plt.subplot(2, 1, 2)
# plt.yscale('log')
# plt.plot(np.array(adaptation_error_values))
# # plt.yscale('log')
# plt.show()
# plt.subplot(2, 1, 1)
plt.title('acrobot')
plt.plot(loss_values, label=r'loss', color='black')
# plt.title('loss')
# plt.yscale('log')
# plt.xticks([])
# plt.subplot(2, 1, 2)
# plt.title('W test')
test_indices = np.arange(100)*test_interval
plt.plot(test_indices, W_test_values, label=r'parameter', color='red')
plt.yscale('log')
# plt.subplot(2, 1, 3)
# plt.title('V test')
plt.plot(test_indices, V_test_values, label='features', color='blue')
plt.xticks([0, n_gradient])
# plt.yticks([1e-1, 1e1])
plt.ylim((7e-2, 1e1))
plt.savefig('output/plots/arm-id.pdf')
# plt.legend()
# plt.yscale('log')
plt.show()


