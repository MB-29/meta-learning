import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm 
from matplotlib import rc

from models import TaskLinearMetaModel, MAML, ANIL
from systems.dipole import Dipole

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

np.random.seed(5)
torch.manual_seed(5)

system = Dipole()
W_train = system.W_train
T_train, r = W_train.shape
W_test = system.W_test
T_test = W_test.shape[0]
V_target = system.generate_V_data()


training_data = system.generate_training_data()
meta_dataset = [(system.grid, torch.tensor(training_data[t]).float()) for t in range(T_train)]
test_data = system.generate_test_data()


shots = 5
adaptation_indices = np.random.randint(400, size=shots)
adaptation_points = system.grid[adaptation_indices]


V_net = torch.nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, r)
)
net = torch.nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, r),
    nn.Tanh(),
    nn.Linear(r, 1)
)
# c_net = nn.Linear(2, 2)
# torch.randn(T, 2, requires_grad=True)
# W_init = torch.abs(torch.randn(T_train, r))
head = torch.nn.Linear(r, 1)
metamodel = TaskLinearMetaModel(T_train, r, V_net, c=None)
# metamodel = ANIL(T_train, V_net, head, lr=0.1)
# metamodel = MAML(T_train, net, lr=0.1)


n_gradient = 10000
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
    for task_index in range(T_train):
        # print(f'task {task_index}')
        task_points = system.grid
        task_targets = torch.tensor(training_data[task_index], dtype=torch.float)
        task_model = metamodel.parametrizer(task_index, meta_dataset)
        task_predictions = system.predict(task_model).squeeze()
        task_loss = loss_function(task_predictions, task_targets)
        # print(f'task {task_index}, loss {task_loss}')
        loss += task_loss
        # print(f'task {task_index}, loss {loss}')
        # print(f'task {task_index}, model {task_model.net[-1].weight}')
    # loss += metamodel.W.norm()
    # loss /= T_train
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
        adaptation_dataset = (system.grid[adaptation_indices], task_targets[adaptation_indices])
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=20)
        predictions = system.predict(adapted_model).squeeze()
        task_adaptation_error = loss_function(predictions, task_targets)
        adaptation_error[test_task_index] = task_adaptation_error
    adaptation_error_values.append(adaptation_error)

    # metamodel.get_context(meta_dataset, n_steps=10)

    W_bar = metamodel.W.data
    tldr = TaskLinearMetaModel(T_train, r, V_net, W=W_bar)

    V_hat, W_hat = tldr.calibrate(W_train[:2])
    # V_hat, W_hat = metamodel.calibrate(W_train)
    W_error = torch.norm(W_hat - W_train) / T_train
    W_test_values.append(W_error)

    V_predictions = V_hat(system.grid)
    V_error = loss_function(V_predictions, V_target)
    V_test_values.append(V_error.data)

plt.subplot(2, 1, 1)
plt.yscale('log')
plt.plot(loss_values)
plt.subplot(2, 1, 2)
plt.yscale('log')
plt.plot(np.array(adaptation_error_values))
plt.show()

fig = plt.figure(figsize=(3, 2.5))
# fig = plt.figure()
fig.set_tight_layout(True)
plt.title('dipole')
# plt.plot(loss_values, label=r'loss', color='black')
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
# plt.savefig('output/plots/dipole-id.pdf')
# plt.legend()
# plt.yscale('log')
plt.show()



