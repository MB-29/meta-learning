import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm 

from model import MetaModel
from systems.dipole import Dipole

# np.random.seed(5)
# torch.manual_seed(5)

system = Dipole()
W_train = system.W_train
T, r = W_train.shape
V_target = system.generate_test_data()

for index in range(system.T):
    plt.subplot(3, 3, index+1)
    w = system.W_train[index]
    plt.title(f'U = {w[0]}, p = {w[1]}')
    # system.plot_potential(w)
    potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
    system.plot_field(potential_map)
plt.show()

training_data = system.generate_training_data()

m = 1
V_net = torch.nn.Sequential(
    nn.Linear(2, 8),
    nn.Tanh(),
    nn.Linear(8, r)
)
# c_net = nn.Linear(2, 2)
W_values = torch.abs(torch.randn(T, r))
# torch.randn(T, 2, requires_grad=True)
meta_model = MetaModel(V_net, c=None)
training_task_models = meta_model.define_task_models(W_values)

# W_calibration = 

n_gradient = 10000
test_interval = n_gradient // 100
W_test_values, V_test_values = [], []
loss_values = np.zeros(n_gradient)
optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.005)
# optimizer_weights = torch.optim.Adam(W_values, lr=0.01)
loss_function = nn.MSELoss()
for step in tqdm(range(n_gradient)):

    loss = system.loss(meta_model, training_data)

    loss_values[step] = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % test_interval != 0:
        continue
    # print(f'step {step}')
    V_hat, W_hat = meta_model.recalibrate(W_train[:2])
    # V_hat, W_hat = meta_model.recalibrate(W_train)
    W_error = torch.norm(W_hat - W_train)
    W_test_values.append(W_error)

    V_predictions = V_hat(system.grid)
    V_error = loss_function(V_predictions, V_target)
    V_test_values.append(V_error.data)


plt.subplot(3, 1, 1)
plt.plot(loss_values)
plt.yscale('log')
plt.subplot(3, 1, 2)
plt.plot(W_test_values)
plt.yscale('log')
plt.subplot(3, 1, 3)
plt.plot(V_test_values)
plt.yscale('log')
plt.show()


for index in range(system.T):
    plt.subplot(3, 3, index+1)
    w = W_train[index]
    # w = meta_model.W[index]
    plt.title(f'U = {w[0]:3f}, p = {w[1]:3f}')
    # system.plot_potential(w)
    potential_map = meta_model.define_model(torch.tensor(w, dtype=torch.float))
    system.plot_field(potential_map)
plt.show()