import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import MetaModel
from systems.lotka_volterra import LotkaVolterra

np.random.seed(5)
torch.manual_seed(5)

system = LotkaVolterra()
W_train = system.W_train


n_points = 10
x1_values = torch.linspace(0, 3, n_points)
x2_values = torch.linspace(0, 3, n_points)
grid_x1, grid_x2 = torch.meshgrid(
    x1_values, x2_values, indexing='ij')
grid = torch.cat([
    grid_x1.reshape(-1, 1),
    grid_x2.reshape(-1, 1),
], 1)


V_target = system.V_star(grid)
c_target = system.c_star(grid)
y_target = V_target@W_train.T + c_target.unsqueeze(2).expand(-1, -1, system.T)

training_data = system.generate_training_data()
    
# for t in range(9):
#     y = y_target[:, :, t]
#     plt.subplot(3, 3, t+1)
#     # plt.xlim((-phi_max, phi_max))
#     # plt.ylim((-dphi_max, dphi_max))
#     vector_x = y[:, 0].reshape(
#         n_points, n_points).detach().numpy()
#     vector_y = y[:, 1].reshape(
#         n_points, n_points).detach().numpy()
#     magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
#     linewidth = magnitude / magnitude.max()
#     plt.streamplot(
#         grid_x1.numpy().T,
#         grid_x2.numpy().T,
#         vector_x.T,
#         vector_y.T,
#         color='black',
#         linewidth=linewidth*2,
#         arrowsize=.8,
#         density=.8)
#     trajectory = training_data[t, :, 0]
#     plt.scatter(trajectory[:, 0], trajectory[:, 1], marker='x', color='red')
# plt.show()

r = 2
d_ = 2
V_net = torch.nn.Sequential(
    nn.Linear(2, 8),
    nn.Tanh(),
    nn.Linear(8, d_*r),
    nn.Unflatten(1, (d_, r))
)
c_net = nn.Linear(2, 2)
W0 = torch.abs(torch.randn_like(W_train))
W = W0.clone().requires_grad_(True)
# torch.randn(T, 2, requires_grad=True)
meta_model = MetaModel(V_net, c_net)
task_models = meta_model.define_task_models(W)
# W_calibration = 

n_gradient = 50000
test_interval = n_gradient // 100
W_test_values, V_test_values = [], []
loss_values = np.zeros(n_gradient)
optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
# optimizer_weights = torch.optim.Adam(W_values, lr=0.01)
loss_function = nn.MSELoss()
for step in tqdm(range(n_gradient)):
    # print(W_values)
    loss = 0
    for t in range(meta_model.T):
        model = meta_model.task_models[t]
        predictions = model(grid)   
        loss += loss_function(y_target[:, :, t], predictions)

    # loss = system.trajectory_loss(meta_model.W, meta_model.V, meta_model.c, training_data)
    # print(f'loss = {loss}')

    loss_values[step] = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % test_interval != 0:
        continue
    # print(f'step {step}')
    # V_hat, W_hat = meta_model.recalibrate(W_train[:2])
    V_hat, W_hat = meta_model.recalibrate(W_train)
    W_error = torch.norm(W_hat - W_train)
    W_test_values.append(W_error)

    V_predictions = V_hat(grid)
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

# w_new = torch.tensor([0.05, 0.5])
# y_values = V_target@w_new[:, None]
# adapted_model = meta_model.adapt(grid, y_values)

n_w = 20
w1_values = torch.linspace(0.25, 1.25, n_w)
w2_values = torch.linspace(0.25, 1.25, n_w)
grid_w1, grid_w2 = torch.meshgrid(
    w1_values, w2_values, indexing='ij')
grid_w = torch.cat([
    grid_w1.reshape(-1, 1),
    grid_w2.reshape(-1, 1),
], 1)
w_error_values = np.zeros(n_w*n_w)

y_values = V_target@grid_w.T + c_target.unsqueeze(2).expand(-1, -1,n_w**2)
for index, w in enumerate(grid_w):
    adapted_model = meta_model.adapt(grid, y_values[:, :, index])
    w_hat = adapted_model.w
    w_error =  100 * (np.abs(w - w_hat) / np.abs(w)).mean()
    w_error_values[index] = w_error

plt.pcolormesh(grid_w1, grid_w2, w_error_values.reshape(
    (n_w, n_w)), cmap='gray')
plt.colorbar()
for w in W_train:
    plt.scatter(*w, marker='x', color='red')
plt.show()
