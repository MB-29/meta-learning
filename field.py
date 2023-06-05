import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm 
from hypnettorch.mnets import MLP

from models import TaskLinearMetaModel, TaskLinearModel, MAML, CoDA, ANIL
from systems.dipole import Dipole

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


shots = 2
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
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, r)
)
# c_net = nn.Linear(2, 2)
# W_values = torch.abs(torch.randn(T_train, r))
# torch.randn(T, 2, requires_grad=True)
tldr = TaskLinearMetaModel(T_train, r, V_net, c=None)
# training_task_models = tldr.define_task_models(W_values)

net = torch.nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, 2),
    nn.Tanh(),
    nn.Linear(2, 1)
)

head = torch.nn.Linear(r, 1)

# T_train = 1
maml = MAML(T_train, net, lr=0.01)
anil = ANIL(T_train, V_net, head, lr=0.1)
mnet = MLP(n_in=2, n_out=1, hidden_layers=[16, 2], activation_fn=nn.Tanh())
coda = CoDA(T_train, 2, mnet)


metamodel_choice = {
    'coda': coda,
    'maml': maml,
    'anil': anil,
    'tldr': tldr,
}

metamodel_name = 'tldr'
metamodel_name = 'coda'
metamodel_name = 'maml'
metamodel_name = 'anil'
metamodel = metamodel_choice[metamodel_name]


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
    # print(f'training {metamodel.net[-1].weight}')
    # print(f'training net {net[-1].weight}')
    # print(f'training W {metamodel.W[:10]}')

    # print(f'body {metamodel.body[0].weight}')
    for task_index in range(T_train):
        # print(f'task {task_index}')
        task_points = system.grid
        task_targets = torch.tensor(training_data[task_index], dtype=torch.float)
        task_model = metamodel.parametrizer(task_index, meta_dataset)
        task_predictions = system.predict(task_model).squeeze()
        # print(f'head {metamodel.learner.weight}')
        # task_predictions = metamodel.task_models[0](system.grid).squeeze()
        # print(f'prediction {task_predictions[:10]}')
        # task_predictions = model(system.grid)
        task_loss = loss_function(task_predictions, task_targets)
        # print(f'task {task_index}, loss {task_loss}')
        loss += task_loss
        # print(f'task {task_index}, loss {loss}')
        # print(f'task {task_index}, model {task_model.net[-1].weight}')
    # loss += metamodel.W.norm()
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
        # print(f'test task {test_task_index}')
        # print(f'task {test_task_index},  model {metamodel.learner.module[0].weight}')
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=10)
        # print(f'predict')
        # print(f'task {test_task_index}, adapted model {adapted_model.module[0].weight}')
        # print(f'task {test_task_index}, net {net[0].weight}')
        # print(f'targets {adaptation_targets}')

        # print(f'adapted w = {adapted_model.w}')
        predictions = system.predict(adapted_model).squeeze()
        task_adaptation_error = loss_function(predictions, task_targets)
        adaptation_error[test_task_index] = task_adaptation_error
    adaptation_error_values.append(adaptation_error)

    # if metamodel_name != 'tldr':
    #     continue    
    
    # V_hat, W_hat = metamodel.calibrate(W_train[:2])
    # # V_hat, W_hat = metamodel.calibrate(W_train)
    # W_error = torch.norm(W_hat - W_train)
    # W_test_values.append(W_error)

    # V_predictions = V_hat(system.grid)
    # V_error = loss_function(V_predictions, V_target)
    # V_test_values.append(V_error.data)

path = f'output/models/dipole/{metamodel_name}_ngrad-{n_gradient}.ckpt'
torch.save(metamodel.state_dict(), path)


plt.subplot(2, 1, 1)
plt.yscale('log')
plt.plot(loss_values)
plt.subplot(2, 1, 2)
plt.yscale('log')
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

# v_hat = metamodel.V_hat

for index in range(T_test):
    plt.subplot(2, T_test, index+1)
    w = W_test[index]
    # w = metamodel.W[index]
    plt.title(f'U = {w[0]:0.2f}, p = {w[1]:0.2f}')
    task_targets =  torch.tensor(test_data[index], dtype=torch.float)
    adaptation_points = system.grid[adaptation_indices]
    adaptation_targets = task_targets[adaptation_indices]
    adaptation_dataset = (adaptation_points, adaptation_targets)
    adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=5, plot=False)
    # adapted_model.head.module.weight.data = torch.randn_like(adapted_model.head.weight)
    # adapted_model.head.bias = 0
    print(adapted_model.head.module.weight)
    # print(adapted_model.head.bias)
    predictions = adapted_model(system.grid)
    # print(predictions[:10])
    # print(adapted_model.body[-1].weight)
    # print(f'w {w}')
    # print(f'adapted w {adapted_model.w}')
    # print(f'adapted model {adapted_model.module[0].weight}')
    # print(f'adaptation targets {adaptation_targets}')
    system.plot_field(adapted_model)
    # system.plot_potential(adapted_model)
    plt.scatter(*adaptation_points.T, color="red", s=1, marker='x')
    plt.subplot(2, T_test, index+1+T_test)
    potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
    system.plot_field(potential_map)
    # system.plot_potential(potential_map)

plt.show()

