import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm 
import pickle

from models import TaskLinearMetaModel, MAML, CoDA
from systems.arm import Arm
from hypnettorch.mnets import MLP

np.random.seed(2)
torch.manual_seed(2)
# np.random.seed(5)
# torch.manual_seed(5)

system = Arm()
W_train = system.W_train
T_train, r = W_train.shape
W_test = system.W_test
T_test = W_test.shape[0]
V_target = system.generate_V_data()


training_data = system.generate_training_data()
test_data = system.generate_test_data()


max_shots = 50
adaptation_indices = np.random.randint(400, size=max_shots)
adaptation_points = system.grid[adaptation_indices]

# adaptation_indices = np.arange(400)


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
tldr = TaskLinearMetaModel(V_net, c=None)
training_task_models = tldr.define_task_models(W_values)

net = torch.nn.Sequential(
    nn.Linear(8, 8),
    nn.Tanh(),
    nn.Linear(8, r),
    # nn.Tanh(),
    # nn.Linear(8, r),
    nn.Tanh(),
    nn.Linear(r, 1)
)

# T_train = 1
maml = MAML(net, lr=0.02, first_order=False)
mnet = MLP(n_in=8, n_out=1, hidden_layers=[8, 8], activation_fn=nn.Tanh())
coda = CoDA(T_train, 2, mnet)

metamodel_choice = {
    'tldr': tldr,
    'maml': maml,
    'coda': coda,
}   

metamodel_name = 'tldr'
metamodel_name = 'coda'
metamodel_name = 'maml'
metamodel = metamodel_choice[metamodel_name]


n_gradient = 5000
test_interval = max(n_gradient // 100, 1)
W_test_values, V_test_values = [], []
adaptation_error_values = []
loss_values = np.zeros(n_gradient)
optimizer = torch.optim.Adam(metamodel.parameters(), lr=0.05)
# optimizer_weights = torch.optim.Adam(W_values, lr=0.01)
loss_function = nn.MSELoss()
for step in tqdm(range(n_gradient)):
    optimizer.zero_grad()
    
    loss = 0
    # print(f'training {metamodel.net[-1].weight}')
    # print(f'training net {net[-1].weight}')
    # print(f'training W {metamodel.W[:10]}')

    for task_index in range(T_train):
        training_indices = np.random.randint(system.grid.shape[0], size=1000)
        task_points = system.grid[training_indices]
        task_targets = torch.tensor(training_data[task_index], dtype=torch.float)[training_indices]
        task_model = metamodel.get_training_task_model(task_index, task_points, task_targets)
        # task_predictions = system.predict(task_model).squeeze()
        task_predictions = task_model(task_points).squeeze()
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
        adapted_model = metamodel.adapt_task_model(adaptation_points, adaptation_targets, 10)
        # print(f'predict')
        # print(f'task {test_task_index}, adapted model {adapted_model.module[0].weight}')
        # print(f'task {test_task_index}, net {net[0].weight}')
        # print(f'targets {adaptation_targets}')

        # print(f'adapted w = {adapted_model.w}')
        predictions = system.predict(adapted_model).squeeze()
        task_adaptation_error = loss_function(predictions, task_targets)
        adaptation_error[test_task_index] = task_adaptation_error
    adaptation_error_values.append(adaptation_error)

    if metamodel_name != 'tldr':
        continue    

    V_hat, W_hat = metamodel.calibrate(W_train[:5])
    # V_hat, W_hat = metamodel.calibrate(W_train)
    W_error = torch.norm(W_hat - W_train)
    W_test_values.append(W_error)

    V_predictions = V_hat(system.grid)
    V_error = loss_function(V_predictions, V_target)
    V_test_values.append(V_error.data)

if metamodel_name != 'coda':
    path = f'output/models/inverse/{metamodel_name}_ngrad-{n_gradient}.dat'
    with open(path, 'wb') as file:
        torch.save(metamodel, file)

plt.subplot(2, 1, 1)
plt.yscale('log')
plt.plot(loss_values)
plt.subplot(2, 1, 2)
plt.yscale('log')
plt.plot(np.array(adaptation_error_values))
# plt.yscale('log')
plt.show()
if metamodel_name =='tldr':
    plt.subplot(3, 1, 1)
    plt.plot(loss_values)
    plt.title('loss')
    plt.yscale('log')
    plt.subplot(3, 1, 2)
    plt.title('W test')
    plt.plot(W_test_values)
    plt.yscale('log')
    plt.subplot(3, 1, 3)
    plt.title('V test')
    plt.plot(V_test_values)
    plt.yscale('log')
    plt.show()

# v_hat = metamodel.V_hat
shot_values = np.arange(3, 50)
adaptation_error_values = []
for shot_index, max_shots in enumerate(shot_values):
    print(f'shot {max_shots}')
    adaptation_error = np.zeros(T_test)
    for task_index in range(T_test):
        w = W_test[task_index]
        task_targets =  torch.tensor(test_data[task_index], dtype=torch.float)
        shot_indices = adaptation_indices[:max_shots]
        adaptation_points = system.grid[shot_indices]
        adaptation_targets = task_targets[shot_indices]
        adapted_model = metamodel.adapt_task_model(adaptation_points, adaptation_targets, 50)
        predictions = system.predict(adapted_model).squeeze()
        task_adaptation_error = loss_function(predictions, task_targets)
        adaptation_error[task_index] = task_adaptation_error
    adaptation_error_values.append(adaptation_error.copy())


with open(f'output/models/inverse/{metamodel_name}_adaptation_values.pkl', 'wb') as file:
    pickle.dump(adaptation_error_values, file)

plt.plot(adaptation_error_values)
plt.yscale('log')
plt.show()

