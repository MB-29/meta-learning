import numpy as np
import torch
import torch.nn as nn
from hypnettorch.mnets import MLP
import matplotlib.pyplot as plt

from systems import Capacitor
from models import TLDR, ANIL, MAML, CoDA
from meta_training import meta_train, test_model



sigma = 0
# sigma = 5e-3
data_path = 'data/capacitor/epsilon_05'
system = Capacitor(data_path, sigma=sigma)
d, r = system.d, 3

meta_dataset = system.generate_training_data()
T_train = len(meta_dataset)
test_dataset = system.generate_test_data()
T_test = len(test_dataset)

# w = np.array([1., -1, 1., -1])
# potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
# system.plot_field(potential_map)
# plt.show()

V_net = torch.nn.Sequential(
    nn.Linear(d, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    # nn.Tanh(),
    # nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, r)
)
c_net = torch.nn.Sequential(
    nn.Linear(d, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

regularizer = 1e-3

tldr = TLDR(T_train, r, V_net, c=c_net, W=torch.zeros(T_train, r), regularizer=regularizer)

net = torch.nn.Sequential(
    nn.Linear(d, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, r),
    nn.Tanh(),
    nn.Linear(r, 1)
)

head = torch.nn.Linear(r, 1)

# T_train = 1
maml = MAML(T_train, net, lr=0.05)
anil = ANIL(T_train, V_net, head, lr=0.05)
mnet = MLP(n_in=d, n_out=1, hidden_layers=[64, 64, 64, r], activation_fn=nn.Tanh())
coda = CoDA(T_train, r, mnet)


metamodel_choice = {
    'coda': coda,
    'maml': maml,
    'anil': anil,
    'tldr': tldr,
}

metamodel_name = 'maml'
metamodel_name = 'tldr'
metamodel_name = 'coda'
metamodel_name = 'anil'
metamodel = metamodel_choice[metamodel_name]

if __name__ == '__main__':


    # shots = 40
    adaptation_indices = np.concatenate([20 + k*50 +np.arange(7)*9_000 for k in range(6)])

    test = {
        'function': test_model,
        'args':{
            'test_dataset': test_dataset,
            'adaptation_indices': adaptation_indices,
            'n_steps': 20,
            }
        }
    n_gradient = 30_000
    n_gradient = 1_000
    # n_gradient = 10_000
    batch_size = 500
    loss_values, test_values = meta_train(
        metamodel,
        meta_dataset,
        lr=0.005,
        n_gradient=n_gradient,
        test=test,
        batch_size=batch_size
        )
    torch.save(metamodel.state_dict(), f'output/models/{metamodel_name}_{n_gradient}.ckpt')

    plt.plot(loss_values)
    plt.yscale('log')   
    plt.show()  
    plt.plot(test_values)
    plt.yscale('log')
    plt.show()
    # test_dataset = meta_dataset[:3]

    n_plots = len(test_dataset)
    for index in range(n_plots):
        w = system.W_test[index]
        task_test_data = test_dataset[index]
        test_points, test_targets = task_test_data
        adaptation_dataset = (test_points[adaptation_indices], test_targets[adaptation_indices])
        adaptation_points = system.grid[adaptation_indices]
        adaptation_targets = test_targets[adaptation_indices]
        adaptation_dataset = (adaptation_points, adaptation_targets)

        plt.subplot(2, n_plots, index+1)
        potential_values = test_dataset[index][1]
        system.plot_potential_values(potential_values)
        plt.scatter(*adaptation_points.T, color="red", marker='x')

        plt.subplot(2, n_plots, n_plots+index+1)
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=100)
        predictions = adapted_model(system.grid)
        system.plot_potential(adapted_model)
        

plt.show()
