import numpy as np
import torch
import torch.nn as nn
from hypnettorch.mnets import MLP
import matplotlib.pyplot as plt

from systems import Dipole
from models import TLDR, ANIL, MAML, CoDA
from meta_training import meta_train, test_model



system = Dipole()
d, r = system.d, system.r

meta_dataset = system.generate_training_data()
T_train = len(meta_dataset)
test_dataset = system.generate_test_data()
T_test = len(test_dataset)

V_net = torch.nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, r)
)
tldr = TLDR(T_train, r, V_net, c=None)

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

metamodel_name = 'maml'
metamodel_name = 'anil'
metamodel_name = 'coda'
metamodel_name = 'tldr'
metamodel = metamodel_choice[metamodel_name]

if __name__ == '__main__':
    np.random.seed(5)
    torch.manual_seed(5)

    shots = 20
    adaptation_indices = np.random.randint(400, size=shots)
    test = {
        'function': test_model,
        'args':{
            'test_dataset': test_dataset,
            'adaptation_indices': adaptation_indices,
            'n_steps': 20,
            }
        }
    n_gradient = 5000
    loss_values, test_values = meta_train(
        metamodel,
        meta_dataset,
        lr=0.005,
        n_gradient=n_gradient,
        test=test
        )
    torch.save(metamodel.state_dict(), f'output/models/dipole/{metamodel_name}_{n_gradient}.ckpt')

    plt.plot(loss_values)
    plt.yscale('log')
    plt.show()
    plt.plot(test_values)
    plt.yscale('log')
    plt.show()


    for index in range(T_test):
        w = system.W_test[index]
        plt.subplot(2, T_test, index+1)
        task_test_data = test_dataset[index]
        test_points, test_targets = task_test_data
        adaptation_dataset = (test_points[adaptation_indices], test_targets[adaptation_indices])
        adaptation_points = system.grid[adaptation_indices]
        adaptation_targets = test_targets[adaptation_indices]
        adaptation_dataset = (adaptation_points, adaptation_targets)
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=100, lr=0.1, plot=False)
        predictions = adapted_model(system.grid)
        system.plot_field(adapted_model)
        plt.scatter(*adaptation_points.T, color="red", s=1, marker='x')
        plt.subplot(2, T_test, index+1+T_test)
        potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
        system.plot_field(potential_map)

    plt.show()
