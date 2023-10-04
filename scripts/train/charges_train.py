import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from tqdm import tqdm 
from matplotlib import rc
from hypnettorch.mnets import MLP

from meta_training import meta_train, test_model
from models import CAMEL, ANIL, MAML, CoDA
from systems import Charges
# from scripts.train.CH import metamodel_choice

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

np.random.seed(10)
torch.manual_seed(10)

system = Charges(sigma=1e-4)
d, r = system.d, 3

meta_dataset = system.generate_training_data()
T_train = len(meta_dataset)
test_dataset = system.generate_test_data()
T_test = len(test_dataset)

V_net = torch.nn.Sequential(
    nn.Linear(d, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    # nn.Linear(16, 16),
    # nn.Tanh(),
    nn.Linear(16, r)
)
tldr = CAMEL(T_train, r, V_net, c=None)

net = torch.nn.Sequential(
    nn.Linear(d, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, r),
    nn.Tanh(),
    nn.Linear(r, 1)
)

head = torch.nn.Linear(r, 1)

# T_train = 1
maml = MAML(T_train, net, lr=0.002)
anil = ANIL(T_train, V_net, head, lr=0.1)
mnet = MLP(n_in=d, n_out=1, hidden_layers=[16, 16, r], activation_fn=nn.Tanh())
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
    np.random.seed(5)
    torch.manual_seed(5)

    shots = 30
    adaptation_indices = adaptation_indices = (1 + 12*np.arange(30))
    test = {
        'function': test_model,
        'args':{
            'test_dataset': test_dataset,
            'adaptation_indices': adaptation_indices,
            'n_steps': 20,
            }
        }
    n_gradient = 5_000
    batch_size = 300
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

    n_plots = 2
    for index in range(n_plots):
        w = system.W_test[index]
        plt.subplot(2, n_plots, index+1)
        task_test_data = test_dataset[index]
        test_points, test_targets = task_test_data
        adaptation_dataset = (test_points[adaptation_indices], test_targets[adaptation_indices])
        adaptation_points = system.grid[adaptation_indices]
        adaptation_targets = test_targets[adaptation_indices]
        adaptation_dataset = (adaptation_points, adaptation_targets)
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=100)
        predictions = adapted_model(system.grid)
        system.plot_field(adapted_model)
        plt.scatter(*adaptation_points.T, color="red", marker='x')
        plt.subplot(2, n_plots, index+1+n_plots)
        potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
        system.plot_field(potential_map)

plt.show()
