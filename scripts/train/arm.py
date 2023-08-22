import numpy as np
import torch
import torch.nn as nn
from hypnettorch.mnets import MLP
import matplotlib.pyplot as plt

from systems.arm import ActuatedArm
from models import TLDR, MAML, ANIL, CoDA
from meta_training import meta_train, test_model

sigma = 0
sigma = 1e-4
system = ActuatedArm(sigma=sigma)
d, r = system.d, 4

meta_dataset = system.generate_training_data()
T_train = len(meta_dataset)
test_dataset = system.generate_test_data()
T_test = len(test_dataset)

V_net = torch.nn.Sequential(
    nn.Linear(d, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    # nn.Tanh(),
    # nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, r),
)
tldr = TLDR(T_train, r, V_net, c=None)

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

maml = MAML(T_train, net, lr=0.01)
anil = ANIL(T_train, V_net, head, lr=0.1)
mnet = MLP(n_in=d, n_out=1, hidden_layers=[64, 64, 64, r], activation_fn=nn.Tanh())
coda = CoDA(T_train, r, mnet)


metamodel_choice = {
    'coda': coda,
    'maml': maml,
    'anil': anil,
    'tldr': tldr,
}

metamodel_name = 'tldr'
metamodel_name = 'anil'
metamodel_name = 'maml'
metamodel_name = 'coda'
metamodel = metamodel_choice[metamodel_name]

if __name__ == '__main__':
    np.random.seed(5)
    torch.manual_seed(5)
    shots = 200
    adaptation_indices = np.random.randint(shots, size=shots)
    test = {
        'function': test_model,
        'args':{
            'test_dataset': test_dataset,
            'adaptation_indices': adaptation_indices,
            'n_steps': 100,
            }
        }
    n_gradient = 40_000
    n_gradient = 35_000
    batch_size = 500
    # batch_size = None
    loss_values, test_values = meta_train(
        metamodel,
        meta_dataset,
        # lr=0.001,
        lr=0.0002,
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

