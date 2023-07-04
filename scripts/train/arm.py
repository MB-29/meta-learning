import numpy as np
import torch
import torch.nn as nn
from hypnettorch.mnets import MLP
import matplotlib.pyplot as plt

from systems.arm import ActuatedArm
from models import TLDR, MAML, ANIL, CoDA
from meta_training import meta_train, test_model

system = ActuatedArm()
d, r = system.d, system.r

meta_dataset = system.generate_training_data()
T_train = len(meta_dataset)
test_dataset = system.generate_test_data()
T_test = len(test_dataset)

V_net = torch.nn.Sequential(
    nn.Linear(d, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, r),
)
tldr = TLDR(T_train, r, V_net, c=None)

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

maml = MAML(T_train, net, lr=0.01)
anil = ANIL(T_train, V_net, head, lr=0.1)
mnet = MLP(n_in=d, n_out=1, hidden_layers=[16, 16, r], activation_fn=nn.Tanh())
coda = CoDA(T_train, 2, mnet)


metamodel_choice = {
    'coda': coda,
    'maml': maml,
    'anil': anil,
    'tldr': tldr,
}

metamodel_name = 'coda'
metamodel_name = 'maml'
metamodel_name = 'anil'
metamodel_name = 'tldr'
metamodel = metamodel_choice[metamodel_name]

if __name__ == '__main__':
    np.random.seed(5)
    torch.manual_seed(5)
    shots = 10
    adaptation_indices = np.random.randint(200, size=shots)
    test = {
        'function': test_model,
        'args':{
            'test_dataset': test_dataset,
            'adaptation_indices': adaptation_indices,
            'n_steps': 10,
            }
        }
    n_gradient = 50_000
    batch_size = 200
    # batch_size = None
    loss_values, test_values = meta_train(
        metamodel,
        meta_dataset,
        lr=0.002,
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

