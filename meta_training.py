import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm 

loss_function = nn.MSELoss()

def meta_train(metamodel, meta_dataset, lr, n_gradient, test=None):
    T_train = len(meta_dataset)
    loss_values = np.zeros(n_gradient)
    optimizer = torch.optim.Adam(metamodel.parameters(), lr=lr)

    test_interval = max(n_gradient // 100, 1)
    test_values = []

    for step in tqdm(range(n_gradient)):
        optimizer.zero_grad()
        
        loss = 0
        for task_index in range(T_train):
            task_points, task_targets = meta_dataset[task_index]
            task_model = metamodel.parametrizer(task_index, meta_dataset)
            task_predictions = task_model(task_points).squeeze()
            task_loss = loss_function(task_predictions, task_targets)
            loss += task_loss 
        loss += metamodel.regularization()
        loss.backward()
        loss_values[step] = loss.item()
        optimizer.step()

        if step % test_interval != 0 or test is None:
            continue
        test_function = test['function']
        test_error = test_function(metamodel, **test['args'])
        test_values.append(test_error)

    return loss_values, test_values

def test_model(metamodel, test_dataset, adaptation_indices, n_steps):
    T_test = len(test_dataset)
    adaptation_error = np.zeros(T_test)
    for test_task_index in range(T_test):
        task_test_data = test_dataset[test_task_index]
        test_points, test_targets = task_test_data
        adaptation_dataset = (test_points[adaptation_indices], test_targets[adaptation_indices])
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=n_steps)
        predictions = adapted_model(test_points).squeeze()
        task_adaptation_error = loss_function(predictions, test_targets)
        adaptation_error[test_task_index] = task_adaptation_error
    return adaptation_error

if __name__ == '__main__':

    from models import TLDR
    from systems import Cartpole, Dipole

    # np.random.seed(5)
    # torch.manual_seed(5)

    system = Dipole()
    system = Cartpole()
    d, r = system.d, system.r

    shots = 2
    adaptation_indices = np.random.randint(400, size=shots)

    meta_dataset = system.generate_training_data()
    T_train = len(meta_dataset)
    # meta_dataset = [(system.grid, torch.tensor(training_data[t]).float()) for t in range(T_train)]
    test_dataset = system.generate_test_data()
    T_test = len(test_dataset)
    
    V_net = torch.nn.Sequential(
    nn.Linear(d, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, r)
    )  

    metamodel = TLDR(T_train, r, V_net, c=None)
    test = {
        'function': test_model,
        'dataset': test_dataset,
        'indices': adaptation_indices
        }
    
    loss_values, test_values = meta_train(
        metamodel,
        meta_dataset,
        lr=0.005,
        n_gradient=5000,
        test=test
        )

    plt.subplot(2, 1, 1)
    plt.plot(loss_values)
    plt.yscale('log')
    plt.subplot(2, 1, 2)
    plt.plot(test_values)
    plt.yscale('log')
    plt.show()

    torch.save(metamodel.state_dict(), 'output/trained_tldr.ckpt')

    for index in range(T_test):
        w = system.W_test[index]
        plt.subplot(2, T_test, index+1)
        task_test_data = test_dataset[index]
        test_points, test_targets = task_test_data
        adaptation_dataset = (test_points[adaptation_indices], test_targets[adaptation_indices])
        adaptation_points = system.grid[adaptation_indices]
        adaptation_targets = test_targets[adaptation_indices]
        adaptation_dataset = (adaptation_points, adaptation_targets)
        adapted_model = metamodel.adapt_task_model(adaptation_dataset, n_steps=0)
        predictions = adapted_model(system.grid)
        system.plot_field(adapted_model)
        plt.scatter(*adaptation_points.T, color="red", s=1, marker='x')
        plt.subplot(2, T_test, index+1+T_test)
        potential_map = system.define_environment(torch.tensor(w, dtype=torch.float))
        system.plot_field(potential_map)

    plt.show()
