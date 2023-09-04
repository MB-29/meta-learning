import numpy as np
import torch
import torch.nn as nn
from hypnettorch.hnets import HMLP
from hypnettorch.utils.torch_ckpts import save_checkpoint, load_checkpoint
import matplotlib.pyplot as plt

from models.metamodel import MetaModel

loss_function = nn.MSELoss()

class TaskCoDA(nn.Module):

    def __init__(self, net, weights, xi):
        super().__init__()
        self.net = net
        self.weights = weights
        self.xi = xi

    def forward(self, x):
        return self.net.forward(x, weights=self.weights)
    
    def get_context(self):
        return self.xi

    

class CoDA(MetaModel):

    def __init__(self, T_train, d_xi, mnet) -> None:
        super().__init__(T_train)
        self.d_xi = d_xi

        # self.learner = l2la.MAML(net, lr=0.01, first_order=False)
        self.mnet = mnet
        # self.theta = parameters_to_vector(mnet.parameters())
        # self.d_theta = self.theta.shape[0]  
        self.hnet  = HMLP(target_shapes=mnet.param_shapes, uncond_in_size=d_xi, cond_in_size=0, num_cond_embs=0, layers=[])
        Xi = torch.zeros(1, d_xi, self.T_train)
        # self.Xi = Xi
        self.Xi = nn.Parameter(Xi)

        # self.W = W
        # self.task_models = []
        # for t in range(T):
        #     xi = self.Xi[:, :, t]
        #     task_weights = self.hnet.forward(uncond_input=xi)
        #     model = TaskCoDA(mnet, task_weights)
        #     # model = lambda x : self.mnet(x, weights=task_weights)
        #     self.task_models.append(model)
        # self.task_models = []
        # for t in range(T):
        #     xi = self.Xi[:, :, t]
        
        # model = lambda x : self.mnet(x, weights=task_weights)
        
    
    def parametrizer(self, task_index, dataset):
        xi = self.Xi[:, :, task_index]
        task_weights = self.hnet.forward(uncond_input=xi)
        task_model = TaskCoDA(self.mnet, task_weights, xi=xi.clone())
        return task_model
    
    def adapt_task_model(self, dataset, n_steps, lr=0.05, plot=False):
        points, targets = dataset
        # learner = self.net    
        xi = torch.zeros(1, self.d_xi, requires_grad=True)
        # xi = self.Xi[:, :, 0].clone().detach().requires_grad_(True)
        
        # model = TaskCoDA(self.mnet, )
        # for parameter in learner.parameters():
        #     parameter.requires_grad = False
        # print(f'adapting {learner.module[-1].weight}')
        # print(f'adapt {n_steps} steps')
        # print(f'targets {targets[:10]}')
        train_error_values = []
        adapter = torch.optim.Adam([xi], lr=lr)
        task_weights = self.hnet.forward(uncond_input=xi)
        for adaptation_step in range(n_steps):
            task_weights = self.hnet.forward(uncond_input=xi)
            predictions = self.mnet.forward(points, weights=task_weights)
            train_error = loss_function(predictions.squeeze(), targets.squeeze()) + 1e-3*xi.norm(p=1)**2
            # train_error = loss_function(predictions.squeeze(), targets.squeeze())
            train_error_values.append(train_error.item())
            adapter.zero_grad()
            train_error.backward()
            # print(train_error)
            adapter.step()
            # print(f'adapt, step{adaptation_step}')
        if plot:
            plt.figure()
            plt.plot(train_error_values) ; plt.yscale('log') ; plt.show()
        # plt.pause(0.1)  ; plt.close()
        model = TaskCoDA(self.mnet, task_weights, xi.data)
        return model
    
    def regularization(self):
        r_xi = 1e-5*(torch.norm(self.Xi, p=1, dim=1)**2).sum()
        W = list(self.hnet.parameters())[0]
        r_W = 1e-2*torch.norm(W, p=2, dim=1).sum()
        return r_xi + r_W
    def get_context_values(self, dataset, n_steps):
        self.context_values = self.Xi.squeeze().T.detach()
        return self.context_values
    # def save(self, path):
    #     information = {'state_dict': self.hnet.state_dict()}
    #     save_checkpoint(information, path, 0)

    # def load(self, path):
    #     load_checkpoint(path, self.hnet)