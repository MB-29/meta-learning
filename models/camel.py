import numpy as np
import torch
import torch.nn as nn

from models.metamodel import MetaModel


class TaskLinearModel(nn.Module):

    def __init__(self, w, v, c=None, **kwargs):
        super().__init__()
        self.w = w
        self.V = v
        self.c = c 

        if kwargs.get('dynamics') is not None:
            self.predict = self.predict_dynamics

    def forward(self, x):
        assert x.ndim > 1
        V = self.V(x) 
        y = V @ self.w
        y = y if self.c is None else y+self.c(x).squeeze()
        return y
    
    def get_context(self):
        return self.w
    
    def predict(self, x):
        return self.forward(x)
    
    def d_flow(self, t, x):
        return self.forward(x)


class CAMEL(MetaModel):

    def __init__(self, T_train, r, V, c=None, W=None, regularizer=0.0, **kwargs) -> None:
        super().__init__(T_train)
        self.V = V
        self.V_hat = None
        self.c = c
        self.regularizer = regularizer
        
        self.r = r
        self.r_hat = None
        W = W if W is not None else torch.abs(torch.randn(self.T_train, self.r))
        self.W = nn.Parameter(W)

        self.kwargs = kwargs

        self.inv_gram = 100*np.eye(r)
        self.gram = 1e-3*np.eye(r)
        self.cov = np.zeros(r)
        self.w = np.zeros(r)
        # self.define_task_models(W)

    # def define_task_models(self, W):
        
        # # self.W = W
        # # self.task_models = []
        # for t in range(self.T_train):
        #     w = self.W[t]
        #     model = TaskLinearModel(w, self.V_hat, self.c_hat)
        #     self.task_models.append(model)
        # return self.task_models
    # def forward(self, x):
    #     y = self.V(x)@self.W[0]
    #     return y

    def parametrizer(self, task_index, dataset):
        w = self.W[task_index]
        model = TaskLinearModel(w, self.V, self.c, **self.kwargs)
        return model
        # W = W if W is not None else 

    def estimate_context_transform(self, W_star, indices=None):
        calibration_size, _ = W_star.shape
        W = self.W.detach().numpy()
        W_regression = np.concatenate((W, np.ones((calibration_size, 1))), axis=1)
        # print(f'W_regression = {W_regression}')
        estimator, residuals, rank, s = np.linalg.lstsq(
        W_regression, W_star, rcond=None) 
        # print(f'rank {rank}')
        # print(f's {s}')
        return estimator

    def calibrate(self, W_star):
    
        # self.estimator = self.estimate_transform(W_star)
        # self.W_hat = self.W.detach() @ torch.tensor(self.estimator, dtype=torch.float)
        r_star = W_star.shape[1]

        # transform = np.linalg.pinv(self.estimator + 1e-7*np.eye(self.r, r_star))
        calibration_size, _ = W_star.shape
        transform = np.linalg.pinv(W_star)@self.W[:calibration_size].detach().numpy()
        # print(f'pinverse has shape {transform.shape}')
        tensor = torch.tensor(transform, dtype=torch.float, requires_grad=False)
        layer = nn.Linear(r_star, self.r, bias=False)
        layer.weight.data = tensor
        self.r_hat = r_star
        self.V_hat = nn.Sequential(self.V, layer)
        return transform
        # self.c_hat = nn.Sequential(self.c, layer) if self.c is not None else None
        # return self.V_hat, self.W_hat
    
    # def adapt_task_model(self, points, targets, n_steps=None):
    #     v_values = self.V_hat(points)
    #     c_values = self.c_hat(points).detach() if self.c is not None else torch.zeros_like(targets)
        
    #     X = v_values.view(-1, self.r).detach() 
    #     Y = (targets - c_values).view(-1)
        
    #     w_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    #     w = torch.tensor(w_hat, dtype=torch.float)
    #     model = TaskLinearModel(w, self.V_hat, self.c_hat)
    #     return model
    # def get_context_values(self, meta_dataset, **kwargs):
    #     context_values = self.W
    #     return context_values
    def get_context_values(self, dataset, **kwargs):

        T = len(dataset)
        context_values = torch.zeros(T, self.r)
        for task_index in range(T):
            data = dataset[task_index]
            task_model = self.adapt_task_model(data)
            context_values[task_index] = task_model.w
        self.context_values = context_values
        return context_values
    
    
    def adapt_task_model(self, data, **kwargs):
        points, targets = data
        V = self.V_hat if self.V_hat is not None else self.V
        c = self.c
        r = self.r_hat if self.r_hat is not None else self.r
        v_values = V(points)
        c_values = c(points).squeeze().detach() if self.c is not None else torch.zeros_like(targets)

        if self.V_hat is not None:
            print(f'adapting an interpretable model')
        
        X = v_values.detach() 
        Y = (targets - c_values).view(-1)
        # print(f'targets {targets.shape}')
        # print(f'c_values {c_values.shape}')
        # print(X.shape)    
        # print(Y.shape)
        # print(f'X= {X}')
        
        # w_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        lambd = 0
        lambd = 1e-7
        cov = X.T@Y
        gram = X.T@X 
        gram = X.T@X + lambd * len(data) * np.eye(r)
        w_hat = np.linalg.solve(gram, cov)
        w = torch.tensor(w_hat, dtype=torch.float)
        model = TaskLinearModel(w, V, self.c)
        # print(f'adapted w = {w}')
        return model
    
    def create_task_model(self, w):
        return TaskLinearModel(w, self.V, self.c)
    
    # def adapt_online(self, point, target):
    #     v = self.V(point).detach().numpy().T
    #     # print(f'v = {v}')

    #     # Mv = self.inv_gram @ v
    #     numerator = (self.inv_gram @ v) @ (v.T @ self.inv_gram)
    #     denominator = (1+ v.T @ self.inv_gram@v).squeeze()
    #     self.inv_gram += -(1/denominator) * numerator.squeeze()

    #     residual = v.squeeze()@self.w - target.item()

    #     # print(f'residual = {residual}, v = {v} Mv = {Mv}')
    #     self.w = self.w -residual* self.inv_gram@v.squeeze()

    #     return TaskLinearModel(torch.tensor(self.w).float(), self.V, self.c)
    def adapt_online(self, point, target):
        v = self.V(point).detach().numpy().T
        # print(f'v = {v}')

        # Mv = self.inv_gram @ v
        self.gram += v @ v.T
        self.cov += target.item()*v.squeeze()


        self.w = np.linalg.solve(self.gram + 1e-2*np.eye(self.r), self.cov)

        return TaskLinearModel(torch.tensor(self.w).float(), self.V, self.c)


    
    def regularization(self):
        if self.regularizer == 0.0:
            return 0.0
        r_W = self.regularizer * torch.norm(self.W) / self.T_train
        return r_W
    # def adapt_task_model(self, points, targets, n_steps, lr=0.01):
    #     # learner = self.net    
    #     w = torch.randn(self.r, requires_grad=True)
    #     model = TaskLinearModel(w, self.V, self.c)
        
    #     # model = TaskCoDA(self.mnet, )
    #     # for parameter in learner.parameters():
    #     #     parameter.requires_grad = False
    #     # print(f'adapting {learner.module[-1].weight}')
    #     # print(f'adapt {n_steps} steps')
    #     # print(f'targets {targets[:10]}')
    #     adapter = torch.optim.Adam([w], lr=lr)
    #     for adaptation_step in range(n_steps):
    #         predictions = model(points)
    #         train_error = loss_function(predictions.squeeze(), targets.squeeze()) + 00.1*torch.norm(w)**2
    #         adapter.zero_grad()
    #         train_error.backward()
    #         # print(train_error)
    #         adapter.step()
    #         # print(train_error)
    #         # print(f'adapt, step{adaptation_step}')
    #     return model


        


