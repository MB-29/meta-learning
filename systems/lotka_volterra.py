import numpy as np
import torch


from utils import integrate_trajectory
from systems.system import System




class LotkaVolterra(System):

    d, m = 2, 2

    alpha, gamma = 0.5, 0.5
    constant_parameters = torch.tensor([alpha, -gamma], dtype=torch.float)

    beta_values_train = [0.5, 0.75, 1.0]
    delta_values_train = [0.5, 0.75, 1.0]
    T = 9
    W_train = np.zeros((T, 2))
    for i in range(3):
        for j in range(3):
            beta, delta = beta_values_train[i], delta_values_train[j]
            index = 3*i + j
            W_train[index] = beta, delta
    # b, d = np.meshgrid(beta_values_train, delta_values_train)
    # W_train = np.concatenate((b.reshape((-1, 1)), d.reshape(-1, 1)), 1)
    W_train = torch.tensor(W_train, dtype=torch.float)
    training_task_n_samples = 4

    beta_values_test = [0.625, 1.125]
    delta_values_test = [0.625, 1.125]
    T_test = 4
    W_test = np.zeros((T_test, 2))
    for i in range(2):
        for j in range(2):
            beta, delta = beta_values_test[i], delta_values_test[j]
            index = 2*i + j
            W_test[index] = beta, delta
    W_test = torch.tensor(W_test, dtype=torch.float)
    test_task_samples = 32

    Nt = 20
    time_values = np.linspace(0, 10, Nt)


    mse_loss = torch.nn.MSELoss()

    def __init__(self) -> None:
        super().__init__(self.W_train, self.d)

        self.test_data = None
        
    
    def V_star(self, x):
        batch_size, _ = x.shape
        x1, x2 = torch.unbind(x, dim=1)
        v = torch.zeros(batch_size, 2, 2)
        v[:, 0, 0] = -x1*x2
        v[:, 1, 1] = x1*x2
        return v

    def c_star(self, x):
        return x * self.constant_parameters
    
    def simulate(self, dynamics, x0_values, diff):
        if diff:
            return integrate_trajectory(x0_values, dynamics, torch.tensor(self.time_values))
        n_samples = x0_values.shape[0]
        trajectories = np.zeros((self.Nt, n_samples, 2))
        for sample_index in range(n_samples):
                # beta, delta = W[t]
            x0 = x0_values[sample_index]
            sample_trajectory = integrate_trajectory(x0, dynamics, self.time_values)
            trajectories[:, sample_index, :] = sample_trajectory
        return trajectories
    
    # def simulate(self, V, c, w, x0_values, diff):
    #     if diff:
    #         return integrate_trajectory(x0_values, V, c, w, torch.tensor(self.time_values))
    #     n_samples = x0_values.shape[0]
    #     trajectories = np.zeros((self.Nt, n_samples, 2))
    #     for sample_index in range(n_samples):
    #             # beta, delta = W[t]
    #         x0 = x0_values[sample_index]
    #         sample_trajectory = integrate_trajectory(x0, V, c, w, self.time_values)
    #         trajectories[:, sample_index, :] = sample_trajectory
    #     return trajectories
    
    def generate_data(self, W, n_samples):
        T = W.shape[0]
        x0_values = 2*np.random.rand(T, n_samples, 2) + 1
        print(f'generate data from x0 {x0_values}')
        task_data = np.zeros((T, self.Nt, n_samples, 2))
        for t in range(T):
            w = W[t]
            environment = self.define_environment(w)
            trajectories = self.simulate(
                environment,
                x0_values[t],
                diff=False
            )
            task_data[t, :, :, :] = trajectories
            
        return task_data
    
    # def simulate(self, w, V, c, x0_values, diff):
    #     trajectories = self.simulate(
    #         V,
    #         c,
    #         w,
    #         x0_values,
    #         diff
    #     )
    #     return trajectories


    def trajectory_loss(self, W, V, c, data):
        T, n_samples = data.shape[:2]
        assert W.shape[0] == T
        loss = 0
        for t in range(T):
            w = W[t]
            task_data = torch.tensor(data[t, :, :, :], dtype=torch.float)
            task_x0_values = task_data[0, :, :]
            predictions = self.simulate(V, c, w, task_x0_values, diff=True)
            task_loss = self.mse_loss(predictions, task_data)
            loss += task_loss
        return loss



    # def evaluate_model(self, meta_model):

    #     test_data = self.test_data
    #     test_data = test_data if test_data is not None else self.generate_test_data()
    #     V, c = meta_model.V, meta_model.c
    #     for t in range(self.T_test):
    #         w = self.W_test[t:t+1]
    #         adaptation_trajectory = self.generate_data(w, n_samples=1)
    #         adapted_model = meta_model.adapt(x, y)
    #         w_hat = adapted_model.w
    #     predictions = self.simulate(V, c, W_adapt)
    #     mse = np.mean((predictions - test_data)**2)
    #     return mse



# beta_values = [0.5, 0.75, 1.0]
# delta_values = [0.5, 0.75, 1.0]
# T = 9
# W_train = torch.zeros((T, 2), dtype=torch.float)
# for i in range(3):
#     for j in range(3):
#         beta, delta = beta_values[i], delta_values[j]
#         index = 3*i + j
#         W_train[index] = beta, delta
# # W_values = torch.randn(5, 2, 3)
# W_calibration = W_train[:2]