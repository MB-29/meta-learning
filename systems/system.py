
class System:

    def __init__(self, W_train, d) -> None:
        self.W_train = W_train
        self.T, self.r_star = W_train.shape
        self.d = d

    def V_star(self, x):
        raise NotImplementedError

    def c_star(self, x):
        raise NotImplementedError

    def f_star(self, x, w):
        V = self.V(x)
        return V@w

    def generate_data(self, W):
        raise NotImplementedError
    
    def generate_training_data(self):
        return self.generate_data(self.W_train, self.training_task_samples)
    
    def generate_test_data(self):
        return self.generate_data(self.W_test, self.test_task_samples)
    
    def test_model(self, model):
        raise NotImplementedError

