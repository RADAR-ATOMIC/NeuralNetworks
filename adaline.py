'''adaline.py
ADAptive LInear NEuron: a single layer artificial neural network.

Author: Max Atomic
Last updated: 9/15/2024
'''
import numpy as np


class Adaline:
    
    def __init__(self, lr=0.001, epochs=50):

        # learning rate - how fast the network learns
        self.lr = lr
        # how many cycles the network will be trained
        self.epochs = epochs
        # weights connecting input to output layer
        self.wts = None
        # bias neuron in input layer
        self.bias = None

        self.loss_hist = None
        self.acc_hist = None

    def get_wts(self):
        return self.wts.copy()
    
    def get_bias(self):
        return self.bias

    def net_input(self, x):
        return x @ self.wts + self.bias

    def activation(self, net_in):
        net_act = net_in.copy()
        return net_act
    
    def get_predictions(self, x):
        net_in = self.net_input(x)
        net_act = self.activation(net_in)

        return np.where(net_act >= 0, 1, -1)
    
    def accuracy(self, y, y_pred):
        return np.count_nonzero(y==y_pred) / y.size
    
    def loss(self, y, net_act):
        return np.sum(np.power(y - net_act, 2)) / 2
    
    def gradient(self, errs, x):
        grad_bias = -np.sum(errs)
        grad_wts = -1 * (errs @ x)
        return grad_bias, grad_wts

    def fit(self, x, y, rseed=0):
        self.loss_hist = []
        self.acc_hist = []
        rng = np.random.default_rng(seed=rseed)
        self.wts = rng.uniform(low=-0.01, high=0.01, size=(x.shape[1],))
        self.bias = rng.uniform(low=-0.01, high=0.01)

        for _ in range(self.epochs):
            net_in = self.net_input(x)
            net_act = self.activation(net_in)
            y_pred = self.get_predictions(x)
            
            # Calculate loss and accuracy
            loss = self.loss(y, net_act)
            acc = self.accuracy(y, y_pred)

            # Append to history
            self.loss_hist.append(loss)
            self.acc_hist.append(acc)

            # Calculate errors
            errors = y - net_act
            # Calculate gradients
            grad_bias, grad_wts = self.gradient(errors, x)
            
            # Update weights and bias
            self.bias -= self.lr * grad_bias
            self.wts -= self.lr * grad_wts

        return self.loss_hist, self.acc_hist

            
if __name__ == '__main__':

    net = Adaline(lr=0.01, epochs=50)
    test_data = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])

    net.init_wts(test_data)
