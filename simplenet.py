# -*- coding: utf-8 -*-

import numpy as np
from gradient import gradient_descent, numerical_gradient
from minibatch_choice import cross_entropy_error
from mnist_forward_batch import softmax


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W)


