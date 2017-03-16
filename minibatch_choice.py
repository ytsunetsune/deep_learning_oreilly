# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = \
            load_mnist(normalize=True, one_hot_label=False)

    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    print(batch_mask)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

