# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:40:23 2024

@author: justd
"""

from scipy.special import expit

def update_weights(x1, x2, y, w1, w2, w0, lr):
    # Compute the output
    z = w1 * x1 + w2 * x2 + w0
    y_hat = expit(z)  # Using scipy's expit for the sigmoid function
    
    # Compute gradients
    nabla_w1 = (y_hat - y) * x1
    nabla_w2 = (y_hat - y) * x2
    nabla_w0 = (y_hat - y)
    
    # Update weights and bias
    w1 -= lr * nabla_w1
    w2 -= lr * nabla_w2
    w0 -= lr * nabla_w0
    
    return w1, w2, w0

import numpy as np
# training data: 4 observations, each with 2 features and a binary label
X_train = np.array([[1, 2], [2, 3], [3, 5], [4, 2]])
y_train = np.array([0, 1, 1, 0])

# Initialize weights and bias
w1, w2, w0 = 0.01, -0.02, 0.0  # Small random values or zeros

# Learning rate
lr = 0.1

# Update weights and bias for each observation in the training data
for x1, x2, y in zip(X_train[:, 0], X_train[:, 1], y_train):
    w1, w2, w0 = update_weights(x1, x2, y, w1, w2, w0, lr)

print(f"Updated weights: w1 = {w1}, w2 = {w2}, w0 = {w0}")
