# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:49:56 2024

@author: Jin Li
"""

import numpy as np
np.random.seed(42)

class PerceptronSGD:
    def __init__(self, input_size, lr=0.1, epochs=1000):
        self.W = np.random.uniform(-1, 1, input_size + 1)
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        z = np.dot(x, self.W[1:]) + self.W[0]
        return self.sigmoid(z)

    def fit(self, X, d):
        for _ in range(self.epochs):
            predictions = self.predict(X)
            errors = d - predictions
            gradient_w = np.dot(X.T, errors * predictions * (1 - predictions))
            gradient_b = np.sum(errors * predictions * (1 - predictions))
            self.W[1:] += self.lr * gradient_w / X.shape[0]
            self.W[0] += self.lr * gradient_b / X.shape[0]

    def evaluate(self, X, d):
        predictions = self.predict(X)
        predictions = (predictions > 0.5).astype(int)
        accuracy = np.mean(predictions == d)
        return accuracy

# Logic functions
def logic_and(x):
    return np.all(x, axis=1) * 1

def logic_or(x):
    return np.any(x, axis=1) * 1

def logic_custom(x):
    return ((x[:,0] & ~x[:,1]) | (~x[:,0] & x[:,1])) & x[:,2]

# Training and testing
if __name__ == "__main__":
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    functions = {"AND": logic_and, "OR": logic_or, "CUSTOM": logic_custom}

    for name, func in functions.items():
        print(f"\nTraining and testing for logic function: {name}")
        d = func(X)  # Generate target outputs for the current logic function
        perceptron = PerceptronSGD(input_size=3, lr=0.19, epochs=10000)
        perceptron.fit(X, d)
        accuracy = perceptron.evaluate(X, d)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        for x, target in zip(X, d):
            pred = perceptron.predict(x)
            print(f"Input: {x}, Predicted: {pred:.2f}, Actual: {target}")

#%% 2

np.random.seed(42)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    return x * (1 - x)

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Weights and biases initialization
        self.weights_input_hidden = np.random.normal(0, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.normal(0, 1, (hidden_size, output_size))
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, x):
        """Forward pass."""
        self.hidden = sigmoid(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return output

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y_true in zip(X, y):
                x = x.reshape(1, -1)  # Reshape x to ensure it's a 2D array
                y_true = y_true.reshape(1, -1)  # Reshape y_true to ensure it's a 2D array
                
                # Forward pass
                y_pred = self.forward(x)
                
                # Backpropagation
                error = y_true - y_pred
                d_output = error * sigmoid_derivative(y_pred)
                
                error_hidden = np.dot(d_output, self.weights_hidden_output.T)
                d_hidden = error_hidden * sigmoid_derivative(self.hidden)
                
                # Update weights and biases
                self.weights_hidden_output += np.dot(self.hidden.T, d_output) * learning_rate
                self.bias_output += np.sum(d_output, axis=0) * learning_rate
                
                self.weights_input_hidden += np.dot(x.T, d_hidden) * learning_rate
                self.bias_hidden += np.sum(d_hidden, axis=0) * learning_rate

    def evaluate_accuracy(self, X, y):
        correct_predictions = 0
        total_predictions = len(X)
        
        for x, y_true in zip(X, y):
            y_pred = self.forward(x)
            predicted_class = np.round(y_pred)
            if predicted_class == y_true:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        return accuracy

# Logic function data
X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
              [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
y_and = np.array([[0], [0], [0], [0], [0], [0], [0], [1]])
y_or = np.array([[0], [1], [1], [1], [1], [1], [1], [1]])
y_custom = np.array([[0], [0], [0], [0], [1], [1], [0], [0]])

# Initialize and train the MLP for a specific logical function
mlp = SimpleMLP(input_size=3, hidden_size=2, output_size=1)
print("Training on and logic function")
mlp.train(X, y_and, epochs=10000, learning_rate=0.1)

# Evaluate accuracy
accuracy = mlp.evaluate_accuracy(X, y_and)
print(f"\nAccuracy on custom logic function: {accuracy * 100:.2f}%")

# Detailed testing output
print("\nTesting trained MLP:")
for x, y in zip(X, y_and):
    predicted = mlp.forward(x)
    print(f"Input: {x}, Predicted: {predicted.round()}, Actual: {y}")



# Repeat training process for OR and CUSTOM by reinitializing the MLP and using y_and and y_custom respectively.



