"""Exercise 1: the neuron, the basic unit of a neural network.

A neuron multiplies each input by a weight, adds a bias, and passes the sum
through an activation function. Here the activation is the sigmoid, the same
one used by the logistic regression, which squeezes the result between 0
and 1.
"""
import numpy as np


class Neuron:
    def __init__(self, weight1, weight2, bias):
        self.weights_1 = weight1
        self.weights_2 = weight2
        self.bias = bias

    def feedforward(self, x1, x2):
        # 1. each input is multiplied by its weight
        # 2. the weighted inputs are added together with the bias
        total = x1 * self.weights_1 + x2 * self.weights_2 + self.bias
        # 3. the sum goes through the activation function
        y = 1 / (1 + np.exp(-total))
        return y


if __name__ == "__main__":
    neuron = Neuron(0, 1, 4)
    print(neuron.feedforward(2, 3))
