"""Exercise 5: the same network, adapted to a regression.

The sigmoid squeezes its output between 0 and 1, which is what a probability
needs. A grade is not a probability, so the neuron of the output layer uses
the identity function instead: f(x) = x, the sum is returned as it is. The
neurons of the hidden layer keep the sigmoid.
"""
import numpy as np

STUDENTS = {
    #          math, chemistry, physics
    "Bob": (12, 15, 16),
    "Eli": (10, 9, 10),
    "Tom": (18, 18, 19),
    "Ryan": (13, 14, 16),
}


class Neuron:
    def __init__(self, weight1, weight2, bias, regression):
        self.weights_1 = weight1
        self.weights_2 = weight2
        self.bias = bias
        self.regression = regression

    def feedforward(self, x1, x2):
        total = x1 * self.weights_1 + x2 * self.weights_2 + self.bias
        if self.regression:
            # the identity function returns the sum as it is
            return total
        return 1 / (1 + np.exp(-total))


class OurNeuralNetwork:
    def __init__(self, neuron_h1, neuron_h2, neuron_o1):
        self.h1 = neuron_h1
        self.h2 = neuron_h2
        self.o1 = neuron_o1

    def feedforward(self, x1, x2):
        out_h1 = self.h1.feedforward(x1, x2)
        out_h2 = self.h2.feedforward(x1, x2)
        return self.o1.feedforward(out_h1, out_h2)


# only the neuron of the output layer uses the identity function
neuron_h1 = Neuron(0.05, 0.001, 0, regression=False)
neuron_h2 = Neuron(0.002, 0.003, 0, regression=False)
neuron_o1 = Neuron(2, 7, 10, regression=True)

network = OurNeuralNetwork(neuron_h1, neuron_h2, neuron_o1)

# 2. the physics grade predicted for each student
outputs = {name: network.feedforward(math, chemistry)
           for name, (math, chemistry, _) in STUDENTS.items()}

# 3. the error of those predictions
y_true = np.array([physics for _, _, physics in STUDENTS.values()])
y_pred = np.array(list(outputs.values()))
mse = np.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    # 1. a single neuron with the identity function: 2*0 + 3*1 + 4 = 7
    neuron = Neuron(0, 1, 4, True)
    print("1.", neuron.feedforward(2, 3), end="\n\n")

    print("2.")
    for name, output in outputs.items():
        print(f"{name}: {output}")

    print("\n3. MSE:", mse)
