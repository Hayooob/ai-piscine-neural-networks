"""Exercise 2: three neurons connected together.

h1 and h2 form the hidden layer: they both read the inputs. o1 is the output
layer, and it reads the outputs of h1 and h2, not the inputs.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "ex01"))

from neuron import Neuron  # noqa: E402


class OurNeuralNetwork:
    def __init__(self, neuron_h1, neuron_h2, neuron_o1):
        self.h1 = neuron_h1
        self.h2 = neuron_h2
        self.o1 = neuron_o1

    def feedforward(self, x1, x2):
        out_h1 = self.h1.feedforward(x1, x2)
        out_h2 = self.h2.feedforward(x1, x2)
        # The inputs for o1 are the outputs from h1 and h2
        y = self.o1.feedforward(out_h1, out_h2)
        return y


neuron_h1 = Neuron(1, 2, -1)
neuron_h2 = Neuron(0.5, 1, 0)
neuron_o1 = Neuron(2, 0, 1)

network = OurNeuralNetwork(neuron_h1, neuron_h2, neuron_o1)

if __name__ == "__main__":
    print(network.feedforward(2, 3))
