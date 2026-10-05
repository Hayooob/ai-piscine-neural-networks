"""Exercise 4: forward propagation on a tiny data set.

The network reads the math and chemistry grades of a student and outputs the
probability of passing the exam. The log loss then says how far those
probabilities are from the truth.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "ex01"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "ex02"))

from neural_network import OurNeuralNetwork  # noqa: E402
from neuron import Neuron  # noqa: E402

STUDENTS = {
    #          math, chemistry, exam_success
    "Bob": (12, 15, 1),
    "Eli": (10, 9, 0),
    "Tom": (18, 18, 1),
    "Ryan": (13, 14, 1),
}

neuron_h1 = Neuron(0.05, 0.001, 0)
neuron_h2 = Neuron(0.02, 0.003, 0)
neuron_o1 = Neuron(2, 0, 0)

network = OurNeuralNetwork(neuron_h1, neuron_h2, neuron_o1)

# 1. the output of the network for each student
outputs = {name: network.feedforward(math, chemistry)
           for name, (math, chemistry, _) in STUDENTS.items()}

# 2. how far those predictions are from the truth
y_true = np.array([success for _, _, success in STUDENTS.values()])
y_pred = np.array(list(outputs.values()))
logloss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

if __name__ == "__main__":
    for name, output in outputs.items():
        print(f"{name}: {output}")
    print("\nlogloss:", logloss)
