# Neural networks

Solutions for the Neural networks day of the AI piscine: a neuron, a small
network built with three of them, the log loss, the forward propagation, and
the change needed to turn the network into a regression.

## Structure

```
.
├── ex00/
│   ├── requirements.txt
│   └── Notebook_ex00.ipynb
├── ex01/neuron.py
├── ex02/neural_network.py
├── ex03/log_loss.py
├── ex04/forward_propagation.py
└── ex05/regression.py
```

## Virtual environment (ex00)

```bash
python -m venv env/ex00
source env/ex00/Scripts/activate      # Windows (Git Bash); Linux/macOS: source env/ex00/bin/activate
pip install -r ex00/requirements.txt
python --version
python -c "import jupyter, numpy"
```

Tested with Python 3.13 and NumPy 2.5.3. `scikit-learn` is only used in ex03
to check the custom log loss against `sklearn.metrics.log_loss`.

## Running an exercise

```bash
python ex04/forward_propagation.py
```

Everything runs instantly: there is no data set and no training here, only
the forward pass.

## How a neuron works

A neuron does three things:

1. multiply each input by a weight: `x1 * w1`, `x2 * w2`
2. add them together with a bias: `x1*w1 + x2*w2 + b`
3. pass the sum through an activation function: `y = f(x1*w1 + x2*w2 + b)`

The activation is what makes the neuron non linear. Without it, stacking
neurons would be pointless: a chain of linear functions is still a linear
function.

A **network** is only neurons connected together. In ex02, `h1` and `h2`
form the hidden layer and both read the inputs; `o1` is the output layer and
reads the outputs of `h1` and `h2`. The input layer holds no neuron, it is
just the data.

## Results

- **ex01** — `Neuron(0, 1, 4).feedforward(2, 3)` returns
  **0.9990889488055994**.
- **ex02** — the network returns **0.9524917424084265**.
- **ex03** — `log_loss_custom` returns **0.5472899351247816**, the same
  value as `log_loss` from Scikit-learn.
- **ex04** — the probabilities of passing the exam are
  `Bob: 0.7855`, `Eli: 0.7772`, `Tom: 0.8068`, `Ryan: 0.7892`, for a log
  loss of **0.5485133607757963**.
- **ex05** — the neuron with the identity function returns **7**, the
  predicted physics grades are `Bob: 14.92`, `Eli: 14.83`, `Tom: 15.09`,
  `Ryan: 14.94`, and the MSE is **10.237608699909138**.

All of them match the audit to the last digit.

## Classification or regression

The sigmoid is used on the output of a **classification** because it
squeezes the result between 0 and 1, which is what a probability needs. A
grade is not a probability, so for a **regression** the output neuron uses
the identity function `f(x) = x` and returns the sum as it is. Only the
output layer changes, the hidden neurons keep their sigmoid.

The weights here are given, not learnt, which is why the results are poor:
in ex04 the network answers around 0.78 for everyone, including Eli who
failed, and in ex05 it predicts about 15 for every student, an MSE of 10.
Training the network would consist in correcting these weights, which is the
job of the backpropagation.
