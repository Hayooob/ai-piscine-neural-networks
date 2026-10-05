"""Exercise 3: the log loss, the loss used for classification.

The MSE measures how far a prediction is from a number. The log loss
measures how far a predicted probability is from the true class, and
punishes a confident mistake much more than a hesitant one.
"""
import numpy as np
from sklearn.metrics import log_loss

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.1, 0.8, 0.6, 0.5, 0.3])


def log_loss_custom(y_true, y_pred):
    """- 1/n * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))."""
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


custom = log_loss_custom(y_true, y_pred)
sklearn_loss = log_loss(y_true, y_pred)

if __name__ == "__main__":
    print("log_loss_custom:        ", custom)
    print("log_loss from sklearn:  ", sklearn_loss)
    print("they match:", np.isclose(custom, sklearn_loss))
