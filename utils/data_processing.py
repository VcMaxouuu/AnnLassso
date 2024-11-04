import numpy as np
import torch
import pandas as pd

class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims, correction=0)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)


def X_to_tensor(X):
    if isinstance(X, pd.DataFrame):
        X = torch.tensor(X.values, dtype= torch.float)
    elif isinstance(X, torch.Tensor):
        X = X.float()
    else:
        X = torch.tensor(X, dtype=torch.float)
    return X


def y_to_tensor(y):
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values.squeeze()
    if isinstance(y, (list, np.ndarray)):
        y = np.array(y).flatten()

    y_tensor = torch.tensor(y, dtype=torch.float)

    return y_tensor

def data_to_tensor(X, y):
    X = X_to_tensor(X)
    y = y_to_tensor(y)
    return X, y


def get_hat_p(y):
    if isinstance(y, torch.Tensor):
        y = y.to(torch.int64)
    else:
        y = torch.tensor(y, dtype=torch.int64)

    n_items = len(y)
    n_classes = len(y.unique())
    class_counts = torch.zeros(n_classes, dtype=torch.float)

    for class_index in range(n_classes):
        class_counts[class_index] = (y == class_index).sum()

    hat_p = class_counts / n_items
    return hat_p
