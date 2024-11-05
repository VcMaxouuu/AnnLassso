import torch
import numpy as np

def function_derivative(func, u):
    y = func(u)
    y.backward()
    return u.grad.item()


def lambda_qut_regression(X, act_fun, hidden_dims = (20, ), n_samples=5000, alpha=0.05, option='quantile'):

    n, _ = X.shape
    fullList = torch.zeros(n_samples)

    for index in range(n_samples):
        y_sample = torch.normal(mean=0., std=1, size=(n, 1))
        y = y_sample - (torch.mean(y_sample, dim=0))
        xy = torch.matmul(X.T, y)
        xy_max = torch.amax(torch.sum(torch.abs(xy), dim=1), dim=0)
        norms = torch.norm(y, p=2, dim=0)

        fullList[index]= xy_max/norms

    if act_fun is not None: # None for linear and 'act_fun' for neural network
        if len(hidden_dims) == 1: pi_l = 1
        else: pi_l = np.sqrt(np.prod(hidden_dims[1:]))

        sigma_diff = function_derivative(act_fun, torch.tensor(0, dtype=torch.float, requires_grad=True)) ** len(hidden_dims)

        fullList *= pi_l * sigma_diff

    if option=='full':
        return fullList
    elif option=='quantile':
        return torch.quantile(fullList, 1-alpha)
    else:
        pass


def lambda_qut_classification(X, hat_p, act_fun, hidden_dims = (20, ), n_samples=5000, alpha=0.05, option='quantile'):
    n, _ = X.shape
    fullList = torch.zeros(n_samples)

    for index in range(n_samples):
        y_sample = torch.multinomial(hat_p, n, replacement=True)
        y_sample = torch.nn.functional.one_hot(y_sample).type(X.dtype)
        y = y_sample - torch.mean(y_sample, axis=0)
        xy = torch.matmul(X.T, y)
        xy_sum = torch.sum(torch.abs(xy), axis=1)
        xy_max = torch.amax(xy_sum)

        fullList[index] = xy_max

    if act_fun is not None: # None for linear and 'act_fun' for neural network
        if len(hidden_dims) == 1: pi_l = 1
        else: pi_l = np.sqrt(np.prod(hidden_dims[1:]))

        sigma_diff = function_derivative(act_fun, torch.tensor(0, dtype=torch.float, requires_grad=True)) ** len(hidden_dims)

        fullList = fullList * pi_l * sigma_diff

    if option == 'full':
        return fullList
    elif option == 'quantile':
        return torch.quantile(fullList, 1-alpha)
    else:
        pass
