import torch
import numpy as np

def function_derivative(func, u):
    y = func(u)
    y.backward()
    return u.grad.item()


def lambda_qut_regression(X, act_fun, hidden_dims = (20, ), n_samples=5000, mini_batch_size=500, alpha=0.05, option='quantile'):

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



def lambda_qut_classification(X, hat_p, act_fun, hidden_dims = (20, ), n_samples=5000, mini_batch_size=500, alpha=0.05, option='quantile'):
    offset = 0 if n_samples % mini_batch_size == 0 else 1
    n_samples_per_batch = n_samples // mini_batch_size + offset

    n, _ = X.shape
    fullList = torch.zeros((mini_batch_size * n_samples_per_batch,))
    num_classes = len(hat_p)

    for index in range(n_samples_per_batch):
        y_sample = torch.multinomial(hat_p, num_samples=n * mini_batch_size, replacement=True)
        y_sample = torch.nn.functional.one_hot(y_sample, num_classes=num_classes).float()
        y_sample = y_sample.view(n, mini_batch_size, num_classes)
        y_mean = y_sample.mean(dim=1, keepdim=True)
        y = (y_mean - y_sample).squeeze(1)

        # Perform einsum operation on the subsampled X
        xy = torch.einsum('ij,ikl->ijkl', X, y)
        xy_sum = xy.sum(dim=0).abs().sum(dim=2)
        xy_max = xy_sum.max(dim=0).values

        fullList[index * mini_batch_size:(index + 1) * mini_batch_size] = xy_max

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
