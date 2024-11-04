import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ShiftedReLu(nn.Module):
    def __init__(self):
        super(ShiftedReLu, self).__init__()
        self.M = torch.tensor(20.0, dtype=torch.float32)
        self.u0 = torch.tensor(1.0, dtype=torch.float32)
        self.k = torch.tensor(1.0, dtype=torch.float32)

    def forward(self, x):
        f_0 = torch.nn.functional.softplus(self.u0, beta=self.M, threshold=20)
        f_u = torch.nn.functional.softplus(x + self.u0, beta=self.M, threshold=20)
        return 1 / self.k * (f_u.pow(self.k) - f_0.pow(self.k))


class PenalizedLoss(nn.Module):
    def __init__(self, lambda_, nu, type='reg'):
        super(PenalizedLoss, self).__init__()
        self.lambda_ = lambda_
        self.nu = nu
        self.type = type
        self.criterion = nn.CrossEntropyLoss(reduction='sum') if type == 'class' else nn.MSELoss(reduction='sum')

    def custom_penalty(self, penalized_tensor):
        epsilon = 1e-8
        if self.nu is not None:
            pow_term = torch.pow(torch.abs(penalized_tensor) + epsilon, 1 - self.nu)
            penalty = torch.sum(torch.abs(penalized_tensor) / (1 + pow_term))
        else:
            penalty = torch.sum(torch.abs(penalized_tensor))
        return penalty

    def forward(self, input, target, penalized_tensors):
        if self.type == 'class':
            loss = self.criterion(input, target)
        else:
            loss = torch.sqrt(self.criterion(input, target))

        penalty = torch.tensor(0.0, device=input.device, requires_grad=True)
        for _, param in penalized_tensors:
            penalty = penalty + self.custom_penalty(param)

        penalized_loss = loss + self.lambda_ * penalty
        return penalized_loss, loss



def nonlinearity_index(model, return_graph=False, ax=None):
    X = model.X

    def phi(x):
        return torch.where(x <= 0.5, 2 * x, -2 * x + 2)

    active_neurons = torch.nonzero(~torch.all(model.layers[0].weight == 0, dim=1), as_tuple=True)[0]
    active_features = model.important_features()[1]
    pruned_layers = {}
    pruned_X = X[:, active_features]
    for i, layer in enumerate(model.layers):
        pruned_layers[i] = {}
        if i==0:
            # Prune the first layer
            pruned_weight = layer.weight.detach().clone()[active_neurons, :][:, active_features]
            pruned_bias = layer.bias.detach().clone()[active_neurons]
        else:
            pruned_weight = F.normalize(layer.weight.detach().clone(), p=2, dim=1)
            if i==1:
                # Prune second layer
                pruned_weight = pruned_weight[:, active_neurons]
            pruned_bias = layer.bias.detach().clone()

        pruned_layers[i]['weight'] = pruned_weight
        pruned_layers[i]['bias'] = pruned_bias

    indexes = []
    V = pruned_X

    for i in range(len(pruned_layers) - 1):  # Exclude the output layer
        layer = pruned_layers[i]
        weight = layer['weight']
        bias = layer['bias']

        U = torch.matmul(V, weight.T)
        # Calculate the proportion of inputs that activate each neuron
        smaller_count = ((U < -bias - 1).sum(dim=0)).float() / X.shape[0]
        index = phi(smaller_count)
        indexes.append(index.tolist())

        V = model.act_fun(bias + U)

    if return_graph:
        # If ax is not provided, create a new figure and axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        ax.boxplot(indexes)
        ax.set_title("Nonlinearity Index for Each Hidden Layer")
        ax.set_ylabel("Value")
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.05, 1.05)
        # If ax was created internally, show the plot
        if ax is None:
            plt.show()
        return

    return indexes, pruned_layers
