import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
import utils

class AnnLassoRegressor(nn.Module):
    """Model with non linear penalty used for regression"""
    def __init__(self, hidden_dims=(20,), penalty=0, lambda_qut=None):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.penalty = penalty
        self.lambda_qut = lambda_qut
        self.act_fun = utils.ShiftedReLu()

        # Parameters to be determined during training
        self.imp_feat = None
        self.scaler = None

        # Model layers
        self.layers = nn.ModuleList()

    def _build_layers(self, input_dim):
        self.layers.append(nn.Linear(input_dim, self.hidden_dims[0]))

        for i in range(1, len(self.hidden_dims)+1):
            output = 1 if i == len(self.hidden_dims) else self.hidden_dims[i]
            layer = nn.Linear(self.hidden_dims[i - 1], output)
            self.layers.append(layer)


    def penalized_parameters(self):
        penalized_params = []
        last_dim = len(self.hidden_dims)
        for name, param in list(self.named_parameters())[:-1]:
            if f'layers.{last_dim}' in name:
                continue
            if 'layers.0.weight' in name or 'bias' in name:
                penalized_params.append((name, param))

        return penalized_params

    def unpenalized_parameters(self):
        unpenalized_params = []
        last_dim = len(self.hidden_dims)
        for name, param in list(self.named_parameters()):
            if f'layers.{last_dim}' in name:
                unpenalized_params.append((name, param))
                continue
            if 'layers.0.weight' not in name and 'bias' not in name:
                unpenalized_params.append((name, param))

        return unpenalized_params


    def forward(self, X):
        output = X
        for i, layer in enumerate(self.layers):
            if i > 0:
                normalized_weight = F.normalize(layer.weight, p=2, dim=1)
                output = F.linear(output, normalized_weight, layer.bias)
            else:
                output = layer(output)

            if i < len(self.layers) - 1:
                output = self.act_fun(output)
        return output.squeeze()


    def predict(self, X):
        self.eval()
        X = utils.X_to_tensor(X)
        X = self.scaler.transform(X)
        with torch.no_grad():
            output = self.forward(X)
        return output.cpu().numpy()


    def fit(self, X, y, verbose=False):
        self.scaler = utils.StandardScaler()
        X, y = utils.data_to_tensor(X, y)
        X = self.scaler.fit_transform(X)
        self.X = X

        input_dim = X.shape[1]
        self._build_layers(input_dim)
        for _, p in self.penalized_parameters():
            nn.init.normal_(p, mean = 0.0, std = y.std())
        if self.lambda_qut is None:
            self.lambda_qut = utils.lambda_qut_regression(X, self.act_fun, self.hidden_dims)

        self._train_loop(X, y, verbose)
        self.imp_feat = self.important_features()

    def _train_loop(self, X, y, verbose):
        self.train()
        utils.sgd(self, X, y, 0.01, torch.tensor(0), 1, 1e-3, verbose)
        for i in range(6):
            nu = None if self.penalty == 1 else [1, 0.7, 0.4, 0.3, 0.2, 0.1][i]
            lambda_i = self.lambda_qut * (np.exp(i - 1) / (1 + np.exp(i - 1)) if i < 5 else 1)
            rel_err = 1e-6 if i<5 else 1e-10
            lr = 0.01/(2*(i+1))
            if verbose:
                if nu is not None:
                    print(f"Lambda = {lambda_i.item():.4f} -- Nu = {nu}")
                else:
                    print(f"Lambda = {lambda_i.item():.4f}")

            if i<5:
                utils.sgd(self, X, y, lr, lambda_i, nu, rel_err, verbose)
            else:
                utils.training(self, X, y, lambda_i, nu, rel_err, verbose)

    def important_features(self):
        weight = self.layers[0].weight.data
        non_zero_columns = torch.any(weight != 0, dim=0)
        count = torch.count_nonzero(non_zero_columns).item()
        indices = torch.where(non_zero_columns)[0].tolist()
        return count, sorted(indices)


    def save(self, filepath):
        save_dict = {
            'state_dict': self.state_dict(),
            'scaler': self.scaler,
            'lambda_qut': self.lambda_qut,
            'training set': self.X
        }
        torch.save(save_dict, filepath)

    def load(self, filepath, strict=True):
        checkpoint = torch.load(filepath)

        hidden_dims = []
        for layer_name, layer_param in list(checkpoint['state_dict'].items())[:-1]:
            if 'bias' in layer_name:
                hidden_dims.append(layer_param.shape[0])
        self.hidden_dims = tuple(hidden_dims)
        input_dim = checkpoint['state_dict']['layers.0.weight'].shape[1]
        self._build_layers(input_dim)

        self.load_state_dict(checkpoint['state_dict'], strict=strict)
        self.scaler = checkpoint.get('scaler', None)
        self.lambda_qut = checkpoint.get('lambda_qut', None)
        self.X = checkpoint.get('training set', None)
