import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import root_scalar, newton
import torch

import utils

def FISTAsqrtLASSO(X, y, lambda_, tol=1.e-11):
    gamma = 0.5
    p = X.shape[1]
    n = X.shape[0]
    alpha = np.zeros(p)
    alpha_int = np.mean(y)
    ym = y - alpha_int
    g = -np.dot(X.T, ym) / np.sqrt(np.sum(ym**2)).squeeze()
    lambda0 = np.max(np.abs(g))
    maxepochs = 1000
    lambdas = []
    ALPHAs = np.empty((p, 6))
    for i in range(6):
        lambdai = lambda_ * (np.exp(i-1) / (1 + np.exp(i-1)) if i < 5 else 1)
        lambdas.append(lambdai)
        continue1 = True
        ypred = np.dot(X, alpha)
        cost = np.sqrt(np.sum((y - alpha_int - ypred)**2)) + lambdai * np.sum(np.abs(alpha))
        costs = [cost]
        epoch = 0
        while continue1:
            continue2 = True
            ym = y - alpha_int
            g = np.dot(X.T, (ypred - ym)) / np.sqrt(np.sum((ym - ypred)**2))
            j = 0
            while continue2:
                gammaj = gamma**j
                uj = alpha - gammaj * g
                lambdaj = gammaj * lambdai
                solj = utils.shrinkage_operator(torch.tensor(uj), torch.tensor(lambdaj), None).numpy()
                ypredj = np.dot(X, solj)
                resj = y - ypredj
                alpha_intj = np.mean(resj)
                costnew = np.sqrt(np.sum((resj - alpha_intj)**2)) + lambdai * np.sum(np.abs(solj))
                if costnew > cost:
                    j += 1
                else:
                    continue2 = False
                    diff = cost - costnew
                    cost = costnew
                    ypred = ypredj
                    alpha = solj
                    alpha_int = alpha_intj

            costs.append(cost)
            continue1 = ((diff / costnew) > tol)

            epoch += 1
            if epoch >= maxepochs:
                continue1 = False
        ALPHAs[:, i] = alpha

    alphahat = ALPHAs[:, -1]

    result = {
        "ALPHAs": ALPHAs,
        "alphahat": alphahat,
        "lambdas": lambdas,
        "alpha_int": alpha_int,
        "lambda0": lambda0,
    }

    return result



class LinearModell1():
    def __init__(self, lambda_qut = None):
        self.lambda_qut = lambda_qut
        self.alpha = None

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        if self.lambda_qut is None:
            self.lambda_qut = utils.lambda_qut_regression(utils.X_to_tensor(X), None).item()
        if isinstance(self.lambda_qut, torch.Tensor):
            self.lambda_qut = self.lambda_qut.item()
        self.alpha = FISTAsqrtLASSO(X, y, self.lambda_qut)['alphahat']
        self.imp_feat = self._important_features()

    def predict(self, X):
        X = self.scaler.transform(X)
        y_pred = np.dot(X, self.alpha)
        return y_pred

    def _important_features(self):
        indices = np.nonzero(self.alpha)[0]
        return len(indices), sorted(indices)
