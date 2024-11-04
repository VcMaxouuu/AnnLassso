from lassonet import LassoNetClassifier as LC
from sklearn.preprocessing import StandardScaler
import numpy as np
from lassonet import LassoNetClassifierCV

class LNClassifier():
    def __init__(self, p2=20, cv_lambda=None):
        self.model = LC(hidden_dims = (p2,), verbose=0)
        self.cv_lambda = cv_lambda

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        if self.cv_lambda is None:
            LassoModel = LassoNetClassifierCV(verbose=0, cv=10)
            LassoModel.path(X, y)
            self.cv_lambda = LassoModel.best_lambda_

        lambda_seq = [self.cv_lambda * (np.exp(i-1) / (1 + np.exp(i-1)) if i < 5 else 1) for i in range(6)]
        self.path = self.model.path(X, y, lambda_seq=lambda_seq)
        self.imp_feat = self._important_features()

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def _important_features(self):
        indices = np.nonzero(self.path[-1].selected.numpy())[0]
        return len(indices), sorted(indices)