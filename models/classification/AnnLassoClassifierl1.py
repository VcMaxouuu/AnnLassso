from models.AnnLasso import AnnLassoClassifier

class AnnLassoClassifierl1(AnnLassoClassifier):
    def __init__(self, hidden_dims=(20, ), lambda_qut=None):
        super().__init__(hidden_dims=hidden_dims, penalty=1, lambda_qut=lambda_qut)
