from models.AnnLasso import AnnLassoClassifier

class AnnLassoClassifierl0(AnnLassoClassifier):
    def __init__(self, hidden_dims=(20, ), lambda_qut=None):
        super().__init__(hidden_dims=hidden_dims, penalty=0, lambda_qut=lambda_qut)
