from models.AnnLasso import AnnLassoRegressor

class AnnLassoRegressorl0(AnnLassoRegressor):
    def __init__(self, hidden_dims=(20, ), lambda_qut=None):
        super().__init__(hidden_dims=hidden_dims, penalty=0, lambda_qut=lambda_qut)
