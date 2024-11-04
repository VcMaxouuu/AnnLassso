import torch
import numpy as np
from scipy.optimize import root_scalar, newton


def find_thresh(nu, lambda_):
    if nu is not None:
        '''Calculates and returns the threshold value.'''
        def func(kappa):
            return kappa**(2-nu) + 2*kappa + kappa**nu + 2*lambda_*(nu-1)

        kappa = root_scalar(func, bracket=[0, lambda_*(1-nu)/2]).root
        phi = kappa / 2 + lambda_ / (1 + kappa**(1-nu))
        return phi
    return lambda_


def _nonzerosol(u, lambda_, nu):
    '''Calculate the nonzero solution of the new penalty function.'''
    def func(x, y, lambda_, nu):
        return x - np.abs(y) + lambda_ * (1 + nu * x**(1-nu)) / (1 + x**(1-nu))**2

    def fprime(x, y, lambda_, nu):
        return 1 - lambda_ * (((1-nu)*(2+nu*x**(1-nu)-nu)) / (x**nu * (1+x**(1-nu))**3))
    try:
        root = newton(func, np.abs(u), fprime=fprime, args=(u, lambda_, nu), maxiter=500, tol=1e-5)
    except RuntimeError:
        print("Warning : Newton didn't converged")
        # If Newton-Raphson fails,
        root = np.abs(u)
    return root * np.sign(u)


def shrinkage_operator(u, lambda_, nu):
    if nu is not None:
        phi = find_thresh(nu, lambda_)
        sol = torch.zeros_like(u)
        abs_u = u.abs()
        ind = abs_u > phi

        if not ind.any():
            return sol

        sol_values = torch.tensor(
            _nonzerosol(u[ind].cpu().numpy(), lambda_.item(), nu),
            dtype=u.dtype,
            device=u.device
        )
        sol[ind] = sol_values
        return sol

    return u.sign() * torch.clamp(u.abs() - lambda_, min=0.0)
