import cvxpy as cp
import torch


def sharpe_loss(w, fut_ret, cov):
    loss = -(w @ fut_ret) / torch.sqrt(w @ cov @ w.T)
    return loss


def max_return(w, fut_ret, cov):
    loss = -(w @ fut_ret)
    return loss


def p_var(z, c, x):
    """Variance
    Inputs
    z: (n x 1) vector of portfolio weights (decision variable)
    c: Scalar. Centering parameter that serves as a proxy to the expected value (auxiliary variable)
    x: (n x 1) vector of realized returns (data)

    Output: Single squared deviation.
    Note: This function is only one component of the portfolio variance, and must be aggregated
    over all scenarios 'x' to recover the complete variance
    """
    return cp.square(x @ z - c)
