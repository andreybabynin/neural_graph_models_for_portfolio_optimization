import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


def base_optim(n_y, n_obs, prisk, max_weight=0.25):
    # Variables
    z = cp.Variable((n_y, 1), nonneg=True)  # weights
    c_aux = cp.Variable()
    obj_aux = cp.Variable(n_obs)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)

    # Constraints
    constraints = [cp.sum(z) == 1, mu_aux == y_hat @ z, z <= max_weight]

    for i in range(n_obs):
        constraints += [obj_aux[i] >= prisk(z, c_aux, ep[i])]

    # Objective function
    objective = cp.Minimize((1 / n_obs) * cp.sum(obj_aux) - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma], variables=[z])
