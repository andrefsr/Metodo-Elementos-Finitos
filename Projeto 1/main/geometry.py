import numpy as np
from shape_functions import shape_functions

def xi_eta_to_xy(xi, eta, nodes, p):

    N, _ = shape_functions(xi, eta, p)

    x = N @ nodes[:, 0]
    y = N @ nodes[:, 1]

    return x, y

def jacobian(xi, eta, nodes, p):

    _, dN = shape_functions(xi,eta,p)

    dx_dxi = np.sum(dN[:, 0] * nodes[:, 0])
    dx_deta = np.sum(dN[:, 1] * nodes[:, 0])

    dy_dxi = np.sum(dN[:, 0] * nodes[:, 1])
    dy_deta = np.sum(dN[:, 1] * nodes[:, 1])

    J = np.array([
        [dx_dxi,  dx_deta],
        [dy_dxi,  dy_deta]
    ])

    return J