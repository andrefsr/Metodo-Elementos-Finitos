import numpy as np
from shape_functions import shape_functions
from geometry import jacobian

def triangle_quadrature(p):

    if p == 1:
        points = np.array([[1/3, 1/3]])
        weights = np.array([1/2])

    elif p == 2:
        points = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
        weights = np.array([1/6, 1/6, 1/6])

    elif p == 3:
        points = np.array([ [0.445948490915965, 0.108103018168070], [0.108103018168070, 0.445948490915965],
                            [0.445948490915965, 0.445948490915965], [0.091576213509771, 0.816847572980459],
                            [0.816847572980459, 0.091576213509771], [0.091576213509771, 0.091576213509771]])
        weights = np.array([0.111690794839005, 0.111690794839005, 0.111690794839005, 0.054975871827661, 0.054975871827661, 0.054975871827661])

    else:
        raise ValueError("Quadratura implementada apenas para p = 1, 2 e 3.")

    return points, weights

def element_stiffness(nodes,p,k=1.0):
    n = len(nodes)
    Ke = np.zeros((n,n))

    points, weights = triangle_quadrature(p)

    for q in range(len(weights)):

        xi, eta = points[q]
        w = weights[q]

        _, dN = shape_functions(xi,eta,p)

        J = jacobian(xi,eta,nodes,p)

        detJ = np.linalg.det(J)

        dN_xy = dN @ np.linalg.inv(J).T

        B = dN_xy.T

        Ke += k* (B.T @ B) * abs(detJ) * w

    return Ke

