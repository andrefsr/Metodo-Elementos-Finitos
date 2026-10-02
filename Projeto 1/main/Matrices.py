import numpy as np
from main.shape_functions import shape_functions
from main.geometry import jacobian, xi_eta_to_xy

def triangle_quadrature(nq=8):

    # Gauss-Legendre em [−1,1]
    x, w = np.polynomial.legendre.leggauss(nq)

    # transforma para [0,1]
    s = 0.5 * (x + 1.0)
    ws = 0.5 * w

    points = []
    weights = []

    for i in range(nq):

        for j in range(nq):

            si = s[i]
            tj = s[j]

            # Transformação de Duffy
            xi = si
            eta = (1.0 - si) * tj

            # Jacobiano da transformação
            weight = (ws[i] * ws[j] * (1.0 - si))

            points.append([xi,eta])
            weights.append(weight)

    return (np.array(points),np.array(weights))

def element_stiffness(nodes,p,k=1,eps_r=1):
    n = len(nodes)
    Ke = np.zeros((n, n))

    points, weights = triangle_quadrature(8)

    for q in range(len(weights)):

        xi, eta = points[q]
        w = weights[q]

        _, dN = shape_functions(xi,eta,p)

        J = jacobian(xi,eta,nodes,p)
        detJ = abs(np.linalg.det(J))

        dN_xy = (dN @ np.linalg.inv(J))

        B = dN_xy.T

        Ke += k * (eps_r * B.T @ B * detJ * w)

    return Ke

def element_load(nodes, p, f):
    n = len(nodes)
    Fe = np.zeros(n)

    points, weights = triangle_quadrature(8)

    for q in range(len(weights)):
        xi, eta = points[q]
        w = weights[q]

        N, _ = shape_functions(xi,eta,p) ##funções de forma e suas derivadas

        J = jacobian(xi,eta,nodes,p)
        detJ = np.linalg.det(J)

        x, y = xi_eta_to_xy(xi,eta,nodes,p) ##coordenadas físicas do ponto de Gauss

        fq = f(x,y) ##fonte

        Fe += N*fq*abs(detJ)*w

    return Fe