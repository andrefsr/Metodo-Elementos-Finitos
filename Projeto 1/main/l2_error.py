import numpy as np
from main.Matrices import triangle_quadrature
from main.shape_functions import shape_functions
from main.Matrices import jacobian
from main.geometry import xi_eta_to_xy



def exact_coaxial(x, y):

    a = 2e-3
    c = 5e-3
    b = 8e-3
    er1 = 2.0
    er2 = 4.0

    x = np.asarray(x)
    y = np.asarray(y)

    r = np.sqrt(x**2 + y**2)

    denominator = (
        np.log(c/a)/er1
        +
        np.log(b/c)/er2
    )

    u = np.zeros_like(r, dtype=float)

    mask1 = (r >= a) & (r <= c)
    mask2 = (r > c) & (r <= b)

    u[mask1] = (
        np.log(r[mask1]/a)/er1
    ) / denominator

    u[mask2] = (
        np.log(c/a)/er1
        +
        np.log(r[mask2]/c)/er2
    ) / denominator

    # Se a entrada for escalar, retorna escalar
    if u.ndim == 0:
        return float(u)

    return u

def l2_error(malha, u, p):
    error_squared = 0.0
    # Quadratura mais precisa
    points, weights = triangle_quadrature(3)

    for triangle in malha.triangles:

        element_nodes = malha.nodes[triangle]
        element_u = u[triangle]

        _, dN = shape_functions(0,0,p)


        for q in range(len(weights)):

            xi, eta = points[q]
            w = weights[q]

            J = jacobian(xi,eta,element_nodes,p)
            detJ = abs(np.linalg.det(J))

            N, _ = shape_functions(xi,eta,p)

            xq, yq = xi_eta_to_xy(xi,eta,element_nodes,p)

            uh = N @ element_u

            uex = (np.sin(np.pi*xq) * np.sin(np.pi*yq))

            error_squared += ((uex - uh)**2 * detJ * w)

    return np.sqrt(error_squared)



def l2_error_coaxi(mesh, u, p, exact_solution):

    error_squared = 0.0

    # Pontos e pesos da quadratura
    points, weights = triangle_quadrature(8)

    # Percorre os elementos
    for element in mesh.triangles:

        # Coordenadas dos nós do elemento
        element_nodes = mesh.nodes[element]

        # Valores de u nos nós do elemento
        element_u = u[element]

        # Integração
        for (xi, eta), w in zip(points, weights):

            # Funções de forma
            N, _ = shape_functions(xi, eta, p)

            # Coordenadas físicas
            xq, yq = xi_eta_to_xy(
                xi,
                eta,
                element_nodes,
                p
            )

            # Solução numérica
            uh = N @ element_u

            # Solução exata
            uex = exact_solution(xq, yq)

            # Jacobiano
            J = jacobian(
                xi,
                eta,
                element_nodes,
                p
            )

            detJ = abs(np.linalg.det(J))

            # Erro
            error_squared += (
                (uex - uh)**2
                * detJ
                * w
            )

    return np.sqrt(error_squared)