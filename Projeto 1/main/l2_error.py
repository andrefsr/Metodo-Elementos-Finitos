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

    r = np.sqrt(x**2 + y**2)

    denominator = (
        np.log(c/a)/er1
        +
        np.log(b/c)/er2
    )

    if a <= r <= c:

        return (
            np.log(r/a)/er1
        ) / denominator

    elif c < r <= b:

        return (
            np.log(c/a)/er1
            +
            np.log(r/c)/er2
        ) / denominator

    else:

        raise ValueError(
            f"Ponto fora do domínio: r = {r}"
        )

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
    quadrature = triangle_quadrature(p)

    # Percorre todos os elementos
    for element in mesh.triangles:

        # Coordenadas dos nós do elemento
        element_nodes = mesh.nodes[element]

        # Valores de u nos nós do elemento
        element_u = u[element]

        # Integração no elemento de referência
        for xi, eta, w in quadrature:

            # Funções de forma
            N = shape_functions(xi, eta, p)

            # Coordenadas físicas do ponto de quadratura
            xq, yq = xi_eta_to_xy(
                xi,
                eta,
                element_nodes,
                p
            )

            # Solução numérica no ponto
            uh = N @ element_u

            # Solução exata no ponto
            uex = exact_solution(xq, yq)

            # Jacobiano
            J = jacobian(
                xi,
                eta,
                element_nodes,
                p
            )

            detJ = abs(np.linalg.det(J))

            # Contribuição para a norma L2 ao quadrado
            error_squared += (
                (uex - uh)**2
                * detJ
                * w
            )

    return np.sqrt(error_squared)