import numpy as np

def l2_error(malha, u, p):

    error_squared = 0.0

    # Quadratura mais precisa
    points, weights = triangle_quadrature(3)

    for triangle in malha.triangles:

        element_nodes = malha.nodes[triangle]
        element_u = u[triangle]

        dN = shape_function_derivatives(
            0, 0, p
        )

        J = jacobian(
            element_nodes,
            dN
        )

        detJ = abs(np.linalg.det(J))

        for q in range(len(weights)):

            xi, eta = points[q]
            w = weights[q]

            N = shape_functions(
                xi,
                eta,
                p
            )

            xq, yq = xi_eta_to_xy(
                xi,
                eta,
                element_nodes,
                p
            )

            uh = N @ element_u

            uex = (
                np.sin(np.pi*xq)
                * np.sin(np.pi*yq)
            )

            error_squared += (
                (uex - uh)**2
                * detJ
                * w
            )

    return np.sqrt(error_squared)