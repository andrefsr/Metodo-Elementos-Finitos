import numpy as np
from Matrices import element_stiffness, element_load


def assemble_system(nodes, elements, p, f, k=1.0):

    n_nodes = len(nodes)

    K = np.zeros((n_nodes,n_nodes))
    F = np.zeros(n_nodes)

    for connectivity in elements:
        element_nodes = nodes[connectivity] ##coordenadas dos nós do elemento

        Ke = element_stiffness(element_nodes,p,k) ##matriz local
        Fe = element_load(element_nodes,p,f)

        #montagem
        for i, I in enumerate(connectivity):
            F[I] += Fe[i]
            for j, J in enumerate(connectivity):
                K[I,J] += Ke[i,j]

    return K, F