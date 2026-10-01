import numpy as np
from main.Matrices import element_stiffness, element_load


def assemble_system(nodes, elements, p, f, k=1.0,eps_r=1.0):

    n_nodes = len(nodes)

    K = np.zeros((n_nodes,n_nodes))
    F = np.zeros(n_nodes)

    for connectivity in elements:
        element_nodes = nodes[connectivity] ##coordenadas dos nós do elemento

        Ke = element_stiffness(element_nodes,p,k,eps_r) ##matriz local
        Fe = element_load(element_nodes,p,f)

        #montagem
        for i, I in enumerate(connectivity):
            F[I] += Fe[i]
            for j, J in enumerate(connectivity):
                K[I,J] += Ke[i,j]

    return K, F

def assemble_coax_system(nodes,triangles,element_material,p):

    n_nodes = len(nodes)

    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    eps_r = {1: 2.0,2: 4.0}

    for e, connectivity in enumerate(triangles):

        element_nodes = nodes[connectivity]
        material = element_material[e]

        eps = eps_r[material]

        Ke = element_stiffness(element_nodes,p,eps)     

        for i, I in enumerate(connectivity):
            for j, J in enumerate(connectivity):
                K[I, J] += Ke[i, j]

    return K, F

