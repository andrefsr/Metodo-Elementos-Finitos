import numpy as np

def apply_dirichlet(K, F, dirichlet_dofs, dirichlet_values):

    all_dofs = np.arange(len(F))
    free_dofs = np.setdiff1d(all_dofs, dirichlet_dofs)
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfc = K[np.ix_(free_dofs, dirichlet_dofs)]
    Ff = (F[free_dofs] - Kfc @ dirichlet_values)

    return Kff, Ff, free_dofs