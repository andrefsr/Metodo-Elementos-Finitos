import numpy as np
from scipy.sparse.linalg import spsolve
from engine import assemble_system

from mesh import sqr_mesh2D

malha = sqr_mesh2D(lc=0.2,lim_inf=0.0,lim_sup=1.0)

f = lambda x, y: 2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)

K, F = assemble_system(malha.nodes,malha.elements,1,f)

u = spsolve(K, F)