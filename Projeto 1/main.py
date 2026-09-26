import numpy as np

from mesh import sqr_mesh2D

malha = sqr_mesh2D(lc=0.2,lim_inf=0.0,lim_sup=1.0)

f = lambda x, y: 2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)

#from scipy.sparse.linalg import spsolve
#u = spsolve(K, F)