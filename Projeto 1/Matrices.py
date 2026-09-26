import numpy as np

def triangle_quadrature():

    points = np.array([[1/3,1/3]])
    weights = np.array([1/2])

    return points, weights

def element_stiffness(nodes,p,k=1.0):
    n = len(nodes)
    Ke = np.zeros((n,n))

    points, weights = triangle_quadrature()