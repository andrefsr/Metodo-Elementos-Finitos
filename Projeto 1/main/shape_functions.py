import numpy as np

def p1(xi,eta):
    return np.array([1-xi-eta,xi,eta]) ##3 nós

def grad_p1(xi,eta):
    '''dN/d(xi,eta)'''
    return np.array([[-1,-1],[1,0],[0,1]])

def p2(xi,eta):
    ##coordenadas baricêntricas
    l1 = 1-xi-eta
    l2 = xi
    l3 = eta
    ##6 nós
    N = np.array([
        l1*(2*l1-1), l2*(2*l2-1), l3*(2*l3-1),
        4*l1*l2, 4*l2*l3, 4*l3*l1 
        ])

    return N

def grad_p2(xi,eta):

    l1 = 1-xi-eta
    l2 = xi
    l3 = eta

    dN = np.array([
        [-(4*l1-1),-(4*l1-1)],
        [4*l2-1,0],
        [0,4*l3-1],
        [4*(l1-l2),-4*l2],
        [4*l3,4*l2],
        [-4*l3,4*(l1-l3)],
    ])

    return dN

## IMPLEMENTAR Pn GERAL

#def p3(xi,eta):


#def grad_p3(xi,eta):
    

    #return

def shape_functions(xi:float,eta:float,order:int):
    if order == 1:
        return p1(xi,eta), grad_p1(xi,eta)
    elif order == 2:
        return p2(xi,eta), grad_p2(xi,eta)
    #elif order == 3:
        #return p3(xi,eta), grad_p3(xi,eta)

    else: raise ValueError('Ordem não implementada')