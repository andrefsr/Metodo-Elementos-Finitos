import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from main.engine import assemble_system
from main.boundary_conditions import apply_dirichlet
from main.mesh import mesh_2D
from main.l2_error import l2_error

def fem_poisson(p_values,lc_values):
    for p in p_values:
        print('Ordem p =',p)
        erros = []
        for lc in lc_values:

            malha = mesh_2D(lc=lc,lim_inf=0.0,lim_sup=1.0,show_mesh=False,p=p)

            f = lambda x, y: 2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)

            K, F = assemble_system(malha.nodes,malha.triangles,p,f)

            dirichlet_dofs = np.unique(malha.boundary_faces)
            dirichlet_values = np.zeros(len(dirichlet_dofs))

            Kff, Ff, free_dofs = apply_dirichlet(K, F, dirichlet_dofs, dirichlet_values)

            u_free = spsolve(csr_matrix(Kff),Ff)

            u = np.zeros(len(F))

            u[dirichlet_dofs] = dirichlet_values
            u[free_dofs] = u_free

            x = malha.nodes[:,0]
            y = malha.nodes[:,1]
            u_exact = (np.sin(np.pi * x)* np.sin(np.pi * y))

            error_l2 = l2_error(malha,u,p)
            erros.append(error_l2)
                    
            if len(erros) >= 2:
                rate = np.log(erros[-2] / erros[-1]) / np.log(2)
                print(f"lc = {lc} ==> Norma L2 do erro = {error_l2:1.2e} ==> Taxa de convergência = {rate:.2f}")
            else: print(f"lc = {lc} ==> Norma L2 do erro = {error_l2:1.2e}")

        x_reta = np.linspace(np.min(np.log(lc_values)),np.max(np.log(lc_values)),100)
        b = np.log(erros[-1])
        a = [2,3,4]

        y_reta = a[p-1]*(x_reta - np.min(np.log(lc_values))) + b

        plt.figure()
        plt.title(f'Convergência Espacial - p = {p}')
        plt.grid(alpha=0.3)
        plt.plot(x_reta,y_reta,ls='--',color='black')
        plt.scatter(np.log(lc_values),np.log(erros),color='red')
        plt.ylabel(r'$||u - u_h||_{L^2}$ (log)')
        plt.xlabel('Tamanho característico do elemento (log)')
        plt.show()

    #--------------------------------------------------#

    triang = mtri.Triangulation(x, y, malha.triangles[:, :3])

    plt.figure()
    plt.tricontourf(triang,u,levels=20)
    plt.colorbar(label="u")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Solução numérica - MEF - lc = {lc} - p = {p}")
    plt.show()

    plt.figure()
    plt.tricontourf(triang,u_exact,levels=20)
    plt.colorbar(label="u")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solução analítica - Exata")
    plt.show()