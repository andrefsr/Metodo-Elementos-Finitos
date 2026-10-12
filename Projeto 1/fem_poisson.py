import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import gmsh
from types import SimpleNamespace

from Matrices import triangle_quadrature
from shape_functions import shape_functions
from Matrices import jacobian
from geometry import xi_eta_to_xy
from Matrices import element_stiffness, element_load

def mesh_2D(lc=0.1,lim_inf=0.0,lim_sup=1.0,p=1,show_mesh=False):

    if p not in [1, 2, 3]:
        raise ValueError("p deve ser 1, 2 ou 3.")

    gmsh.initialize()
    gmsh.model.add("mesh_2D")

    p1 = gmsh.model.geo.addPoint(lim_inf, lim_inf, 0, lc)
    p2 = gmsh.model.geo.addPoint(lim_sup, lim_inf, 0, lc)
    p3 = gmsh.model.geo.addPoint(lim_sup, lim_sup, 0, lc)
    p4 = gmsh.model.geo.addPoint(lim_inf, lim_sup, 0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    domain = gmsh.model.addPhysicalGroup(2,[surface],1)

    gmsh.model.setPhysicalName(2,domain,"Domain")

    boundary = gmsh.model.addPhysicalGroup(1,[l1, l2, l3, l4],2)

    gmsh.model.setPhysicalName(1,boundary,"Boundary")

    gmsh.model.mesh.generate(2)

    # transforma os elementos em P1, P2 ou P3
    gmsh.model.mesh.setOrder(p)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

    node_coords = np.asarray(node_coords,dtype=float).reshape(-1, 3)

    # somente x,y
    nodes = node_coords[:, :2]

    # Mapeamento: tag do Gmsh -> índice do numpy
    node_map = {int(tag): i for i, tag in enumerate(node_tags)}

    # Tipos de elementos triangulares do Gmsh
    triangle_type = {1: 2,2: 9,3: 21}
    nodes_per_element = {1: 3,2: 6,3: 10}

    expected_type = triangle_type[p]
    npe = nodes_per_element[p]

    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(dim=2)

    triangles = None

    for etype, tags, conn in zip(element_types,element_tags,element_node_tags):
        if etype == expected_type:
            conn = np.asarray(conn,dtype=int).reshape(-1, npe)
            # converter tags do Gmsh para índices Python
            triangles = np.array([[node_map[int(tag)] for tag in element] for element in conn])
            break

    if triangles is None:
        gmsh.finalize()
        raise RuntimeError(f"Não foi encontrado elemento triangular P{p}.")

    line_type = {1: 1,2: 8,3: 26}

    expected_line_type = line_type[p]

    boundary_element_types, boundary_tags, boundary_node_tags = gmsh.model.mesh.getElements(dim=1)

    boundary_faces = []

    for etype, tags, conn in zip(boundary_element_types,boundary_tags,boundary_node_tags):
        if etype == expected_line_type:
            nface = p + 1
            conn = np.asarray(conn,dtype=int).reshape(-1, nface)
            for face in conn: boundary_faces.append([node_map[int(tag)] for tag in face])

    boundary_faces = np.asarray(boundary_faces,dtype=int)

    if len(boundary_faces) > 0:dirichlet_dofs = np.unique(boundary_faces.flatten())
    else: dirichlet_dofs = np.array([],dtype=int)

    if show_mesh: gmsh.fltk.run()

    gmsh.finalize()

    msh = SimpleNamespace(
        nodes=nodes,
        triangles=triangles,
        boundary_faces=boundary_faces,
        dirichlet_dofs=dirichlet_dofs,
        p=p)

    return msh

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

def apply_dirichlet(K, F, dirichlet_dofs, dirichlet_values):

    all_dofs = np.arange(len(F))
    free_dofs = np.setdiff1d(all_dofs, dirichlet_dofs)
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfc = K[np.ix_(free_dofs, dirichlet_dofs)]
    Ff = (F[free_dofs] - Kfc @ dirichlet_values)

    return Kff, Ff, free_dofs

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