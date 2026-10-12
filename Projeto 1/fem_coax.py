import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import gmsh
from types import SimpleNamespace

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from Matrices import element_stiffness, triangle_quadrature, jacobian
from shape_functions import shape_functions
from geometry import xi_eta_to_xy

def coax_mesh2D(lc=1e-3,a=2e-3,c=5e-3,b=8e-3,p=1,show_mesh=False):

    if p not in [1, 2, 3]:
        raise ValueError("p deve ser 1, 2 ou 3.")

    gmsh.initialize()
    gmsh.model.add("coaxial")

    center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
   
    def add_circle(r):
        p1 = gmsh.model.geo.addPoint(r, 0, 0, lc)
        p2 = gmsh.model.geo.addPoint(0, r, 0, lc)
        p3 = gmsh.model.geo.addPoint(-r, 0, 0, lc)
        p4 = gmsh.model.geo.addPoint(0, -r, 0, lc)
        c1 = gmsh.model.geo.addCircleArc(p1, center, p2)
        c2 = gmsh.model.geo.addCircleArc(p2, center, p3)
        c3 = gmsh.model.geo.addCircleArc(p3, center, p4)
        c4 = gmsh.model.geo.addCircleArc(p4, center, p1)
        loop_ccw = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
        loop_cw = gmsh.model.geo.addCurveLoop([-c1, -c2, -c3, -c4])
        return loop_ccw, loop_cw, [c1, c2, c3, c4]

    loop_a_ccw, loop_a_cw, circle_a = add_circle(a)
    loop_c_ccw, loop_c_cw, circle_c = add_circle(c)
    loop_b_ccw, loop_b_cw, circle_b = add_circle(b)

    # Região ε1: a < r < c
    region_1 = gmsh.model.geo.addPlaneSurface([loop_c_ccw,loop_a_cw])

    # Região ε2: c < r < b
    region_2 = gmsh.model.geo.addPlaneSurface([loop_b_ccw,loop_c_cw])

    gmsh.model.geo.synchronize()

    phys_region_1 = gmsh.model.addPhysicalGroup(2,[region_1],1)
    gmsh.model.setPhysicalName(2,phys_region_1,"Dielectric_1")

    phys_region_2 = gmsh.model.addPhysicalGroup(2,[region_2],2)
    gmsh.model.setPhysicalName(2,phys_region_2,"Dielectric_2")

    # Condutor interno
    inner_boundary = gmsh.model.addPhysicalGroup(1,circle_a,10)
    gmsh.model.setPhysicalName(1,inner_boundary,"InnerConductor")

    # Interface
    interface = gmsh.model.addPhysicalGroup(1,circle_c,11)
    gmsh.model.setPhysicalName(1,interface,"Interface")

    # Condutor externo
    outer_boundary = gmsh.model.addPhysicalGroup(1,circle_b,12)
    gmsh.model.setPhysicalName(1,outer_boundary,"OuterConductor")

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(p)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

    node_coords = np.asarray(node_coords,dtype=float).reshape(-1, 3)

    nodes = node_coords[:, :2]

    node_map = {int(tag): i for i, tag in enumerate(node_tags)}

    triangle_type = {1: 2,2: 9,3: 21}
    nodes_per_element = {1: 3,2: 6,3: 10}

    expected_type = triangle_type[p]
    npe = nodes_per_element[p]

    triangles = []
    element_material = []

    for surface, material_id in [(region_1, 1),(region_2, 2)]:
        element_types, _, element_node_tags = gmsh.model.mesh.getElements(dim=2,tag=surface)
        for etype, conn in zip(element_types,element_node_tags):
            if etype != expected_type:
                continue
            conn = np.asarray(conn,dtype=int).reshape(-1, npe)
            for element in conn:
                triangles.append([node_map[int(tag)] for tag in element])
                element_material.append(material_id)

    triangles = np.asarray(triangles,dtype=int)

    element_material = np.asarray(element_material,dtype=int)
    line_type = {1: 1,2: 8,3: 26}

    expected_line_type = line_type[p]
    nface = p + 1

    def get_boundary_faces(curves):

        faces = []
        for curve in curves:
            element_types, _, node_data = gmsh.model.mesh.getElements(dim=1,tag=curve)

            for etype, conn in zip(element_types,node_data):
                if etype != expected_line_type:
                    continue

                conn = np.asarray(conn,dtype=int).reshape(-1, nface)

                for face in conn:
                    faces.append([node_map[int(tag)] for tag in face])

        return np.asarray(faces,dtype=int)

    inner_faces = get_boundary_faces(circle_a)
    interface_faces = get_boundary_faces(circle_c)
    outer_faces = get_boundary_faces(circle_b)

    inner_dofs = np.unique(inner_faces.flatten())
    outer_dofs = np.unique(outer_faces.flatten())
    dirichlet_dofs = np.concatenate([inner_dofs,outer_dofs])
    dirichlet_values = np.concatenate([np.zeros(len(inner_dofs)),np.ones(len(outer_dofs))])

    if show_mesh: gmsh.fltk.run()

    gmsh.finalize()

    (nodes,triangles,inner_faces,interface_faces,outer_faces,dirichlet_dofs,dirichlet_values) = remove_unused_nodes(nodes,triangles,inner_faces,interface_faces,outer_faces,dirichlet_dofs,dirichlet_values)

    msh = SimpleNamespace(
        nodes=nodes,
        triangles=triangles,
        element_material=element_material,
        inner_faces=inner_faces,
        interface_faces=interface_faces,
        outer_faces=outer_faces,
        dirichlet_dofs=dirichlet_dofs,
        dirichlet_values=dirichlet_values,
        p=p )

    return msh

def remove_unused_nodes(nodes,triangles,inner_faces=None,interface_faces=None,outer_faces=None,dirichlet_dofs=None,dirichlet_values=None):
    used_nodes = np.unique(triangles.flatten())
    # Novas coordenadas
    new_nodes = nodes[used_nodes]
    old_to_new = {old: new for new, old in enumerate(used_nodes)}
    new_triangles = np.array([[old_to_new[node] for node in element] for element in triangles], dtype=int)

    def remap_faces(faces):
        if faces is None:
            return None
        return np.array([[old_to_new[node] for node in face]for face in faces], dtype=int)

    new_inner_faces = remap_faces(inner_faces)
    new_interface_faces = remap_faces(interface_faces)
    new_outer_faces = remap_faces(outer_faces)

    if dirichlet_dofs is not None:
        new_dirichlet_dofs = np.array([old_to_new[node] for node in dirichlet_dofs], dtype=int)
    else:
        new_dirichlet_dofs = None

    new_dirichlet_values = dirichlet_values

    return (new_nodes,new_triangles,new_inner_faces,new_interface_faces,new_outer_faces,new_dirichlet_dofs,new_dirichlet_values)

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

def apply_dirichlet_coax(K,F,dirichlet_dofs,dirichlet_values):

    dirichlet_dofs = np.asarray(dirichlet_dofs,dtype=int)
    dirichlet_values = np.asarray(dirichlet_values,dtype=float)
    n = len(F)
    all_dofs = np.arange(n)
    free_dofs = np.setdiff1d(all_dofs,dirichlet_dofs)
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfc = K[np.ix_(free_dofs, dirichlet_dofs)]
    Ff = (F[free_dofs]-Kfc @ dirichlet_values)
    
    return Kff, Ff, free_dofs

def exact_coaxial(x, y):

    a = 2e-3
    c = 5e-3
    b = 8e-3
    er1 = 2.0
    er2 = 4.0

    x = np.asarray(x)
    y = np.asarray(y)

    r = np.sqrt(x**2 + y**2)

    denominator = (
        np.log(c/a)/er1
        +
        np.log(b/c)/er2
    )

    u = np.zeros_like(r, dtype=float)

    mask1 = (r >= a) & (r <= c)
    mask2 = (r > c) & (r <= b)

    u[mask1] = (
        np.log(r[mask1]/a)/er1
    ) / denominator

    u[mask2] = (
        np.log(c/a)/er1
        +
        np.log(r[mask2]/c)/er2
    ) / denominator

    # Se a entrada for escalar, retorna escalar
    if u.ndim == 0:
        return float(u)

    return u

def l2_error_coaxi(mesh, u, p, exact_solution):

    error_squared = 0.0

    # Pontos e pesos da quadratura
    points, weights = triangle_quadrature(8)

    # Percorre os elementos
    for element in mesh.triangles:

        # Coordenadas dos nós do elemento
        element_nodes = mesh.nodes[element]

        # Valores de u nos nós do elemento
        element_u = u[element]

        # Integração
        for (xi, eta), w in zip(points, weights):

            # Funções de forma
            N, _ = shape_functions(xi, eta, p)

            # Coordenadas físicas
            xq, yq = xi_eta_to_xy(
                xi,
                eta,
                element_nodes,
                p
            )

            # Solução numérica
            uh = N @ element_u

            # Solução exata
            uex = exact_solution(xq, yq)

            # Jacobiano
            J = jacobian(
                xi,
                eta,
                element_nodes,
                p
            )

            detJ = abs(np.linalg.det(J))

            # Erro
            error_squared += (
                (uex - uh)**2
                * detJ
                * w
            )

    return np.sqrt(error_squared)

def fem_coax(p_values,lc_values):
    for p in p_values:
        print('Ordem p =',p)
        erros = []
        for lc in lc_values:

            malha = coax_mesh2D(lc=lc,a=0.002,c=0.005,b=0.008,p=p,show_mesh=False)

            K, F = assemble_coax_system(malha.nodes,malha.triangles,malha.element_material,p)

            Kff, Ff, free_dofs = apply_dirichlet_coax(K,F,malha.dirichlet_dofs,malha.dirichlet_values)

            u_free = spsolve(csr_matrix(Kff),Ff)
            
            u = np.zeros(len(F))
            u[malha.dirichlet_dofs] = malha.dirichlet_values
            u[free_dofs] = u_free

            x = malha.nodes[:, 0]
            y = malha.nodes[:, 1]

            u_exact = exact_coaxial(x, y)

            error_l2 = l2_error_coaxi(malha,u,p,exact_coaxial)
            erros.append(error_l2)
                    
            if len(erros) >= 2:
                rate = np.log(erros[-2] / erros[-1]) / np.log(2)
                print(f"lc = {lc} ==> Norma L2 do erro = {error_l2:1.2e} ==> Taxa de convergência = {rate:.2f}")
            else: print(f"lc = {lc} ==> Norma L2 do erro = {error_l2:1.2e}")

            eps0 = 8.8541878128e-12

            C_numeric = (eps0 * (u @ (K @ u)))
            print(f"C' = {(C_numeric * 1e12):.2f} pF/m ===> Valor esperado : 96.64 pF/m")

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

    triang = mtri.Triangulation(x,y,malha.triangles[:, :3])
    plt.figure()
    plt.tricontourf(triang,u,levels=30)
    plt.colorbar(label="u")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Potencial eletrostático - MEF - lc = {lc} - p = {p}")
    plt.show()

    plt.figure()
    plt.tricontourf(triang,u_exact,levels=30)
    plt.colorbar(label="u")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potencial eletrostático - Analítico")
    plt.show()