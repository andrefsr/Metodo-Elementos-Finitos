import gmsh
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace


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

    # Mapeamento:
    # tag do Gmsh -> índice do numpy
    node_map = {int(tag): i for i, tag in enumerate(node_tags)}

    # Tipos de elementos triangulares do Gmsh
    triangle_type = {
        1: 2,    # P1 -> 3 nós
        2: 9,    # P2 -> 6 nós
        3: 21    # P3 -> 10 nós
    }

    nodes_per_element = {
        1: 3,
        2: 6,
        3: 10
    }

    expected_type = triangle_type[p]
    npe = nodes_per_element[p]

    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(dim=2)

    triangles = None

    for etype, tags, conn in zip(
        element_types,
        element_tags,
        element_node_tags
    ):

        if etype == expected_type:

            conn = np.asarray(
                conn,
                dtype=int
            ).reshape(-1, npe)

            # converter tags do Gmsh para índices Python
            triangles = np.array([
                [
                    node_map[int(tag)]
                    for tag in element
                ]
                for element in conn
            ])

            break

    if triangles is None:
        gmsh.finalize()

        raise RuntimeError(
            f"Não foi encontrado elemento triangular P{p}."
        )

    # Tipos das arestas:
    #
    # P1 -> 2 nós
    # P2 -> 3 nós
    # P3 -> 4 nós

    line_type = {
        1: 1,    # linha P1
        2: 8,    # linha P2
        3: 26    # linha P3
    }

    expected_line_type = line_type[p]

    boundary_element_types, boundary_tags, boundary_node_tags = gmsh.model.mesh.getElements(dim=1)

    boundary_faces = []

    for etype, tags, conn in zip(
        boundary_element_types,
        boundary_tags,
        boundary_node_tags
    ):

        if etype == expected_line_type:

            nface = p + 1

            conn = np.asarray(
                conn,
                dtype=int
            ).reshape(-1, nface)

            for face in conn:

                boundary_faces.append([
                    node_map[int(tag)]
                    for tag in face
                ])

    boundary_faces = np.asarray(boundary_faces,dtype=int)

    if len(boundary_faces) > 0:
        dirichlet_dofs = np.unique(boundary_faces.flatten())
    else:
        dirichlet_dofs = np.array([],dtype=int)

    if show_mesh:
        gmsh.fltk.run()

    gmsh.finalize()

    msh = SimpleNamespace(
        nodes=nodes,
        triangles=triangles,
        boundary_faces=boundary_faces,
        dirichlet_dofs=dirichlet_dofs,
        p=p
    )

    return msh

