import gmsh
import numpy as np
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

def remove_unused_nodes_L(nodes,triangles,boundary_faces):

    used_nodes = np.unique(triangles.flatten())
    new_nodes = nodes[used_nodes]
    old_to_new = {old: new for new, old in enumerate(used_nodes)}
    new_triangles = np.array([[old_to_new[i] for i in element] for element in triangles], dtype=int)
    new_boundary_faces = np.array([[old_to_new[i] for i in face] for face in boundary_faces], dtype=int)

    return (new_nodes,new_triangles,new_boundary_faces)

def L_mesh2D(lc=0.1,p=1,show_mesh=False):
    if p not in [1, 2, 3]:
        raise ValueError("p deve ser 1, 2 ou 3.")

    gmsh.initialize()
    gmsh.model.add("L_domain")

    p1 = gmsh.model.geo.addPoint(-1.0, -1.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(1.0, -1.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(1.0,  1.0, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(-1.0,  1.0, 0.0, lc)
    p5 = gmsh.model.geo.addPoint(-1.0,  0.0, 0.0, lc)
    p6 = gmsh.model.geo.addPoint(0.0,  0.0, 0.0, lc)
    p7 = gmsh.model.geo.addPoint(0.0, -1.0, 0.0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p1)

    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7])
    surface = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    domain = gmsh.model.addPhysicalGroup(2,[surface],1)

    gmsh.model.setPhysicalName(2,domain,"Domain")

    boundary = gmsh.model.addPhysicalGroup(1,[l1, l2, l3, l4, l5, l6, l7],2)

    gmsh.model.setPhysicalName(1,boundary,"Boundary")

    gmsh.model.mesh.generate(2)

    print("Entidades 2D:")
    print(gmsh.model.getEntities(dim=2))

    print("Superfície criada:", surface)

    types, tags, conn = gmsh.model.mesh.getElements(
        dim=2,
        tag=surface
    )

    print("Tipos na superfície:", types)
    print("Número de grupos:", len(types))

    for etype, etags, econn in zip(types, tags, conn):
        print(
            "tipo =", etype,
            "n_elementos =", len(etags),
            "n_nos_total =", len(econn)
        )

    gmsh.model.mesh.setOrder(p)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.asarray(node_coords,dtype=float).reshape(-1, 3)
    nodes = node_coords[:, :2]
    node_map = {int(tag): i for i, tag in enumerate(node_tags)}

    triangle_type = {1: 2,2: 9,3: 21}
    nodes_per_element = {1: 3,2: 6,3: 10}

    expected_type = triangle_type[p]
    npe = nodes_per_element[p]

    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(dim=2)

    triangles = None

    for etype, tags, conn in zip(element_types,element_tags,element_node_tags):
        if etype == expected_type:
            conn = np.asarray(conn,dtype=int).reshape(-1, npe)
            triangles = np.array([[node_map[int(tag)] for tag in element] for element in conn], dtype=int)
            break

    if triangles is None:
        gmsh.finalize()
        raise RuntimeError(f"Triângulos P{p} não encontrados.")

    line_type = {1: 1,2: 8,3: 26}

    expected_line_type = line_type[p]

    nface = p + 1

    boundary_faces = []

    boundary_element_types, _, boundary_node_tags = gmsh.model.mesh.getElements(dim=1)

    for etype, conn in zip(boundary_element_types,boundary_node_tags):
        if etype != expected_line_type:
            continue
        conn = np.asarray(conn,dtype=int).reshape(-1, nface)
        for face in conn:
            boundary_faces.append([node_map[int(tag)] for tag in face])

    boundary_faces = np.asarray(boundary_faces,dtype=int)

    (nodes,triangles,boundary_faces) = remove_unused_nodes_L(nodes,triangles,boundary_faces)

    dirichlet_dofs = np.unique(boundary_faces.flatten())

    if show_mesh:
        gmsh.fltk.run()

    gmsh.finalize()

    msh = SimpleNamespace(
        nodes=nodes,
        triangles=triangles,
        boundary_faces=boundary_faces,
        dirichlet_dofs=dirichlet_dofs,
        p=p)

    return msh
