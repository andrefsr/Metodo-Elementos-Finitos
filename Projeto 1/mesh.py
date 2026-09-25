import gmsh
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

def sqr_mesh2D(lc:float, lim_inf:float, lim_sup:float,show_mesh:bool = False):
    ''' Malha quadrada - lc = 0.2 ## tamanho característico dos elementos (tamanho alvo) '''

    gmsh.initialize()
    gmsh.model.add('dominio')
    
    #pontos ## (x,y,z, tamanho do elemento)
    p1 = gmsh.model.geo.addPoint(lim_inf,lim_inf,0,lc) 
    p2 = gmsh.model.geo.addPoint(lim_sup,lim_inf,0,lc)
    p3 = gmsh.model.geo.addPoint(lim_sup,lim_sup,0,lc)
    p4 = gmsh.model.geo.addPoint(lim_inf,lim_sup,0,lc)

    #linhas que ligam os pontos
    l1 = gmsh.model.geo.addLine(p1,p2)
    l2 = gmsh.model.geo.addLine(p2,p3)
    l3 = gmsh.model.geo.addLine(p3,p4)
    l4 = gmsh.model.geo.addLine(p4,p1)

    #gerando a superficie
    loop = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4]) ##conectando todas as linhas
    surface = gmsh.model.geo.addPlaneSurface([loop]) ##definindo a superfície interna definida pelo loop

    gmsh.model.geo.synchronize() ##envia a geometria para o gmsh

    ##grupos físicos (dominio e borda) --- atribui tags diferentes para cada grupo
    domain = gmsh.model.add_physical_group(2,[surface]) ##dominio interno
    gmsh.model.setPhysicalName(2,domain,'Domain')

    boundary = gmsh.model.addPhysicalGroup(1,[l1,l2,l3,l4])
    gmsh.model.setPhysicalName(1,boundary,'Boundary')

    gmsh.model.mesh.generate(dim=2) #o argumento é a dimensão do espaço a ser gerado

    ##extraindo nós
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    print('Nós antes do reshape:') ## REMOVER 
    print(node_coords) ## REMOVER 
    nodes_coords = np.array(node_coords).reshape(-1,3)

    nodes = nodes_coords[:, :2] ##pegando somente a coordenada x e y

    print('Nós:') ## REMOVER 
    print(node_coords) ## REMOVER 
    print('Número de Nós:',len(nodes)) ## REMOVER 

    ##mapa de tags do gmsh para indices do numpy
    #traduzindo as tags para indices python (usando dicionário)
    node_map = {tag: i for i,tag in enumerate(node_tags)}

    #extraindo elementos (triangulos) 
    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(dim=2)
    
    triangles = None
    for etype, tags, node_tags_element in zip(element_types,element_tags,element_node_tags):
        if etype == 2: ##3 é o triângulo linear
            triangles = np.arrray([node_map[tag] for tag in node_tags_element]).reshape(-1,3)
            break

    ##construindo as faces
    local_faces = [(0,1),(1,2),(2,0)]

    ##dicionario: chave = face : valor = [elemento, face_local]
    face_dict = {}

    for elem_id, elem in enumerate(triangles):
        for local_face_id, (i,j) in enumerate(local_faces):
            n1 = elem[i]
            n2 = elem[j]

            face = tuple(sorted((n1,n2))) ##ordenar para identificar a mesma face independente da orientação

            if face not in face_dict:
                face_dict[face] = []

            face_dict[face].append((elem_id,local_face_id))

    ##separando faces internas e de fronteira
    faces = []
    boundary_faces = []
    interior_faces = []

    for face, connected_elements in face_dict.items():
        faces.append(face)

        if len(connected_elements) == 1:
            boundary_faces.append(face)

        elif len(connected_elements) == 2:
            interior_faces.append(face)

    faces = np.array(faces)
    boundary_faces = np.array(boundary_faces)
    interior_faces = np.array(interior_faces)

    ##identificando elementos vizinhos

    #neighbors[e,f] onde e = elemento e f = face local
    #se for -1: face de fronteira, do contrário é o índice do elemento vizinho

    neighbors = -np.ones((len(triangles),3),dtype=int)

    for face, connected_elements in face_dict.items():
        if len(connected_elements) == 2:

            (e1,f1), (e2,f2) = connected_elements

            neighbors[e1,f1] = e2
            neighbors[e2,f2] = e1

    ##nós da fronteira
    boundary_nodes = np.unique(boundary_faces.flatten())

    if show_mesh == True:

        print('\nNós:')
        print(node_coords)

        print('\nNúmero de Nós:',len(nodes))

        print('\nNúmero de elementos:',len(triangles))

        print('\nConectividade:')
        print(triangles)

        print("\nNúmero total de faces",len(faces))
        print("Faces de fronteira",len(boundary_faces))
        print('Faces internas',len(interior_faces))

        print('\nVizinhança:')
        print(neighbors)

        print('\nNós da fronteira:')
        print(boundary_faces)

        plt.figure()
        plt.triplot(nodes[:,0],nodes[:1],triangles)
        plt.scatter(nodes[:,0],nodes[:1],s=10)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid()
        plt.show

    gmsh.finalize
    
    msh = SimpleNamespace(
        node_coords=node_coords, nodes=nodes, triangles=triangles, neighbors=neighbors,
        faces=faces, boundary_faces=boundary_faces, interior_faces=interior_faces)

    return msh 