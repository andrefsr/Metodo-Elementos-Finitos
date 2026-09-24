import gmsh
import numpy as np

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

            


    if show_mesh == True:
        ### printar malhamatplotlib

    return 