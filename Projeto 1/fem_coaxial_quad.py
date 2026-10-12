import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import gmsh

A, B, C = 2.0, 8.0, 5.0            # mm
EPS_R = np.array([2.0, 4.0])       # eps_r1 (a<r<c), eps_r2 (c<r<b)
V0 = 1.0                           # ddp aplicada
EPS0 = 8.8541878128e-12            # F/m

def g_dirichlet(x, y):
    """Potencial prescrito: 0 em r = a, V0 em r = b."""
    return np.where(np.hypot(x, y) < 0.5 * (A + B), 0.0, V0)

# Solução analítica (campo radial com fluxo ε r dV/dr = k constante)
_K = V0 / (np.log(C / A) / EPS_R[0] + np.log(B / C) / EPS_R[1])

def u_exact(x, y):
    r = np.hypot(x, y)
    return np.where(r <= C,
                    _K / EPS_R[0] * np.log(r / A),
                    _K / EPS_R[0] * np.log(C / A) + _K / EPS_R[1] * np.log(r / C))

def grad_u_exact(x, y):
    r = np.hypot(x, y)
    eps = np.where(r <= C, EPS_R[0], EPS_R[1])
    dr = _K / (eps * r)
    return dr * x / r, dr * y / r

def capacitance_exact():
    """Capacitância por unidade de comprimento (F/m)."""
    return 2.0 * np.pi * EPS0 / (np.log(C / A) / EPS_R[0] + np.log(B / C) / EPS_R[1])

def generate_mesh(lc):
    """
    Retorna:
      coords (nn,2) : coordenadas de todos os nós (inclui nós curvos de aresta e centro)
      conn   (nel,9): conectividade Q2 (4 vértices, 4 pontos médios de aresta, centro),
                      anti-horária; os 4 primeiros nós são os vértices (malha Q1)
      region (nel,) : 0 = dielétrico 1 (a<r<c), 1 = dielétrico 2 (c<r<b)
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 8)               # frontal-Delaunay p/ quads
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # blossom full-quad
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)    # garante só quads
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)   # quad de 9 nós
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)       # nós curvos sobre a geometria
    gmsh.model.add("coax")
    geo = gmsh.model.geo

    O = geo.addPoint(0, 0, 0, lc)

    def circle(r):                                           # 4 arcos de 90 graus
        p = [geo.addPoint(r, 0, 0, lc), geo.addPoint(0, r, 0, lc),
             geo.addPoint(-r, 0, 0, lc), geo.addPoint(0, -r, 0, lc)]
        return [geo.addCircleArc(p[i], O, p[(i + 1) % 4]) for i in range(4)]

    def loop(arcs, reverse=False):
        return geo.addCurveLoop([-t for t in reversed(arcs)] if reverse else arcs)

    arcs_a, arcs_c, arcs_b = circle(A), circle(C), circle(B)
    s1 = geo.addPlaneSurface([loop(arcs_c), loop(arcs_a, True)])   # dielétrico 1
    s2 = geo.addPlaneSurface([loop(arcs_b), loop(arcs_c, True)])   # dielétrico 2
    geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)

    tags, xyz, _ = gmsh.model.mesh.getNodes()
    table = np.zeros((int(np.max(tags)) + 1, 3))
    table[np.asarray(tags, dtype=np.int64)] = np.asarray(xyz).reshape(-1, 3)

    conn_tags, region = [], []
    for k, s in enumerate((s1, s2)):
        etypes, _, enodes = gmsh.model.mesh.getElements(2, s)
        if list(etypes) != [10]:
            raise RuntimeError("A malha contém elementos que não são quadriláteros de 9 nós.")
        c9 = np.asarray(enodes[0], dtype=np.int64).reshape(-1, 9)
        conn_tags.append(c9)
        region.append(np.full(len(c9), k))
    gmsh.finalize()

    conn_tags = np.vstack(conn_tags)
    region = np.concatenate(region)
    used = np.unique(conn_tags)                              # descarta nós sem elemento (ex.: centro O)
    conn = np.searchsorted(used, conn_tags)
    coords = table[used][:, :2]
    return coords, orient_ccw(coords, conn), region

def orient_ccw(coords, conn):
    """Garante vértices em ordem anti-horária (permutando também os nós de aresta)."""
    conn = conn.copy()
    v = coords[conn[:, :4]]
    area = 0.5 * np.sum(v[:, :, 0] * np.roll(v[:, :, 1], -1, axis=1)
                        - np.roll(v[:, :, 0], -1, axis=1) * v[:, :, 1], axis=1)
    neg = area < 0
    conn[neg] = conn[neg][:, [0, 3, 2, 1, 7, 6, 5, 4, 8]]
    return conn

def build_spaces(coords, conn9):
    """Espaços Q1 (só vértices, arestas retas) e Q2 (isoparamétrico, nós curvos do Gmsh)."""
    verts = np.unique(conn9[:, :4])
    remap = -np.ones(len(coords), dtype=np.int64)
    remap[verts] = np.arange(len(verts))
    return {1: (coords[verts], remap[conn9[:, :4]]), 2: (coords, conn9)}

def dirichlet_mask(coords, tol=1e-7):
    r = np.hypot(coords[:, 0], coords[:, 1])
    return (np.abs(r - A) < tol) | (np.abs(r - B) < tol)

LOCAL_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
REF_VERT = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

def ref_nodes(p):
    nodes = [*REF_VERT]
    if p == 2:
        for (i, j) in LOCAL_EDGES:
            nodes.append(0.5 * (REF_VERT[i] + REF_VERT[j]))
        nodes.append(np.array([0.0, 0.0]))
    return np.array(nodes)

def lagrange_1d(t_nodes, t):
    n = len(t_nodes)
    L, dL = np.ones((len(t), n)), np.zeros((len(t), n))
    for k in range(n):
        for m in range(n):
            if m != k:
                L[:, k] *= (t - t_nodes[m]) / (t_nodes[k] - t_nodes[m])
        for j in range(n):
            if j == k:
                continue
            term = np.ones(len(t)) / (t_nodes[k] - t_nodes[j])
            for m in range(n):
                if m != k and m != j:
                    term *= (t - t_nodes[m]) / (t_nodes[k] - t_nodes[m])
            dL[:, k] += term
    return L, dL

def basis(p, pts):
    """phi (nq,nb) e dphi (nq,nb,2) em relação a (xi, eta)."""
    t1 = np.linspace(-1.0, 1.0, p + 1)
    Lx, dLx = lagrange_1d(t1, pts[:, 0])
    Ly, dLy = lagrange_1d(t1, pts[:, 1])
    nodes = ref_nodes(p)
    ia = [int(np.argmin(abs(t1 - x))) for x in nodes[:, 0]]
    ja = [int(np.argmin(abs(t1 - y))) for y in nodes[:, 1]]
    phi = np.stack([Lx[:, i] * Ly[:, j] for i, j in zip(ia, ja)], axis=1)
    dphi = np.stack([np.stack([dLx[:, i] * Ly[:, j], Lx[:, i] * dLy[:, j]], axis=1)
                     for i, j in zip(ia, ja)], axis=1)
    return phi, dphi

def quad_rule(n):
    xg, wg = np.polynomial.legendre.leggauss(n)
    U, V = np.meshgrid(xg, xg, indexing="ij")
    return np.stack([U.ravel(), V.ravel()], axis=1), np.outer(wg, wg).ravel()

def geometry(coords, conn, xq, p):
    """Mapeamento isoparamétrico de ordem p (bilinear p=1, biquadrático p=2)."""
    v = coords[conn]                                         # (nel, nb, 2)
    N, dN = basis(p, xq)
    xp = np.einsum("qk,ekj->eqj", N, v)
    J = np.einsum("ekj,qkm->eqjm", v, dN)                    # J[i,m] = dx_i/dxi_m
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    invJ = np.empty_like(J)
    invJ[..., 0, 0], invJ[..., 0, 1] = J[..., 1, 1] / det, -J[..., 0, 1] / det
    invJ[..., 1, 0], invJ[..., 1, 1] = -J[..., 1, 0] / det, J[..., 0, 0] / det
    return xp, det, invJ

def assemble_stiffness(coords, conn, p, epsr_el):
    xq, wq = quad_rule(p + 3)
    _, dphi = basis(p, xq)
    _, det, invJ = geometry(coords, conn, xq, p)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
    w = wq[None, :] * np.abs(det) * epsr_el[:, None]         # peso * eps_r do elemento
    Ke = np.einsum("eq,eqbi,eqci->ebc", w, G, G)
    nb = conn.shape[1]
    rows = np.repeat(conn, nb, axis=1).ravel()
    cols = np.tile(conn, (1, nb)).ravel()
    n = len(coords)
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()

def solve(coords, conn, p, region):
    epsr_el = EPS_R[region]
    K = assemble_stiffness(coords, conn, p, epsr_el)
    dmask = dirichlet_mask(coords)
    free, dirn = np.where(~dmask)[0], np.where(dmask)[0]
    u = np.zeros(len(coords))
    u[dirn] = g_dirichlet(coords[dirn, 0], coords[dirn, 1])
    # Sistema reduzido (equações dos nós de Dirichlet eliminadas): Kff uf = - Kfd ud
    rhs = -K[free][:, dirn] @ u[dirn]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return u

def compute_errors(coords, conn, u, p, region, nq=10, chunk=2000):

    xq, wq = quad_rule(nq)
    phi, dphi = basis(p, xq)
    e2 = en = W2 = 0.0
    for s in range(0, len(conn), chunk):
        c = conn[s:s + chunk]
        epsr = EPS_R[region[s:s + chunk]][:, None]
        xp, det, invJ = geometry(coords, c, xq, p)
        G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
        ue = u[c]
        uh = np.einsum("qb,eb->eq", phi, ue)
        gh = np.einsum("eqbi,eb->eqi", G, ue)
        ex = u_exact(xp[..., 0], xp[..., 1])
        gx, gy = grad_u_exact(xp[..., 0], xp[..., 1])
        w = wq[None, :] * np.abs(det)
        e2 += np.sum(w * (ex - uh) ** 2)
        en += np.sum(w * epsr * ((gx - gh[..., 0]) ** 2 + (gy - gh[..., 1]) ** 2))
        W2 += np.sum(w * epsr * (gh[..., 0] ** 2 + gh[..., 1] ** 2))
    return np.sqrt(e2), np.sqrt(en), W2

def capacitance_fem(W2):
    """W = (1/2) C V^2 = (1/2) eps0 int eps_r |grad V|^2   =>   C = eps0 * W2 / V^2  (F/m)."""
    return EPS0 * W2 / V0 ** 2

def mesh_h(coords, conn):
    v = conn[:, :4]
    return max(np.max(np.linalg.norm(coords[v[:, i]] - coords[v[:, j]], axis=1))
               for i, j in LOCAL_EDGES)

def convergence_study_b(lcs, orders=(1, 2), mesh_fn=generate_mesh):
    Cex = capacitance_exact()
    res = {p: {"h": [], "L2": [], "E": [], "C": [], "ndof": []} for p in orders}
    for lc in lcs:
        coords9, conn9, region = mesh_fn(lc)
        spaces = build_spaces(coords9, conn9)
        h = mesh_h(coords9, conn9)
        for p in orders:
            coords, conn = spaces[p]
            u = solve(coords, conn, p, region)
            eL2, eE, W2 = compute_errors(coords, conn, u, p, region)
            Ch = capacitance_fem(W2)
            eC = abs(Ch - Cex) / Cex
            for k, v in zip(("h", "L2", "E", "C", "ndof"), (h, eL2, eE, eC, len(coords))):
                res[p][k].append(v)
            print(f"lc={lc:7.4f} h={h:.4f} Q{p} nel={len(conn):6d} ndof={len(coords):7d} "
                  f"errL2={eL2:.3e} errE={eE:.3e}  C={Ch*1e12:.6f} pF/m  errC(rel)={eC:.3e}")
    print(f"C exata = {Cex*1e12:.6f} pF/m")
    return res

def plot_convergence_b(res, fname="convergencia_coaxial.png"):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    titles = ("Erro $L_2$", "Erro na norma de energia", "Erro relativo da capacitância")
    for p, d in res.items():
        h = np.array(d["h"])
        for a, key, nome in zip(ax, ("L2", "E", "C"), titles):
            err = np.array(d[key])
            m = np.polyfit(np.log(h), np.log(err), 1)[0]
            a.loglog(h, err, "o-", label=f"$Q_{p}$  (inclinação ≈ {m:.2f})")
            a.set_xlabel("h (mm)"); a.set_ylabel(nome); a.set_title(nome)
            a.grid(True, which="both", alpha=.3)
    for a in ax:
        a.legend()
    plt.tight_layout()
    #plt.savefig(fname, dpi=150)
    #print("Figura salva em", fname)

def plot_solution_b(lc=0.5, p=2, fname="solucao_coaxial.png"):
    coords9, conn9, region = generate_mesh(lc)
    coords, conn = build_spaces(coords9, conn9)[p]
    u = solve(coords, conn, p, region)
    v = conn[:, :4]
    tri = np.vstack([v[:, [0, 1, 2]], v[:, [0, 2, 3]]])      # só para visualização
    x, y = coords[:, 0], coords[:, 1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for a, z, t in zip(ax, (u, u - u_exact(x, y)), ("Potencial V", "Erro nodal")):
        c = a.tricontourf(x, y, tri, z, 30)
        plt.colorbar(c, ax=a); a.set_aspect("equal"); a.set_title(t)
    plt.tight_layout()#; plt.savefig(fname, dpi=150)


