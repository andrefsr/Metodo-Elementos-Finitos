import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import gmsh

PI = np.pi

def kappa(x, y):                       # coeficiente da equação (aqui: 1)
    return np.ones_like(x)

def f_source(x, y):
    return 2.0 * PI ** 2 * np.sin(PI * x) * np.sin(PI * y)

def g_dirichlet(x, y):                 # potencial prescrito em dOmega
    return np.zeros_like(x)

def u_exact(x, y):
    return np.sin(PI * x) * np.sin(PI * y)

def grad_u_exact(x, y):
    return PI * np.cos(PI * x) * np.sin(PI * y), PI * np.sin(PI * x) * np.cos(PI * y)

def generate_mesh(lc, structured=False):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("square")
    p = [gmsh.model.geo.addPoint(x, y, 0.0, lc) for (x, y) in [(0, 0), (1, 0), (1, 1), (0, 1)]]
    ln = [gmsh.model.geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    loop = gmsh.model.geo.addCurveLoop(ln)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    if structured:
        n = max(1, int(round(1.0 / lc)))
        for l in ln:
            gmsh.model.geo.mesh.setTransfiniteCurve(l, n + 1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surf)
        gmsh.model.geo.mesh.setRecombine(2, surf)
    else:
        gmsh.option.setNumber("Mesh.Algorithm", 8)               # frontal-Delaunay p/ quads
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # blossom full-quad
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)    # garante só quads
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    tags, coords, _ = gmsh.model.mesh.getNodes()
    etypes, _, enodes = gmsh.model.mesh.getElements(2)
    gmsh.finalize()

    if list(etypes) != [3]:
        raise RuntimeError("A malha contém elementos que não são quadriláteros de 4 nós.")
    q_tags = np.asarray(enodes[0], dtype=np.int64).reshape(-1, 4)
    used = np.unique(q_tags)
    tag2xyz = {int(t): c for t, c in zip(tags, np.asarray(coords).reshape(-1, 3))}
    remap = {int(t): i for i, t in enumerate(used)}
    X = np.array([tag2xyz[int(t)][:2] for t in used])
    quad = np.vectorize(remap.get)(q_tags)
    return X, orient_ccw(X, quad)

def orient_ccw(X, quad):
    quad = quad.copy()
    v = X[quad]
    area = 0.5 * np.sum(v[:, :, 0] * np.roll(v[:, :, 1], -1, axis=1)
                        - np.roll(v[:, :, 0], -1, axis=1) * v[:, :, 1], axis=1)
    neg = area < 0
    quad[neg] = quad[neg][:, [0, 3, 2, 1]]
    return quad

LOCAL_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
REF_VERT = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def ref_nodes(p):
    """Nós de referência: 4 vértices, [4 pontos médios de aresta, centro (Q2)]."""
    nodes = [*REF_VERT]
    if p == 2:
        for (i, j) in LOCAL_EDGES:
            nodes.append(0.5 * (REF_VERT[i] + REF_VERT[j]))
        nodes.append(np.array([0.0, 0.0]))
    return np.array(nodes)

def lagrange_1d(t_nodes, t):
    """Polinômios de Lagrange 1D e derivadas em t: arrays (nt, n)."""
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
    """phi (nq,nb) e dphi (nq,nb,2) (derivadas em relação a xi, eta)."""
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
    """Gauss-Legendre tensorial n x n em [-1,1]^2."""
    xg, wg = np.polynomial.legendre.leggauss(n)
    U, V = np.meshgrid(xg, xg, indexing="ij")
    return np.stack([U.ravel(), V.ravel()], axis=1), np.outer(wg, wg).ravel()

def build_space(X, quad, p):
    nel, nv = len(quad), len(X)
    e = np.stack([quad[:, [i, j]] for (i, j) in LOCAL_EDGES], axis=1)      # (nel,4,2)
    uniq, inv = np.unique(np.sort(e, axis=2).reshape(-1, 2), axis=0, return_inverse=True)
    inv = inv.reshape(nel, 4)
    ned = len(uniq)
    bnd_edge = np.bincount(inv.ravel(), minlength=ned) == 1                # aresta em 1 só elemento

    nedge = 1 if p == 2 else 0
    ncen = 1 if p == 2 else 0
    ndof = nv + ned * nedge + nel * ncen
    coords = np.zeros((ndof, 2))
    coords[:nv] = X
    conn = np.zeros((nel, 4 + 4 * nedge + ncen), dtype=np.int64)
    conn[:, :4] = quad
    if p == 2:
        coords[nv:nv + ned] = 0.5 * (X[uniq[:, 0]] + X[uniq[:, 1]])
        for le in range(4):
            conn[:, 4 + le] = nv + inv[:, le]
        idx = nv + ned + np.arange(nel)
        conn[:, -1] = idx
        coords[idx] = X[quad].mean(axis=1)

    dir_mask = np.zeros(ndof, dtype=bool)                                  # nós de Dirichlet
    dir_mask[uniq[bnd_edge].ravel()] = True
    if p == 2:
        dir_mask[nv + np.where(bnd_edge)[0]] = True
    return coords, conn, dir_mask


def geometry(coords, conn, xq):
    """Mapeamento bilinear: pontos físicos, det J e J^{-1} em cada ponto de quadratura."""
    v = coords[conn[:, :4]]
    N, dN = basis(1, xq)
    xp = np.einsum("qk,ekj->eqj", N, v)
    J = np.einsum("ekj,qkm->eqjm", v, dN)                                  # J[i,m] = dx_i/dxi_m
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    invJ = np.empty_like(J)
    invJ[..., 0, 0], invJ[..., 0, 1] = J[..., 1, 1] / det, -J[..., 0, 1] / det
    invJ[..., 1, 0], invJ[..., 1, 1] = -J[..., 1, 0] / det, J[..., 0, 0] / det
    return xp, det, invJ

def assemble(coords, conn, p):
    xq, wq = quad_rule(p + 2)
    phi, dphi = basis(p, xq)
    xp, det, invJ = geometry(coords, conn, xq)
    w = wq[None, :] * np.abs(det)                                          # (nel,nq)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)                            # grad físico
    kap = kappa(xp[..., 0], xp[..., 1])
    Ke = np.einsum("eq,eq,eqbi,eqci->ebc", w, kap, G, G)
    fq = f_source(xp[..., 0], xp[..., 1])
    fe = np.einsum("eq,eq,qb->eb", w, fq, phi)

    nb = conn.shape[1]
    n = len(coords)
    rows = np.repeat(conn, nb, axis=1).ravel()
    cols = np.tile(conn, (1, nb)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    F = np.bincount(conn.ravel(), weights=fe.ravel(), minlength=n)
    return K, F

def solve(X, quad, p):
    coords, conn, dmask = build_space(X, quad, p)
    K, F = assemble(coords, conn, p)
    free, dirn = np.where(~dmask)[0], np.where(dmask)[0]
    u = np.zeros(len(coords))
    u[dirn] = g_dirichlet(coords[dirn, 0], coords[dirn, 1])
    # Sistema reduzido (equações dos nós de Dirichlet eliminadas): Kff uf = Ff - Kfd ud
    rhs = F[free] - K[free][:, dirn] @ u[dirn]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return coords, conn, u

def compute_errors(coords, conn, u, p, nq=8, chunk=2000):
    xq, wq = quad_rule(nq)
    phi, dphi = basis(p, xq)
    e2 = en = 0.0
    for s in range(0, len(conn), chunk):
        c = conn[s:s + chunk]
        xp, det, invJ = geometry(coords, c, xq)
        G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
        ue = u[c]
        uh = np.einsum("qb,eb->eq", phi, ue)
        gh = np.einsum("eqbi,eb->eqi", G, ue)
        ex = u_exact(xp[..., 0], xp[..., 1])
        gx, gy = grad_u_exact(xp[..., 0], xp[..., 1])
        w = wq[None, :] * np.abs(det)
        kap = kappa(xp[..., 0], xp[..., 1])
        e2 += np.sum(w * (ex - uh) ** 2)
        en += np.sum(w * kap * ((gx - gh[..., 0]) ** 2 + (gy - gh[..., 1]) ** 2))
    return np.sqrt(e2), np.sqrt(en)

def mesh_h(X, quad):
    """h = maior aresta da malha."""
    return max(np.max(np.linalg.norm(X[quad[:, i]] - X[quad[:, j]], axis=1))
               for i, j in LOCAL_EDGES)

def convergence_study_a(lcs, orders=(1, 2), structured=False):
    res = {p: {"h": [], "L2": [], "E": [], "ndof": []} for p in orders}
    for lc in lcs:
        X, quad = generate_mesh(lc, structured)
        h = mesh_h(X, quad)
        for p in orders:
            coords, conn, u = solve(X, quad, p)
            eL2, eE = compute_errors(coords, conn, u, p)
            for k, v in zip(("h", "L2", "E", "ndof"), (h, eL2, eE, len(coords))):
                res[p][k].append(v)
            print(f"lc={lc:7.4f} h={h:.4f} Q{p}  nel={len(quad):6d} ndof={len(coords):7d}  "
                  f"errL2={eL2:.3e}  errEnergia={eE:.3e}")
    return res


def plot_convergence_a(res, fname="convergencia_poisson.png"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for p, d in res.items():
        h = np.array(d["h"])
        for a, key, nome in zip(ax, ("L2", "E"), ("Erro $L_2$", "Erro na norma de energia")):
            err = np.array(d[key])
            m = np.polyfit(np.log(h), np.log(err), 1)[0]
            a.loglog(h, err, "o-", label=f"$Q_{p}$  (inclinação ≈ {m:.2f})")
            a.set_xlabel("h"); a.set_ylabel(nome); a.grid(True, which="both", alpha=.3)
            a.set_title(nome)
    for a in ax:
        a.legend()
    plt.tight_layout()
    #plt.savefig(fname, dpi=150)
    #print("Figura salva em", fname)

def plot_solution_a(lc=0.1, p=2, structured=False, fname="solucao_poisson.png"):
    X, quad = generate_mesh(lc, structured)
    coords, conn, u = solve(X, quad, p)
    nv = len(X)
    tri = np.vstack([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])     # só para visualização
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, z, t in zip(ax, (u[:nv], u[:nv] - u_exact(X[:, 0], X[:, 1])), ("Solução FEM", "Erro nodal")):
        c = a.tricontourf(X[:, 0], X[:, 1], tri, z, 30)
        plt.colorbar(c, ax=a); a.set_aspect("equal"); a.set_title(t)
    plt.tight_layout()#; plt.savefig(fname, dpi=150)



