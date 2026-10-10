"""
Método dos Elementos Finitos com elementos triangulares de Lagrange P1, P2 e P3
Problema: Laplace (sem cargas) no domínio em L, com solução exata
        V = R^(2/3) * sin(2*theta/3 + pi/3),   -pi/2 <= theta <= pi

 - Condições de Dirichlet impostas por ELIMINAÇÃO das equações dos nós de Dirichlet
   (o sistema global só contém os graus de liberdade livres).
 - Malhas triangulares geradas pelo Gmsh (import gmsh dentro do Python).
 - Estudo de convergência: erros em L2 e na norma de energia vs h (log-log).

Dependências: numpy, scipy, matplotlib, gmsh   (pip install gmsh)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import gmsh


# ----------------------------------------------------------------------------
# Solução exata e condição de contorno (função fornecida pelo usuário)
# ----------------------------------------------------------------------------
def u_exact(x, y):
    R = np.hypot(x, y)
    th = np.arctan2(y, x)          # para o L: -pi/2 <= theta <= pi
    return R ** (2.0 / 3.0) * np.sin(2.0 * th / 3.0 + np.pi / 3.0)


def grad_u_exact(x, y):
    R = np.maximum(np.hypot(x, y), 1e-14)
    th = np.arctan2(y, x)
    c = (2.0 / 3.0) * R ** (-1.0 / 3.0)
    a = np.pi / 3.0 - th / 3.0     # = (2*th/3 + pi/3) - th
    return c * np.sin(a), c * np.cos(a)


def g_dirichlet(x, y):
    """Potencial prescrito na fronteira (aqui: a própria solução exata)."""
    return u_exact(x, y)


# ----------------------------------------------------------------------------
# Geração de malha com Gmsh
# ----------------------------------------------------------------------------
def generate_mesh(lc, grade=1.0):
    """
    Malha triangular (P1) do domínio em L.
    lc    : tamanho característico dos elementos
    grade : fator de tamanho no vértice reentrante (0,0). grade<1 refina a malha
            em torno da singularidade (ex.: 0.1).
    Retorna: X (nv,2) coordenadas, tri (nt,3) conectividade (anti-horária).
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("L")
    pts = [(-1, 1), (-1, 0), (0, 0), (0, -1), (1, -1), (1, 1)]
    ptag = []
    for (x, y) in pts:
        s = lc * grade if (x == 0 and y == 0) else lc
        ptag.append(gmsh.model.geo.addPoint(x, y, 0.0, s))
    lines = [gmsh.model.geo.addLine(ptag[i], ptag[(i + 1) % 6]) for i in range(6)]
    loop = gmsh.model.geo.addCurveLoop(lines)
    gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    tags, coords, _ = gmsh.model.mesh.getNodes()
    _, _, enodes = gmsh.model.mesh.getElements(2)     # triângulos de 3 nós
    gmsh.finalize()

    tri_tags = np.asarray(enodes[0], dtype=np.int64).reshape(-1, 3)
    used = np.unique(tri_tags)
    tag2xyz = {int(t): c for t, c in zip(tags, np.asarray(coords).reshape(-1, 3))}
    remap = {int(t): i for i, t in enumerate(used)}
    X = np.array([tag2xyz[int(t)][:2] for t in used])
    tri = np.vectorize(remap.get)(tri_tags)
    return X, orient_ccw(X, tri)


def orient_ccw(X, tri):
    tri = tri.copy()
    a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    neg = det < 0
    tri[neg, 1], tri[neg, 2] = tri[neg, 2].copy(), tri[neg, 1].copy()
    return tri


# ----------------------------------------------------------------------------
# Elemento de referência: base de Lagrange P_p (p = 1, 2, 3)
# ----------------------------------------------------------------------------
LOCAL_EDGES = [(0, 1), (1, 2), (2, 0)]
REF_VERT = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def ref_nodes(p):
    """Nós de referência: vértices, nós de aresta (arestas 01,12,20), nó interno."""
    nodes = [*REF_VERT]
    for (i, j) in LOCAL_EDGES:
        for k in range(1, p):
            nodes.append(REF_VERT[i] + k / p * (REF_VERT[j] - REF_VERT[i]))
    if p == 3:
        nodes.append(np.array([1 / 3, 1 / 3]))
    return np.array(nodes)


def monomials(p):
    return [(i, d - i) for d in range(p + 1) for i in range(d + 1)]


def basis(p, pts):
    """phi (nq,nb) e dphi (nq,nb,2) no elemento de referência."""
    mons = monomials(p)
    nodes = ref_nodes(p)
    V = np.array([[x ** i * y ** j for (i, j) in mons] for (x, y) in nodes])
    C = np.linalg.inv(V)                       # phi_k = sum_m C[m,k] * mono_m
    x, y = pts[:, 0], pts[:, 1]
    M = np.stack([x ** i * y ** j for (i, j) in mons], axis=1)
    dMx = np.stack([i * x ** max(i - 1, 0) * y ** j if i > 0 else 0 * x for (i, j) in mons], axis=1)
    dMy = np.stack([j * x ** i * y ** max(j - 1, 0) if j > 0 else 0 * x for (i, j) in mons], axis=1)
    return M @ C, np.stack([dMx @ C, dMy @ C], axis=2)


def quad_tri(n):
    """Quadratura de Gauss em triângulo de referência (Duffy), n x n pontos."""
    xg, wg = np.polynomial.legendre.leggauss(n)
    xg, wg = 0.5 * (xg + 1), 0.5 * wg
    U, V = np.meshgrid(xg, xg, indexing="ij")
    W = np.outer(wg, wg) * (1 - U)
    return np.stack([U.ravel(), (V * (1 - U)).ravel()], axis=1), W.ravel()


# ----------------------------------------------------------------------------
# Numeração dos graus de liberdade P_p a partir da malha P1
# ----------------------------------------------------------------------------
def build_space(X, tri, p):
    nt, nv = len(tri), len(X)
    e = np.stack([tri[:, [i, j]] for (i, j) in LOCAL_EDGES], axis=1)      # (nt,3,2)
    uniq, inv = np.unique(np.sort(e, axis=2).reshape(-1, 2), axis=0, return_inverse=True)
    inv = inv.reshape(nt, 3)
    ne = len(uniq)
    bnd_edge = np.bincount(inv.ravel(), minlength=ne) == 1                # aresta em 1 só elemento

    nper = p - 1
    nint = 1 if p == 3 else 0
    ndof = nv + ne * nper + nt * nint
    coords = np.zeros((ndof, 2))
    coords[:nv] = X
    conn = np.zeros((nt, 3 + 3 * nper + nint), dtype=np.int64)
    conn[:, :3] = tri

    a, b = X[uniq[:, 0]], X[uniq[:, 1]]
    for k in range(nper):
        coords[nv + np.arange(ne) * nper + k] = a + (k + 1) / p * (b - a)
    for le, (i, j) in enumerate(LOCAL_EDGES):
        fwd = tri[:, i] < tri[:, j]       # sentido local coincide com o global (menor->maior)?
        for k in range(nper):
            kk = np.where(fwd, k, nper - 1 - k)
            conn[:, 3 + le * nper + k] = nv + inv[:, le] * nper + kk
    if nint:
        idx = nv + ne * nper + np.arange(nt)
        conn[:, -1] = idx
        coords[idx] = X[tri].mean(axis=1)

    # nós de Dirichlet = todos os nós sobre arestas de fronteira
    dir_mask = np.zeros(ndof, dtype=bool)
    dir_mask[uniq[bnd_edge].ravel()] = True
    for k in range(nper):
        dir_mask[nv + np.where(bnd_edge)[0] * nper + k] = True
    return coords, conn, dir_mask


def geometry(coords, conn):
    """Jacobiano (afim): det e inversa por elemento."""
    v = coords[conn[:, :3]]
    J = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]], axis=2)         # colunas
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    invJ = np.empty_like(J)
    invJ[:, 0, 0], invJ[:, 0, 1] = J[:, 1, 1] / det, -J[:, 0, 1] / det
    invJ[:, 1, 0], invJ[:, 1, 1] = -J[:, 1, 0] / det, J[:, 0, 0] / det
    return v[:, 0], J, det, invJ


# ----------------------------------------------------------------------------
# Montagem e solução com eliminação dos nós de Dirichlet
# ----------------------------------------------------------------------------
def assemble_stiffness(coords, conn, p):
    xq, wq = quad_tri(p + 1)
    _, dphi = basis(p, xq)
    _, _, det, invJ = geometry(coords, conn)
    G = np.einsum("qbk,eki->eqbi", dphi, invJ)                           # grad físico
    Ke = np.einsum("q,eqbi,eqci,e->ebc", wq, G, G, np.abs(det))
    nb = conn.shape[1]
    rows = np.repeat(conn, nb, axis=1).ravel()
    cols = np.tile(conn, (1, nb)).ravel()
    n = len(coords)
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def solve(X, tri, p):
    coords, conn, dmask = build_space(X, tri, p)
    K = assemble_stiffness(coords, conn, p)
    free = np.where(~dmask)[0]
    dirn = np.where(dmask)[0]
    u = np.zeros(len(coords))
    u[dirn] = g_dirichlet(coords[dirn, 0], coords[dirn, 1])
    # Sistema reduzido (equações dos nós de Dirichlet eliminadas): Kff uf = f - Kfd ud, f = 0
    rhs = -K[free][:, dirn] @ u[dirn]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return coords, conn, u


# ----------------------------------------------------------------------------
# Erros em L2 e na norma de energia (|u-uh|_{H1})
# ----------------------------------------------------------------------------
def compute_errors(coords, conn, u, p, nq=10, chunk=4000):
    xq, wq = quad_tri(nq)
    phi, dphi = basis(p, xq)
    e2 = h1 = 0.0
    for s in range(0, len(conn), chunk):
        c = conn[s:s + chunk]
        x0, J, det, invJ = geometry(coords, c)
        xp = x0[:, None, :] + np.einsum("eij,qj->eqi", J, xq)
        G = np.einsum("qbk,eki->eqbi", dphi, invJ)
        ue = u[c]
        uh = np.einsum("qb,eb->eq", phi, ue)
        gh = np.einsum("eqbi,eb->eqi", G, ue)
        ex = u_exact(xp[..., 0], xp[..., 1])
        gx, gy = grad_u_exact(xp[..., 0], xp[..., 1])
        a = np.abs(det)
        e2 += np.einsum("q,eq,e->", wq, (ex - uh) ** 2, a)
        h1 += np.einsum("q,eq,e->", wq, (gx - gh[..., 0]) ** 2 + (gy - gh[..., 1]) ** 2, a)
    return np.sqrt(e2), np.sqrt(h1)


def mesh_h(X, tri):
    d = [np.linalg.norm(X[tri[:, i]] - X[tri[:, j]], axis=1) for i, j in LOCAL_EDGES]
    return max(np.max(x) for x in d)


# ----------------------------------------------------------------------------
# Estudo de convergência
# ----------------------------------------------------------------------------
def convergence_study(lcs, orders=(1, 2, 3), grade=1.0):
    res = {p: {"h": [], "L2": [], "E": [], "ndof": []} for p in orders}
    for lc in lcs:
        X, tri = generate_mesh(lc, grade)
        h = mesh_h(X, tri)
        for p in orders:
            coords, conn, u = solve(X, tri, p)
            eL2, eE = compute_errors(coords, conn, u, p)
            for k, v in zip(("h", "L2", "E", "ndof"), (h, eL2, eE, len(coords))):
                res[p][k].append(v)
            print(f"lc={lc:7.4f} h={h:.4f} P{p}  ndof={len(coords):7d}  "
                  f"errL2={eL2:.3e}  errEnergia={eE:.3e}")
    return res


def plot_convergence(res, fname="convergencia.png"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for p, d in res.items():
        h = np.array(d["h"])
        for a, key, nome in zip(ax, ("L2", "E"), ("Erro $L_2$", "Erro na norma de energia")):
            err = np.array(d[key])
            m = np.polyfit(np.log(h), np.log(err), 1)[0]
            a.loglog(h, err, "o-", label=f"$P_{p}$  (inclinação ≈ {m:.2f})")
            a.set_xlabel("h"); a.set_ylabel(nome); a.grid(True, which="both", alpha=.3)
            a.set_title(nome)
    for a in ax:
        a.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print("Figura salva em", fname)


def plot_solution(lc=0.05, p=2, fname="solucao.png"):
    X, tri = generate_mesh(lc)
    coords, conn, u = solve(X, tri, p)
    nv = len(X)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, z, t in zip(ax, (u[:nv], u[:nv] - u_exact(X[:, 0], X[:, 1])), ("Solução FEM", "Erro nodal")):
        c = a.tricontourf(X[:, 0], X[:, 1], tri, z, 30)
        plt.colorbar(c, ax=a); a.set_aspect("equal"); a.set_title(t)
    plt.tight_layout(); plt.savefig(fname, dpi=150)


if __name__ == "__main__":
    lcs = [0.4, 0.2, 0.1, 0.05, 0.025]
    res = convergence_study(lcs)                 # malhas quase uniformes
    plot_convergence(res, "convergencia.png")
    plot_solution()
    # Opcional: malha refinada perto da singularidade (melhora as taxas de P2 e P3)
    # res_g = convergence_study(lcs, grade=0.05)
    # plot_convergence(res_g, "convergencia_graduada.png")
    plt.show()
