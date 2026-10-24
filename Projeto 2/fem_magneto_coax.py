import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import gmsh

# Dados do problema
MU0 = 4e-7 * np.pi
I = 1.0
A, B, C = 1e-2, np.sqrt(3.0) * 1e-2, 2e-2          # raios a, b, c (m)
MU_R = np.array([1.0, 1.0, 1.0])                   # permeabilidade relativa por região
J_Z = np.array([-I / (np.pi * A ** 2),             # região 0: r < a
                0.0,                               # região 1: a < r < b
                I / (np.pi * (C ** 2 - B ** 2))])  # região 2: b < r < c

def g_dirichlet(x, y):
    """A_z prescrito em r = c."""
    return np.zeros_like(x)

# Solução analítica (válida para mu_r = 1 em todas as regiões)
_K0 = MU0 * I / (2 * np.pi)
_D = C ** 2 - B ** 2

def _A3(r):                                        # b <= r <= c
    return -_K0 / _D * (C ** 2 * np.log(C / r) - (C ** 2 - r ** 2) / 2)

_AB = float(_A3(B))
_AA = _AB - _K0 * np.log(B / A)

def A_exact(x, y):
    r = np.maximum(np.hypot(x, y), 1e-300)
    A1 = _AA - _K0 * (A ** 2 - r ** 2) / (2 * A ** 2)
    A2 = _AB - _K0 * np.log(B / r)
    return np.where(r <= A, A1, np.where(r <= B, A2, _A3(r)))

def dAdr_exact(r):
    r = np.maximum(r, 1e-300)
    return np.where(r <= A, _K0 * r / A ** 2,
                    np.where(r <= B, _K0 / r, _K0 * (C ** 2 - r ** 2) / (_D * r)))

def grad_A_exact(x, y):
    r = np.maximum(np.hypot(x, y), 1e-300)
    d = dAdr_exact(r)
    return d * x / r, d * y / r

def energy_exact():
    """W' = (mu0 I^2 / 4 pi) [1/4 + ln(b/a) + termo da região b<r<c]   (J/m)."""
    t3 = (C ** 4 * np.log(C / B) - C ** 2 * _D + (C ** 4 - B ** 4) / 4) / _D ** 2
    return MU0 * I ** 2 / (4 * np.pi) * (0.25 + np.log(B / A) + t3)

# Geração de malha triangular (6 nós, ordem 2) com Gmsh
def generate_mesh(lc, quarter=False):
    """
    Retorna:
      coords (nn,2) : todos os nós (vértices, pontos médios curvos dos lados)
      conn   (nel,6): conectividade P2 (3 vértices, lados 01, 12, 20), anti-horária;
                      os 3 primeiros nós formam a malha P1
      region (nel,) : 0 = condutor interno, 1 = isolante, 2 = condutor externo
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)       # nós de aresta sobre a geometria
    gmsh.model.add("coax_mag")
    geo = gmsh.model.geo
    O = geo.addPoint(0, 0, 0, lc)

    if not quarter:
        def circle(r):
            p = [geo.addPoint(r, 0, 0, lc), geo.addPoint(0, r, 0, lc),
                 geo.addPoint(-r, 0, 0, lc), geo.addPoint(0, -r, 0, lc)]
            return [geo.addCircleArc(p[i], O, p[(i + 1) % 4]) for i in range(4)]

        def loop(arcs, reverse=False):
            return geo.addCurveLoop([-t for t in reversed(arcs)] if reverse else arcs)

        ra, rb, rc = circle(A), circle(B), circle(C)
        surfs = [geo.addPlaneSurface([loop(ra)]),
                 geo.addPlaneSurface([loop(rb), loop(ra, True)]),
                 geo.addPlaneSurface([loop(rc), loop(rb, True)])]
    else:
        px = [geo.addPoint(r, 0, 0, lc) for r in (A, B, C)]
        py = [geo.addPoint(0, r, 0, lc) for r in (A, B, C)]
        arc = [geo.addCircleArc(px[i], O, py[i]) for i in range(3)]
        lx = [geo.addLine(O, px[0]), geo.addLine(px[0], px[1]), geo.addLine(px[1], px[2])]
        ly = [geo.addLine(O, py[0]), geo.addLine(py[0], py[1]), geo.addLine(py[1], py[2])]
        surf = lambda c: geo.addPlaneSurface([geo.addCurveLoop(c)])
        surfs = [surf([lx[0], arc[0], -ly[0]]),
                 surf([lx[1], arc[1], -ly[1], -arc[0]]),
                 surf([lx[2], arc[2], -ly[2], -arc[1]])]
    geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)

    tags, xyz, _ = gmsh.model.mesh.getNodes()
    table = np.zeros((int(np.max(tags)) + 1, 3))
    table[np.asarray(tags, dtype=np.int64)] = np.asarray(xyz).reshape(-1, 3)

    conn_tags, region = [], []
    for k, s in enumerate(surfs):
        etypes, _, enodes = gmsh.model.mesh.getElements(2, s)
        if list(etypes) != [9]:
            raise RuntimeError("A malha contém elementos que não são triângulos de 6 nós.")
        c6 = np.asarray(enodes[0], dtype=np.int64).reshape(-1, 6)
        conn_tags.append(c6)
        region.append(np.full(len(c6), k))
    gmsh.finalize()

    conn_tags = np.vstack(conn_tags)
    region = np.concatenate(region)
    used = np.unique(conn_tags)                              # descarta nós sem elemento (ex.: centro O)
    conn = np.searchsorted(used, conn_tags)
    coords = table[used][:, :2]
    return coords, orient_ccw(coords, conn), region

def orient_ccw(coords, conn):
    conn = conn.copy()
    a, b, c = (coords[conn[:, k]] for k in range(3))
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    neg = det < 0
    conn[neg] = conn[neg][:, [0, 2, 1, 5, 4, 3]]
    return conn

def build_spaces(coords, conn6):
    """Espaços P1 (vértices, lados retos) e P2 (isoparamétrico, nós curvos do Gmsh)."""
    verts = np.unique(conn6[:, :3])
    remap = -np.ones(len(coords), dtype=np.int64)
    remap[verts] = np.arange(len(verts))
    return {1: (coords[verts], remap[conn6[:, :3]]), 2: (coords, conn6)}

def dirichlet_mask(coords):
    """Nós sobre r = c (A_z = 0). Os eixos de simetria do 1/4 de domínio ficam livres (Neumann)."""
    r = np.hypot(coords[:, 0], coords[:, 1])
    return np.abs(r - C) < 1e-7 * C

# Elemento de referência (triângulo unitário): base de Lagrange P_p, p = 1, 2
LOCAL_EDGES = [(0, 1), (1, 2), (2, 0)]
REF_VERT = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

def ref_nodes(p):
    nodes = [*REF_VERT]
    if p == 2:
        for (i, j) in LOCAL_EDGES:
            nodes.append(0.5 * (REF_VERT[i] + REF_VERT[j]))
    return np.array(nodes)

def basis(p, pts):
    """phi (nq,nb) e dphi (nq,nb,2) em relação a (xi, eta)."""
    mons = [(i, d - i) for d in range(p + 1) for i in range(d + 1)]
    nodes = ref_nodes(p)
    V = np.array([[x ** i * y ** j for (i, j) in mons] for (x, y) in nodes])
    Cm = np.linalg.inv(V)
    x, y = pts[:, 0], pts[:, 1]
    M = np.stack([x ** i * y ** j for (i, j) in mons], axis=1)
    dMx = np.stack([i * x ** (i - 1) * y ** j if i > 0 else 0 * x for (i, j) in mons], axis=1)
    dMy = np.stack([j * x ** i * y ** (j - 1) if j > 0 else 0 * x for (i, j) in mons], axis=1)
    return M @ Cm, np.stack([dMx @ Cm, dMy @ Cm], axis=2)

def quad_tri(n):
    """Gauss no triângulo unitário (Duffy), n x n pontos; pesos somam 1/2."""
    xg, wg = np.polynomial.legendre.leggauss(n)
    xg, wg = 0.5 * (xg + 1), 0.5 * wg
    U, V = np.meshgrid(xg, xg, indexing="ij")
    W = np.outer(wg, wg) * (1 - U)
    return np.stack([U.ravel(), (V * (1 - U)).ravel()], axis=1), W.ravel()

def geometry(coords, conn, xq, p):
    """Mapeamento isoparamétrico de ordem p (afim em P1, quadrático e curvo em P2)."""
    v = coords[conn]
    N, dN = basis(p, xq)
    xp = np.einsum("qk,ekj->eqj", N, v)
    J = np.einsum("ekj,qkm->eqjm", v, dN)                    # J[i,m] = dx_i/dxi_m
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    invJ = np.empty_like(J)
    invJ[..., 0, 0], invJ[..., 0, 1] = J[..., 1, 1] / det, -J[..., 0, 1] / det
    invJ[..., 1, 0], invJ[..., 1, 1] = -J[..., 1, 0] / det, J[..., 0, 0] / det
    return xp, det, invJ

# Montagem e solução com eliminação dos nós de Dirichlet
def assemble(coords, conn, p, region):
    xq, wq = quad_tri(p + 3)
    phi, dphi = basis(p, xq)
    _, det, invJ = geometry(coords, conn, xq, p)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
    w = wq[None, :] * np.abs(det)
    nu = (1.0 / (MU0 * MU_R[region]))[:, None]               # relutividade por elemento
    Ke = np.einsum("eq,eqbi,eqci->ebc", w * nu, G, G)
    fe = np.einsum("eq,qb->eb", w * J_Z[region][:, None], phi)
    nb = conn.shape[1]
    n = len(coords)
    rows = np.repeat(conn, nb, axis=1).ravel()
    cols = np.tile(conn, (1, nb)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    F = np.bincount(conn.ravel(), weights=fe.ravel(), minlength=n)
    return K, F

def solve(coords, conn, p, region):
    K, F = assemble(coords, conn, p, region)
    dmask = dirichlet_mask(coords)
    free, dirn = np.where(~dmask)[0], np.where(dmask)[0]
    u = np.zeros(len(coords))
    u[dirn] = g_dirichlet(coords[dirn, 0], coords[dirn, 1])
    # Sistema reduzido (equações dos nós de Dirichlet eliminadas): Kff uf = Ff - Kfd ud
    rhs = F[free] - K[free][:, dirn] @ u[dirn]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return u

# Energia e erros
def compute_errors(coords, conn, u, p, region, nq=10, chunk=2000):
    xq, wq = quad_tri(nq)
    phi, dphi = basis(p, xq)
    e2 = en = W = 0.0
    for s in range(0, len(conn), chunk):
        c = conn[s:s + chunk]
        nu = (1.0 / (MU0 * MU_R[region[s:s + chunk]]))[:, None]
        xp, det, invJ = geometry(coords, c, xq, p)
        G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
        ue = u[c]
        uh = np.einsum("qb,eb->eq", phi, ue)
        gh = np.einsum("eqbi,eb->eqi", G, ue)
        ex = A_exact(xp[..., 0], xp[..., 1])
        gx, gy = grad_A_exact(xp[..., 0], xp[..., 1])
        w = wq[None, :] * np.abs(det)
        e2 += np.sum(w * (ex - uh) ** 2)
        en += np.sum(w * nu * ((gx - gh[..., 0]) ** 2 + (gy - gh[..., 1]) ** 2))
        W += 0.5 * np.sum(w * nu * (gh[..., 0] ** 2 + gh[..., 1] ** 2))
    return np.sqrt(e2), np.sqrt(en), W

def mesh_h(coords, conn):
    v = conn[:, :3]
    return max(np.max(np.linalg.norm(coords[v[:, i]] - coords[v[:, j]], axis=1))
               for i, j in LOCAL_EDGES)

# Estudo de convergência
def convergence_study(lcs, orders=(1, 2), quarter=False, mesh_fn=generate_mesh):
    fac = 4.0 if quarter else 1.0                            # energia total = 4 x energia do quarto
    Wex = energy_exact()
    res = {p: {"h": [], "L2": [], "E": [], "W": [], "Werr": [], "ndof": []} for p in orders}
    for lc in lcs:
        coords6, conn6, region = mesh_fn(lc, quarter)
        spaces = build_spaces(coords6, conn6)
        h = mesh_h(coords6, conn6)
        for p in orders:
            coords, conn = spaces[p]
            u = solve(coords, conn, p, region)
            eL2, eE, W = compute_errors(coords, conn, u, p, region)
            W *= fac
            Werr = abs(W - Wex) / Wex
            for k, v in zip(("h", "L2", "E", "W", "Werr", "ndof"), (h, eL2, eE, W, Werr, len(coords))):
                res[p][k].append(v)
    print(f"W exata = {Wex:.8e} J/m   (indutância L' = 2W/I^2 = {2*Wex/I**2:.6e} H/m)")
    return res

def plot_convergence(res, fname, titulo=""):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    titles = ("Erro $L_2$ de $A_z$", "Erro na norma de energia", "Erro relativo da energia")
    for p, d in res.items():
        h = np.array(d["h"])
        for a, key, nome in zip(ax, ("L2", "E", "Werr"), titles):
            err = np.array(d[key])
            m = np.polyfit(np.log(h), np.log(err), 1)[0]
            a.loglog(h, err, "o-", label=f"$P_{p}$  (inclinação ≈ {m:.2f})")
            a.set_xlabel("h (m)"); a.set_ylabel(nome); a.set_title(nome + titulo)
            a.grid(True, which="both", alpha=.3)
    for a in ax:
        a.legend()
    plt.tight_layout()

# Visualização: potencial vetor, campo B e perfil radial de |B|
def plot_fields(lc=5e-4, p=2, quarter=False, fname="campos_coaxial.png", mesh_fn=generate_mesh):
    coords6, conn6, region = mesh_fn(lc, quarter)
    coords, conn = build_spaces(coords6, conn6)[p]
    u = solve(coords, conn, p, region)

    # B = ( dA/dy , -dA/dx ) no baricentro de cada elemento
    xq = np.array([[1 / 3, 1 / 3]])
    phi, dphi = basis(p, xq)
    xc, det, invJ = geometry(coords, conn, xq, p)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)[:, 0]
    g = np.einsum("ebi,eb->ei", G, u[conn])
    Bx, By = g[:, 1], -g[:, 0]
    xc = xc[:, 0, :]
    Bmag = np.hypot(Bx, By)

    tri = conn[:, :3]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    c0 = ax[0].tricontourf(coords[:, 0], coords[:, 1], tri, u, 40)
    ax[0].tricontour(coords[:, 0], coords[:, 1], tri, u, 15, colors="k", linewidths=.4)
    plt.colorbar(c0, ax=ax[0], label="$A_z$ (Wb/m)")
    ax[0].set_title("Potencial vetor $A_z$ (linhas = linhas de campo)")

    c1 = ax[1].scatter(xc[:, 0], xc[:, 1], c=Bmag, s=3)
    step = max(1, len(xc) // 700)
    ax[1].quiver(xc[::step, 0], xc[::step, 1], Bx[::step], By[::step], color="k", scale=None, width=.003)
    plt.colorbar(c1, ax=ax[1], label="|B| (T)")
    ax[1].set_title("Campo magnético B")

    r = np.hypot(xc[:, 0], xc[:, 1])
    rr = np.linspace(1e-4 * C, C, 600)
    ax[2].plot(r * 1e2, Bmag * 1e6, ".", ms=2, label="FEM (baricentros)")
    ax[2].plot(rr * 1e2, dAdr_exact(rr) * 1e6, "r-", lw=1, label="analítico")
    ax[2].set_xlabel("r (cm)"); ax[2].set_ylabel("|B| (µT)"); ax[2].legend()
    ax[2].set_title("Perfil radial de |B|"); ax[2].grid(alpha=.3)
    for a in ax[:2]:
        a.set_aspect("equal"); a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    plt.tight_layout()