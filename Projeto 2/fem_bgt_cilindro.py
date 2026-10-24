import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import jv, hankel2
import matplotlib.pyplot as plt
import gmsh

F0 = 300e6
C0 = 299792458.0
LAM = C0 / F0
K0 = 2 * np.pi / LAM
RA = LAM / 2                      # raio do cilindro PEC
H_MAX = LAM / 20
RS = (0.75, 1.0, 1.25, 1.5)       # raios externos em unidades de lambda

def u_exact(x, y, nmax=60):
    r = np.maximum(np.hypot(x, y), 1e-300)
    phi = np.arctan2(y, x)
    u = np.zeros(np.shape(x), dtype=complex)
    for n in range(nmax + 1):
        eps = 1.0 if n == 0 else 2.0
        c = -eps * (-1j) ** n * jv(n, K0 * RA) / hankel2(n, K0 * RA)
        u += c * hankel2(n, K0 * r) * np.cos(n * phi)
    return u

def u_inc(x, y):
    return np.exp(-1j * K0 * x)

def generate_mesh(R, h=H_MAX):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.add("coroa")
    geo = gmsh.model.geo
    O = geo.addPoint(0, 0, 0, h)

    def circle(r):
        p = [geo.addPoint(r, 0, 0, h), geo.addPoint(0, r, 0, h),
             geo.addPoint(-r, 0, 0, h), geo.addPoint(0, -r, 0, h)]
        return [geo.addCircleArc(p[i], O, p[(i + 1) % 4]) for i in range(4)]

    loop = lambda arcs, rev=False: geo.addCurveLoop([-t for t in reversed(arcs)] if rev else arcs)
    s = geo.addPlaneSurface([loop(circle(R)), loop(circle(RA), True)])
    geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    table = np.zeros((int(np.max(tags)) + 1, 3))
    table[np.asarray(tags, dtype=np.int64)] = np.asarray(xyz).reshape(-1, 3)
    etypes, _, enodes = gmsh.model.mesh.getElements(2, s)
    if list(etypes) != [9]:
        raise RuntimeError("A malha contém elementos que não são triângulos de 6 nós.")
    c6 = np.asarray(enodes[0], dtype=np.int64).reshape(-1, 6)
    gmsh.finalize()
    used = np.unique(c6)
    conn = np.searchsorted(used, c6)
    coords = table[used][:, :2]
    return coords, orient_ccw(coords, conn)

def orient_ccw(coords, conn):
    conn = conn.copy()
    a, b, c = (coords[conn[:, k]] for k in range(3))
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    neg = det < 0
    conn[neg] = conn[neg][:, [0, 2, 1, 5, 4, 3]]
    return conn

def build_spaces(coords, conn6):
    verts = np.unique(conn6[:, :3])
    remap = -np.ones(len(coords), dtype=np.int64)
    remap[verts] = np.arange(len(verts))
    return {1: (coords[verts], remap[conn6[:, :3]]), 2: (coords, conn6)}

LOCAL_EDGES = [(0, 1), (1, 2), (2, 0)]
REF_VERT = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

def ref_nodes(p):
    nodes = [*REF_VERT]
    if p == 2:
        for (i, j) in LOCAL_EDGES:
            nodes.append(0.5 * (REF_VERT[i] + REF_VERT[j]))
    return np.array(nodes)

def basis(p, pts):
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
    xg, wg = np.polynomial.legendre.leggauss(n)
    xg, wg = 0.5 * (xg + 1), 0.5 * wg
    U, V = np.meshgrid(xg, xg, indexing="ij")
    W = np.outer(wg, wg) * (1 - U)
    return np.stack([U.ravel(), (V * (1 - U)).ravel()], axis=1), W.ravel()

def geometry(coords, conn, xq, p):
    v = coords[conn]
    N, dN = basis(p, xq)
    xp = np.einsum("qk,ekj->eqj", N, v)
    J = np.einsum("ekj,qkm->eqjm", v, dN)
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    invJ = np.empty_like(J)
    invJ[..., 0, 0], invJ[..., 0, 1] = J[..., 1, 1] / det, -J[..., 0, 1] / det
    invJ[..., 1, 0], invJ[..., 1, 1] = -J[..., 1, 0] / det, J[..., 0, 0] / det
    return xp, det, invJ

def to_sparse(Ke, conn, n):
    nb = conn.shape[1]
    rows = np.repeat(conn, nb, axis=1).ravel()
    cols = np.tile(conn, (1, nb)).ravel()
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()

def assemble(coords, conn, p):
    xq, wq = quad_tri(p + 3)
    phi, dphi = basis(p, xq)
    _, det, invJ = geometry(coords, conn, xq, p)
    w = wq[None, :] * np.abs(det)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
    Ke = np.einsum("eq,eqbi,eqci->ebc", w, G, G) - K0 ** 2 * np.einsum("eq,qb,qc->ebc", w, phi, phi)
    return to_sparse(Ke, conn, len(coords))

def circle_edges(coords, conn, p, R, tol=1e-8):
    r = np.hypot(coords[:, 0], coords[:, 1])
    out = []
    for le, (i, j) in enumerate(LOCAL_EDGES):
        vi, vj = conn[:, i], conn[:, j]
        on = (np.abs(r[vi] - R) < tol) & (np.abs(r[vj] - R) < tol)
        nodes = np.stack([vi, vj], 1) if p == 1 else np.stack([vi, vj, conn[:, 3 + le]], 1)
        out.append(nodes[on])
    return np.vstack(out)

def edge_matrices(coords, edges, p, n=4):
    sg, wg = np.polynomial.legendre.leggauss(n)
    s, w = 0.5 * (sg + 1), 0.5 * wg
    if p == 1:
        N = np.stack([1 - s, s], 1)
        dN = np.stack([-np.ones_like(s), np.ones_like(s)], 1)
    else:
        N = np.stack([(1 - s) * (1 - 2 * s), s * (2 * s - 1), 4 * s * (1 - s)], 1)
        dN = np.stack([-3 + 4 * s, 4 * s - 1, 4 - 8 * s], 1)
    X = coords[edges]
    dl = np.linalg.norm(np.einsum("qb,ebj->eqj", dN, X), axis=2)       # |dx/ds|
    Me = np.einsum("q,eq,qb,qc->ebc", w, dl, N, N)
    Ke = np.einsum("q,eq,qb,qc->ebc", w, 1.0 / dl, dN, dN)
    return Me, Ke

def bgt_matrix(coords, conn, p, R, order):
    edges = circle_edges(coords, conn, p, R)
    Me, Ke = edge_matrices(coords, edges, p)
    n = len(coords)
    Mg, Kg = to_sparse(Me, edges, n), to_sparse(Ke, edges, n)
    jk = 1j * K0
    alpha = jk + 1.0 / (2 * R)
    if order == 0:
        return jk * Mg
    if order == 1:
        return alpha * Mg
    c2 = 1.0 / (2 * R * (jk * R + 1.0))
    return (alpha - c2 / 4) * Mg + c2 * R ** 2 * Kg

def solve(coords, conn, p, R, order):
    A = (assemble(coords, conn, p) + bgt_matrix(coords, conn, p, R, order)).tocsr()
    r = np.hypot(coords[:, 0], coords[:, 1])
    dmask = np.abs(r - RA) < 1e-8
    free, dirn = np.where(~dmask)[0], np.where(dmask)[0]
    u = np.zeros(len(coords), dtype=complex)
    u[dirn] = -u_inc(coords[dirn, 0], coords[dirn, 1])                   # E_esp = -E_inc sobre o PEC
    rhs = -A[free][:, dirn] @ u[dirn]
    u[free] = spla.spsolve(A[free][:, free].tocsc(), rhs)
    return u

def rms_error(coords, conn, u):
    verts = np.unique(conn[:, :3])
    r = np.hypot(coords[verts, 0], coords[verts, 1])
    v = verts[r > RA + 1e-8]
    return np.sqrt(np.mean(np.abs(u_exact(coords[v, 0], coords[v, 1]) - u[v]) ** 2))

def run_table(orders_fem=(1, 2), h=H_MAX, verbose=True):
    err = {p: {} for p in orders_fem}
    for Rl in RS:
        R = Rl * LAM
        coords6, conn6 = generate_mesh(R, h)
        spaces = build_spaces(coords6, conn6)
        for p in orders_fem:
            coords, conn = spaces[p]
            for order in (0, 1, 2):
                u = solve(coords, conn, p, R, order)
                err[p][(Rl, order)] = rms_error(coords, conn, u)
        if verbose:
            print(f"R = {Rl:4.2f} lambda: malha com {len(coords6)} nós, {len(conn6)} elementos")
    return err

def print_table(err, p):
    print(f"\nErro RMS (P{p}, h_max = lambda/20)")
    print(f"  {'R/lambda':>9s} {'BGT-0':>12s} {'BGT-1':>12s} {'BGT-2':>12s}")
    for Rl in RS:
        print(f"  {Rl:9.2f} " + " ".join(f"{err[p][(Rl, o)]:12.4e}" for o in (0, 1, 2)))

def plot_phi(err, p=2, fname="Ez_x_phi.png", h=H_MAX):
    (Rl, ordem_best), e_best = min(err[p].items(), key=lambda kv: kv[1])
    R = Rl * LAM
    print(f"\nMenor erro (P{p}): R = {Rl} lambda, BGT-{ordem_best}, erro = {e_best:.3e}")
    coords6, conn6 = generate_mesh(R, h)
    coords, conn = build_spaces(coords6, conn6)[p]
    r = np.hypot(coords[:, 0], coords[:, 1])
    idx = np.where(np.abs(r - R) < 1e-8)[0]
    phi = np.mod(np.arctan2(coords[idx, 1], coords[idx, 0]), 2 * np.pi)
    o = np.argsort(phi)
    idx, phi = idx[o], phi[o]
    pp = np.linspace(0, 2 * np.pi, 721)
    uex = u_exact(R * np.cos(pp), R * np.sin(pp))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a, f, nome in ((ax[0], np.real, "Parte real"), (ax[1], np.imag, "Parte imaginária")):
        a.plot(np.degrees(pp), f(uex), "k-", lw=2, label="analítico")
        for order, st in ((0, "r--"), (1, "g-."), (2, "b:")):
            u = solve(coords, conn, p, R, order)
            a.plot(np.degrees(phi), f(u[idx]), st, lw=1.6, label=f"BGT-{order}")
        a.set_xlabel(r"$\phi$ (graus)"); a.set_ylabel(r"$E_z^{esp}$")
        a.set_title(f"{nome} de $E_z^{{esp}}$ em r = {Rl}$\\lambda$ (FEM P{p})")
        a.set_xlim(0, 360); a.grid(alpha=.3); a.legend()
    plt.tight_layout()