import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import jv, hankel2
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import gmsh

F0 = 300e6
C0 = 299792458.0
LAM = C0 / F0
K0 = 2 * np.pi / LAM
RA = LAM / 2                         # raio do cilindro PEC
H_MAX = LAM / 20
R_REFL, THETA_I = 1e-4, 0.0
L_REF = LAM / 2
SIGMA0 = -np.log(R_REFL) / (2 * L_REF * np.cos(THETA_I))     # n = 0, L = lambda/2, R = 1e-4

def design_sigma0(n, L, R=R_REFL, theta=THETA_I):
    return -np.log(R) * (n + 1) / (2 * np.cos(theta) * L ** (n + 1))

def reflection(n, L, theta=THETA_I, sigma0=SIGMA0):
    return np.exp(-2 * np.cos(theta) * sigma0 * L ** (n + 1) / (n + 1))

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

def generate_mesh(x0, L, h=H_MAX):
    xo = x0 + L
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.add("pml")
    geo = gmsh.model.geo
    O = geo.addPoint(0, 0, 0, h)

    def circle(r):
        p = [geo.addPoint(r, 0, 0, h), geo.addPoint(0, r, 0, h),
             geo.addPoint(-r, 0, 0, h), geo.addPoint(0, -r, 0, h)]
        return [geo.addCircleArc(p[i], O, p[(i + 1) % 4]) for i in range(4)]

    def square(c):
        p = [geo.addPoint(x, y, 0, h) for (x, y) in ((-c, -c), (c, -c), (c, c), (-c, c))]
        return [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]

    loop = lambda cs, rev=False: geo.addCurveLoop([-t for t in reversed(cs)] if rev else cs)
    cyl, sq_in, sq_out = circle(RA), square(x0), square(xo)
    s_phys = geo.addPlaneSurface([loop(sq_in), loop(cyl, True)])
    s_pml = geo.addPlaneSurface([loop(sq_out), loop(sq_in, True)])
    geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    table = np.zeros((int(np.max(tags)) + 1, 3))
    table[np.asarray(tags, dtype=np.int64)] = np.asarray(xyz).reshape(-1, 3)
    conn_tags, region = [], []
    for reg, s in ((0, s_phys), (1, s_pml)):
        etypes, _, enodes = gmsh.model.mesh.getElements(2, s)
        if list(etypes) != [9]:
            raise RuntimeError("A malha contém elementos que não são triângulos de 6 nós.")
        c6 = np.asarray(enodes[0], dtype=np.int64).reshape(-1, 6)
        conn_tags.append(c6)
        region.append(np.full(len(c6), reg))
    gmsh.finalize()
    conn_tags, region = np.vstack(conn_tags), np.concatenate(region)
    used = np.unique(conn_tags)
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

def sigma_profile(t, x0, n, sigma0=SIGMA0):
    """sigma(t) = sigma0 (|t| - x0)^n  para |t| >= x0, 0 caso contrário."""
    d = np.maximum(np.abs(t) - x0, 0.0)
    return np.where(np.abs(t) > x0, sigma0 * d ** n, 0.0)

def assemble(coords, conn, p, x0, n, sigma0=SIGMA0):
    xq, wq = quad_tri(p + 4)
    phi, dphi = basis(p, xq)
    xp, det, invJ = geometry(coords, conn, xq, p)
    w = wq[None, :] * np.abs(det)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
    sx = 1 - 1j * sigma_profile(xp[..., 0], x0, n, sigma0) / K0
    sy = 1 - 1j * sigma_profile(xp[..., 1], x0, n, sigma0) / K0
    Ae = (np.einsum("eq,eqb,eqc->ebc", w * sy / sx, G[..., 0], G[..., 0])
          + np.einsum("eq,eqb,eqc->ebc", w * sx / sy, G[..., 1], G[..., 1])
          - K0 ** 2 * np.einsum("eq,qb,qc->ebc", w * sx * sy, phi, phi))
    nb = conn.shape[1]
    nn = len(coords)
    rows = np.repeat(conn, nb, axis=1).ravel()
    cols = np.tile(conn, (1, nb)).ravel()
    return sp.coo_matrix((Ae.ravel(), (rows, cols)), shape=(nn, nn)).tocsr()

def solve(coords, conn, p, x0, L, n, sigma0=SIGMA0):
    A = assemble(coords, conn, p, x0, n, sigma0)
    r = np.hypot(coords[:, 0], coords[:, 1])
    xo = x0 + L
    pec = np.abs(r - RA) < 1e-8
    outer = (np.abs(np.abs(coords[:, 0]) - xo) < 1e-8) | (np.abs(np.abs(coords[:, 1]) - xo) < 1e-8)
    dmask = pec | outer
    free, dirn = np.where(~dmask)[0], np.where(dmask)[0]
    u = np.zeros(len(coords), dtype=complex)
    u[dirn[pec[dirn]]] = -u_inc(coords[dirn[pec[dirn]], 0], coords[dirn[pec[dirn]], 1])   # PEC
    # (nós do contorno externo da PML: u = 0)
    rhs = -A[free][:, dirn] @ u[dirn]
    u[free] = spla.spsolve(A[free][:, free].tocsc(), rhs)
    return u

def rms_error(coords, conn, u, x0):
    verts = np.unique(conn[:, :3])
    x, y = coords[verts, 0], coords[verts, 1]
    ok = (np.hypot(x, y) > RA + 1e-8) & (np.abs(x) <= x0 + 1e-8) & (np.abs(y) <= x0 + 1e-8)
    v = verts[ok]
    return np.sqrt(np.mean(np.abs(u_exact(coords[v, 0], coords[v, 1]) - u[v]) ** 2))

def run_cases(cases, h=H_MAX, orders=(1, 2), redesign=False):
    meshes, out = {}, {}
    for (x0, L, n) in cases:
        if (x0, L) not in meshes:
            c6, cn6, _ = generate_mesh(x0, L, h)
            meshes[(x0, L)] = build_spaces(c6, cn6)
        out[(x0, L, n)] = {}
        for p in orders:
            coords, conn = meshes[(x0, L)][p]
            sig = design_sigma0(n, L) if redesign else SIGMA0
            u = solve(coords, conn, p, x0, L, n, sig)
            out[(x0, L, n)][p] = rms_error(coords, conn, u, x0)
    return out

def experiments(h=H_MAX):
    lam = LAM
    a = [(1 * lam, 0.5 * lam, n) for n in (1, 2, 3)]
    b = [(x0 * lam, 0.5 * lam, 2) for x0 in (0.75, 1.0, 1.2)]
    c = [(1 * lam, L * lam, 2) for L in (0.2, 0.3, 0.4)]
    ref = [(1 * lam, 0.5 * lam, 0)]
    todos = a + b + c + ref
    return {"a": a, "b": b, "c": c, "ref": ref}, run_cases(todos, h), run_cases(todos, h, redesign=True)

def print_tables(groups, res, res_re):
    print(f"sigma0 fixo = {SIGMA0:.4f} 1/m (= {SIGMA0*LAM:.4f}/lambda; R = {R_REFL:g}, theta = 0, "
          f"L = lambda/2, n = 0);  h_max = lambda/20")
    print("R teórico: coeficiente de reflexão do caso com sigma0 fixo.  "
          "'P2 (sigma0 reprojetado)': mesmo caso com sigma0 recalculado para R = 1e-4.")

    def table(titulo, lista, chave):
        print(f"\n{titulo}")
        print(f"  {chave:>16s} {'R teórico':>10s} {'erro P1':>11s} {'erro P2':>11s} {'P2 (sigma0 reproj.)':>20s}")
        for (x0, L, n) in lista:
            rot = {"n": f"n = {n}", "x0": f"x0 = {x0/LAM:.2f} lambda", "L": f"L = {L/LAM:.1f} lambda"}[chave]
            print(f"  {rot:>16s} {reflection(n, L):10.3e} {res[(x0, L, n)][1]:11.3e} "
                  f"{res[(x0, L, n)][2]:11.3e} {res_re[(x0, L, n)][2]:20.3e}")

    table("(a) x0 = 1 lambda, L = lambda/2, perfil polinomial n = 1, 2, 3 (n = 0: referência do projeto de sigma0)",
          groups["ref"] + groups["a"], "n")
    table("(b) L = lambda/2, n = 2, distância x0 das faces internas da PML", groups["b"], "x0")
    table("(c) x0 = 1 lambda, n = 2, espessura L da PML", groups["c"], "L")

def plot_summary(groups, res, res_re, fname="pml_erros.png"):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    spec = (("a", lambda t: t[2], "n (perfil polinomial)", "(a) x0 = 1λ, L = λ/2"),
            ("b", lambda t: t[0] / LAM, r"$x_0/\lambda$", "(b) L = λ/2, n = 2"),
            ("c", lambda t: t[1] / LAM, r"$L/\lambda$", "(c) x0 = 1λ, n = 2"))
    for a_, (g, key, xl, tit) in zip(ax, spec):
        xs = [key(t) for t in groups[g]]
        for p, st in ((1, "o--"), (2, "s-")):
            a_.semilogy(xs, [res[t][p] for t in groups[g]], st, label=f"P{p}, sigma0 fixo")
        a_.semilogy(xs, [res_re[t][2] for t in groups[g]], "g^:", label="P2, sigma0 reprojetado")
        a_.set_xlabel(xl); a_.set_ylabel("erro RMS"); a_.set_title(tit)
        a_.grid(alpha=.3, which="both"); a_.legend()
    plt.tight_layout()

def plot_field(x0=1.0 * LAM, L=0.5 * LAM, n=2, p=2, h=H_MAX, fname="pml_campo.png"):
    c6, cn6, region = generate_mesh(x0, L, h)
    coords, conn = build_spaces(c6, cn6)[p]
    u = solve(coords, conn, p, x0, L, n)
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], conn[:, :3])
    ue = u_exact(coords[:, 0], coords[:, 1])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    c = ax[0].tricontourf(tri, np.real(u), 30, cmap="RdBu_r")
    plt.colorbar(c, ax=ax[0]); ax[0].set_title(r"Re$(E_z^{esp})$ - FEM (PML incluída)")
    c = ax[1].tricontourf(tri, np.abs(u), 30)
    plt.colorbar(c, ax=ax[1]); ax[1].set_title(r"$|E_z^{esp}|$ - FEM")
    phys = (np.abs(coords[:, 0]) <= x0 + 1e-8) & (np.abs(coords[:, 1]) <= x0 + 1e-8)
    elem_phys = phys[conn[:, :3]].all(axis=1)
    trp = mtri.Triangulation(coords[:, 0], coords[:, 1], conn[elem_phys, :3])
    c = ax[2].tricontourf(trp, np.abs(u - ue), 30)
    plt.colorbar(c, ax=ax[2]); ax[2].set_title(r"$|u_h - u_{ex}|$ no domínio físico")
    for a_ in ax:
        a_.set_aspect("equal"); a_.set_xlabel("x (m)"); a_.set_ylabel("y (m)")
        a_.plot([-x0, x0, x0, -x0, -x0], [-x0, -x0, x0, x0, -x0], "k--", lw=.8)
    plt.tight_layout()