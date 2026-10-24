import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
import gmsh

MU0 = 4e-7 * np.pi
MUR = 1.0e4                 # permeabilidade relativa da blindagem
A_IN = 0.10                 # raio interno a (m)
R_EXT = 5 * A_IN            # raio máximo do domínio = 5a
B0 = 0.1                    # T
H0 = B0 / MU0               # A/m
ALVO = 1e-3                 # campo interno / campo externo desejado

def g_dirichlet(x, y):
    """A_phi prescrito: 0 no eixo e B0 rho/2 em r = 5a (campo uniforme B0 z)."""
    return 0.5 * B0 * x

def shielding_factor(b, mur=MUR, a=A_IN):
    """|H_interno| / H0 (fórmula fechada do enunciado)."""
    return 9 * mur / ((2 * mur + 1) * (mur + 2) - 2 * (a / b) ** 3 * (mur - 1) ** 2)

class Shield:
    def __init__(self, b, mur=MUR, a=A_IN, finite=False, R=R_EXT):
        self.a, self.b, self.mur, self.R, self.finite = a, b, mur, R, finite
        mu = mur
        # incógnitas (delta, beta, gamma, alpha, h)
        M = np.zeros((5, 5))
        rhs = np.zeros(5)
        M[0] = [1, -1, -1 / a ** 3, 0, 0]                       # H_theta contínuo em r = a
        M[1] = [1, -mu, 2 * mu / a ** 3, 0, 0]                  # B_r contínuo em r = a  (-delta = mu(-beta+2gamma/a^3))
        M[1] *= 1.0
        M[2] = [0, 1, 1 / b ** 3, -1 / b ** 3, 1]               # H_theta contínuo em r = b
        M[3] = [0, -mu, 2 * mu / b ** 3, -2 / b ** 3, -1]       # B_r contínuo em r = b
        if finite:                                              # A(R) = B0 R/2  <=>  h + 2 alpha/R^3 = H0
            M[4] = [0, 0, 0, 2 / R ** 3, 1]
            rhs[4] = H0
        else:                                                   # campo uniforme no infinito: h = H0
            M[4] = [0, 0, 0, 0, 1]
            rhs[4] = H0
        self.delta, self.beta, self.gamma, self.alpha, self.h = np.linalg.solve(M, rhs)

    def closed_form_check(self):
        """Compara os coeficientes com as expressões fechadas do enunciado (domínio infinito)."""
        a, b, mu = self.a, self.b, self.mur
        d = -H0 * 9 * mu / ((2 * mu + 1) * (mu + 2) - 2 * (a ** 3 / b ** 3) * (mu - 1) ** 2)
        be = (2 * mu + 1) * d / (3 * mu)
        ga = a ** 3 * (mu - 1) * d / (3 * mu)
        al = -(2 * mu + 1) * (mu - 1) * (b ** 3 - a ** 3) * d / (9 * mu)
        return {k: (v, w) for k, v, w in zip(("delta", "beta", "gamma", "alpha"),
                                             (d, be, ga, al),
                                             (self.delta, self.beta, self.gamma, self.alpha))}

    def _coeffs(self, r):
        """(c_r, c_t, mu_r) por ponto: H_r = c_r cos(th), H_th = c_t sin(th)."""
        a, b = self.a, self.b
        cr = np.where(r <= a, -self.delta,
                      np.where(r <= b, -self.beta + 2 * self.gamma / r ** 3, self.h + 2 * self.alpha / r ** 3))
        ct = np.where(r <= a, self.delta,
                      np.where(r <= b, self.beta + self.gamma / r ** 3, -self.h + self.alpha / r ** 3))
        mr = np.where((r > a) & (r <= b), self.mur, 1.0)
        return cr, ct, mr

    def B(self, x, y):
        """(B_rho, B_z) exatos em (rho = x, z = y)."""
        r = np.maximum(np.hypot(x, y), 1e-300)
        cr, ct, mr = self._coeffs(r)
        s, c = x / r, y / r
        k = MU0 * mr
        return k * (cr + ct) * s * c, k * (cr * c ** 2 - ct * s ** 2)

    def A(self, x, y):
        r = np.maximum(np.hypot(x, y), 1e-300)
        cr, _, mr = self._coeffs(r)
        return 0.5 * MU0 * mr * cr * x

    def B_inside(self):
        return -MU0 * self.delta                                 # indução uniforme na cavidade

def design_thickness(finite=False, mur=MUR, a=A_IN, alvo=ALVO):
    """Espessura d = b - a tal que |B_cavidade| = alvo * B_externo."""
    if not finite:
        f = lambda b: shielding_factor(b, mur, a) - alvo
    else:
        f = lambda b: abs(Shield(b, mur, a, finite=True).delta) / H0 - alvo
    b = brentq(f, a * 1.0001, 3 * a, xtol=1e-14)
    return b - a

def generate_mesh(d, refine=1.0, n_layers=4, h_t=0.02, h_far=0.08,
                  theta_prog=1.0, axis_refine=1.0):
    a, b, R = A_IN, A_IN + d, R_EXT
    h_t, h_far, n_layers = h_t / refine, h_far / refine, int(np.ceil(n_layers * refine))
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.add("blindagem")
    geo = gmsh.model.geo
    O = geo.addPoint(0, 0, 0, 1.5 * h_t)
    rad, size = (a, b, R), (h_t, h_t, h_far)
    px = [geo.addPoint(r, 0, 0, s) for r, s in zip(rad, size)]
    py = [geo.addPoint(0, r, 0, s) for r, s in zip(rad, size)]
    arc = [geo.addCircleArc(px[i], O, py[i]) for i in range(3)]
    lx = [geo.addLine(O, px[0]), geo.addLine(px[0], px[1]), geo.addLine(px[1], px[2])]
    ly = [geo.addLine(O, py[0]), geo.addLine(py[0], py[1]), geo.addLine(py[1], py[2])]
    surf = lambda c: geo.addPlaneSurface([geo.addCurveLoop(c)])
    s_cav = surf([lx[0], arc[0], -ly[0]])
    s_sh = surf([lx[1], arc[1], -ly[1], -arc[0]])
    s_out = surf([lx[2], arc[2], -ly[2], -arc[1]])

    nth = max(8, int(np.ceil(0.5 * np.pi * b / h_t)))
    for l in (lx[1], ly[1]):                                    # camadas uniformes na espessura
        geo.mesh.setTransfiniteCurve(l, n_layers + 1)
    for l in (arc[0], arc[1]):                                  # do equador (px) ao eixo (py)
        geo.mesh.setTransfiniteCurve(l, nth + 1, "Progression", theta_prog)
    geo.mesh.setTransfiniteSurface(s_sh, "Left", [px[0], px[1], py[1], py[0]])
    geo.synchronize()

    if axis_refine > 1.0:                                       # refino do ar junto ao eixo
        F = gmsh.model.mesh.field
        fd = F.add("Distance")
        F.setNumbers(fd, "CurvesList", [ly[0], ly[2]])
        F.setNumber(fd, "Sampling", 200)
        ft = F.add("Threshold")
        F.setNumber(ft, "InField", fd)
        F.setNumber(ft, "SizeMin", h_t / axis_refine)
        F.setNumber(ft, "SizeMax", h_far)
        F.setNumber(ft, "DistMin", 0.02)
        F.setNumber(ft, "DistMax", 0.3)
        F.setAsBackgroundMesh(ft)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)

    tags, xyz, _ = gmsh.model.mesh.getNodes()
    table = np.zeros((int(np.max(tags)) + 1, 3))
    table[np.asarray(tags, dtype=np.int64)] = np.asarray(xyz).reshape(-1, 3)
    conn_tags, region = [], []
    for reg, s in ((0, s_cav), (1, s_sh), (2, s_out)):
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
    """Espaços P1 (vértices, lados retos) e P2 (isoparamétrico, nós curvos do Gmsh)."""
    verts = np.unique(conn6[:, :3])
    remap = -np.ones(len(coords), dtype=np.int64)
    remap[verts] = np.arange(len(verts))
    return {1: (coords[verts], remap[conn6[:, :3]]), 2: (coords, conn6)}

def dirichlet_mask(coords, tol=1e-9):
    """Eixo (rho = 0) e esfera r = 5a. O plano z = 0 fica livre (Neumann natural)."""
    r = np.hypot(coords[:, 0], coords[:, 1])
    return (coords[:, 0] < tol) | (np.abs(r - R_EXT) < tol)

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

def assemble(coords, conn, p, region, mur):
    mu_r = np.array([1.0, mur, 1.0])
    xq, wq = quad_tri(p + 3)
    phi, dphi = basis(p, xq)
    xp, det, invJ = geometry(coords, conn, xq, p)
    rho = xp[..., 0]
    w = wq[None, :] * np.abs(det)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
    nu = (1.0 / mu_r[region])[:, None]                          # (1/mu_r); fator 1/mu0 omitido
    dr = G[..., 0]
    Ke = (np.einsum("eq,eqbi,eqci->ebc", w * nu * rho, G, G)
          + np.einsum("eq,qc,eqb->ebc", w * nu, phi, dr)
          + np.einsum("eq,qb,eqc->ebc", w * nu, phi, dr)
          + np.einsum("eq,qb,qc->ebc", w * nu / rho, phi, phi))
    nb = conn.shape[1]
    n = len(coords)
    rows = np.repeat(conn, nb, axis=1).ravel()
    cols = np.tile(conn, (1, nb)).ravel()
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()

def solve(coords, conn, p, region, mur):
    K = assemble(coords, conn, p, region, mur)
    dmask = dirichlet_mask(coords)
    free, dirn = np.where(~dmask)[0], np.where(dmask)[0]
    u = np.zeros(len(coords))
    u[dirn] = g_dirichlet(coords[dirn, 0], coords[dirn, 1])
    # Sistema reduzido (equações dos nós de Dirichlet eliminadas): Kff uf = - Kfd ud
    rhs = -K[free][:, dirn] @ u[dirn]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return u

def fields_at_quad(coords, conn, u, p, nq=6):
    xq, wq = quad_tri(nq)
    phi, dphi = basis(p, xq)
    xp, det, invJ = geometry(coords, conn, xq, p)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
    ue = u[conn]
    Ah = np.einsum("qb,eb->eq", phi, ue)
    gh = np.einsum("eqbi,eb->eqi", G, ue)
    Brho, Bz = -gh[..., 1], gh[..., 0] + Ah / xp[..., 0]
    return xp, wq[None, :] * np.abs(det), Ah, Brho, Bz

ZONAS = ((0, "cavidade (blindada)"), (1, "casca magnética"), (2, "ar externo"))

def zone_errors(coords, conn, u, p, region, sh, nq=6, tol=1e-9):
    xp, w, Ah, Brho, Bz = fields_at_quad(coords, conn, u, p, nq)
    Ex, Ey = sh.B(xp[..., 0], xp[..., 1])
    dB = np.hypot(Brho - Ex, Bz - Ey)
    Bm = np.hypot(Ex, Ey)
    wv = w * xp[..., 0]
    axis_el = (coords[conn[:, :3], 0] < tol).any(axis=1)
    out = {}
    for reg, nome in ZONAS:
        for tag, m in (("todos", region == reg), ("eixo", (region == reg) & axis_el)):
            if not m.any():
                continue
            mx = np.max(dB[m] / Bm[m]) if tag == "eixo" else np.max(dB[m]) / np.max(Bm[m])
            out[(reg, tag)] = (100 * np.sqrt(np.sum(wv[m] * dB[m] ** 2) / np.sum(wv[m] * Bm[m] ** 2)),
                               100 * mx)
    # fator de blindagem do FEM: B_z médio na cavidade / B0
    cav = region == 0
    Sh = np.sum((w * Bz)[cav]) / np.sum(w[cav]) / B0 if cav.any() else np.nan
    return out, Sh

def run_case(d, p, refine=1.0, mesh=None, mur=MUR, **mesh_kw):
    t0 = time.time()
    coords6, conn6, region = mesh if mesh is not None else generate_mesh(d, refine, **mesh_kw)
    coords, conn = build_spaces(coords6, conn6)[p]
    u = solve(coords, conn, p, region, mur)
    b = A_IN + d
    sh_fin, sh_inf = Shield(b, mur, finite=True), Shield(b, mur, finite=False)
    err_fin, Sh = zone_errors(coords, conn, u, p, region, sh_fin)
    err_inf, _ = zone_errors(coords, conn, u, p, region, sh_inf)
    return dict(p=p, d=d, nel=len(conn), ndof=len(coords), time=time.time() - t0, coords=coords,
                conn=conn, region=region, u=u, S=Sh, err_fin=err_fin, err_inf=err_inf,
                sh_fin=sh_fin, sh_inf=sh_inf, mur=mur)

def plot_fields(m, fname="blindagem_campos.png"):
    coords, conn, u, region, p = m["coords"], m["conn"], m["u"], m["region"], m["p"]
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], conn[:, :3])
    xp, w, Ah, Br, Bz = fields_at_quad(coords, conn, u, p, nq=1)
    Bm = np.hypot(Br, Bz)[:, 0]
    sh = m["sh_fin"]
    b = sh.b
    fig, ax = plt.subplots(2, 2, figsize=(13, 10.5))
    ax[0, 0].tricontour(tri, u * coords[:, 0], 40, colors="k", linewidths=.5)
    ax[0, 0].set_title(r"Linhas de fluxo ($\rho A_\varphi$ = const)")
    pc = ax[0, 1].tripcolor(tri, facecolors=np.maximum(Bm, 1e-6), norm=LogNorm(1e-5, max(Bm.max(), 0.2)))
    plt.colorbar(pc, ax=ax[0, 1], label="|B| (T)")
    ax[0, 1].set_title("|B| em todo o domínio (escala log)")
    pc2 = ax[1, 0].tripcolor(tri, facecolors=np.maximum(Bm, 1e-6), norm=LogNorm(1e-5, max(Bm.max(), 0.2)))
    plt.colorbar(pc2, ax=ax[1, 0], label="|B| (T)")
    ax[1, 0].set_xlim(0, 1.6 * b); ax[1, 0].set_ylim(0, 1.6 * b)
    ax[1, 0].set_title("|B| na região da blindagem")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_aspect("equal"); a.set_xlabel(r"$\rho$ (m)"); a.set_ylabel("z (m)")
    ax[0, 0].set_xlim(0, 0.3); ax[0, 0].set_ylim(0, 0.3)

    # B_z nos elementos junto ao eixo (baricentros) x analítico
    axis_el = (coords[conn[:, :3], 0] < 1e-9).any(axis=1)
    xb, yb = xp[axis_el, 0, 0], xp[axis_el, 0, 1]
    o = np.argsort(yb)
    Ex, Ey = sh.B(xb[o], yb[o])
    ax[1, 1].semilogy(yb[o], np.abs(Bz[axis_el, 0][o]), "b.", ms=4, label=f"FEM P{p} (elementos do eixo)")
    ax[1, 1].semilogy(yb[o], np.abs(Ey), "r-", lw=1, label="analítico")
    ax[1, 1].set_xlabel("z (m)"); ax[1, 1].set_ylabel(r"$|B_z|$ junto ao eixo (T)")
    for z0 in (A_IN, b):
        ax[1, 1].axvline(z0, color="gray", ls="--", lw=.8)
    twin = ax[1, 1].twinx()
    twin.plot(yb[o], 100 * np.abs(Bz[axis_el, 0][o] - Ey) / np.abs(Ey), "g-", lw=.7, alpha=.7)
    twin.set_ylabel("erro percentual (verde)")
    twin.set_yscale("log")
    ax[1, 1].legend(loc="center right"); ax[1, 1].set_title("Indução junto ao eixo de rotação")
    plt.tight_layout()

def plot_results_fem_blindagem():
    # --- espessura da blindagem
    d_inf = design_thickness(finite=False)
    d_fin = design_thickness(finite=True)
    sh = Shield(A_IN + d_inf)
    mesh_ar = generate_mesh(d_inf)
    for p in (1, 2):
        coords, conn = build_spaces(*mesh_ar[:2])[p]
        u = solve(coords, conn, p, mesh_ar[2], 1.0)
        xp, w, Ah, Br, Bz = fields_at_quad(coords, conn, u, p)
    base = {p: run_case(d_inf, p, refine=1.0, mesh=mesh_ar) for p in (1, 2)}
    plot_fields(base[2])
    for ref in (2, 4):
        mesh = generate_mesh(d_inf, ref)
    plt.show()
    return