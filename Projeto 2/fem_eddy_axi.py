"""
Correntes induzidas (quase estático, regime senoidal) com SIMETRIA AXIAL
Elementos triangulares de Lagrange P1 e P2 (P2 isoparamétrico, lados curvos do Gmsh).

Formulação: potencial vetor A = A_phi(rho,z) phi, fasor com dependência exp(j w t):

    rot( (1/mu) rot A ) = - j w sigma A

Forma fraca no plano meridiano (x = rho, y = z), integrando em 2*pi*rho drho dz:

  int (1/mu) [ rho grad(A).grad(v) + d(A v)/drho + A v / rho ] dS
        + j w int sigma rho A v dS  = 0

 - Dirichlet imposto por ELIMINAÇÃO das equações dos nós de Dirichlet:
       A = 0 no eixo (rho = 0)  e  A = B0 rho / 2 (campo uniforme B0 z) em r = 0,3 m.
 - Simetria em relação ao plano z = 0 (campo uniforme em z => A_phi par em z):
   modela-se só rho >= 0, z >= 0, com dA/dz = 0 em z = 0 (condição natural).
   Grandezas globais (perdas) são multiplicadas por 2.
 - Malhas do Gmsh (import gmsh). A casca condutora é malhada com camadas
   estruturadas, de espessura crescente a partir da superfície externa, adaptadas à
   profundidade de penetração delta = sqrt(2/(w mu sigma)).
 - Condição de contorno de baixa profundidade de penetração (impedância de superfície,
   Leontovich):  E_t = Zs (n x H),  Zs = (1+j)/(sigma delta). O condutor é removido da
   malha e entra um termo de contorno  j w mu0 / Zs * int rho A v dl  na superfície r = b.
 - Solução analítica (esfera oca em campo uniforme, harmônicos esféricos de ordem 1)
   para validação, para o domínio infinito e para o domínio finito de raio 0,3 m
   com a mesma condição de contorno do FEM.

Dependências: numpy, scipy, matplotlib, gmsh   (pip install gmsh)
Unidades: SI. B0 é a amplitude (valor de pico) da indução externa.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
import gmsh

# ----------------------------------------------------------------------------
# Dados do problema
# ----------------------------------------------------------------------------
MU0 = 4e-7 * np.pi
SIGMA = 5e8                 # S/m (casca)
FREQ = 50.0                 # Hz
OMEGA = 2 * np.pi * FREQ
B0 = 1.0                    # T (amplitude)
RA, RB, RR = 0.05, 0.055, 0.30      # raio interno, externo da casca, fronteira externa (m)


def skin_depth(mur):
    return np.sqrt(2.0 / (OMEGA * MU0 * mur * SIGMA))


def g_dirichlet(x, y):
    """A_phi prescrito (eixo e fronteira externa): campo uniforme B0 z => A = B0 rho/2."""
    return 0.5 * B0 * x


# ----------------------------------------------------------------------------
# Solução analítica: esfera oca condutora em campo uniforme senoidal
#   A_phi = f(r) sin(theta);  f = C r (r<a);  P psi+ + Q psi- (casca);  alpha r + D/r^2 (r>b)
# ----------------------------------------------------------------------------
class Analytic:
    def __init__(self, mur, finite=True):
        a, b, R = RA, RB, RR
        nu0, nuS = 1.0 / MU0, 1.0 / (MU0 * mur)
        k = (1 + 1j) / skin_depth(mur)
        self.k, self.mur = k, mur
        self.psip = lambda r: np.exp(k * (r - b)) * (k * r - 1) / (k * r) ** 2
        self.psim = lambda r: np.exp(-k * (r - a)) * (k * r + 1) / (k * r) ** 2
        self.rpsip = lambda r: np.exp(k * (r - b)) * (1 - 1 / (k * r) + 1 / (k * r) ** 2)       # (r psi+)'
        self.rpsim = lambda r: -np.exp(-k * (r - a)) * (1 + 1 / (k * r) + 1 / (k * r) ** 2)     # (r psi-)'
        M = np.zeros((5, 5), dtype=complex)
        rhs = np.zeros(5, dtype=complex)
        # continuidade de f e de nu (r f)' em r = a e r = b   (linhas 2 e 4 divididas por nu0)
        M[0] = [a, -self.psip(a), -self.psim(a), 0, 0]
        M[1] = [2 * a, -(nuS / nu0) * self.rpsip(a), -(nuS / nu0) * self.rpsim(a), 0, 0]
        M[2] = [0, self.psip(b), self.psim(b), -b, -1 / b ** 2]
        M[3] = [0, (nuS / nu0) * self.rpsip(b), (nuS / nu0) * self.rpsim(b), -2 * b, 1 / b ** 2]
        if finite:                                  # f(R) = B0 R / 2
            M[4] = [0, 0, 0, R, 1 / R ** 2]
            rhs[4] = 0.5 * B0 * R
        else:                                       # campo uniforme no infinito
            M[4] = [0, 0, 0, 1, 0]
            rhs[4] = 0.5 * B0
        self.C, self.P, self.Q, self.alpha, self.D = np.linalg.solve(M, rhs)

    def f_rfp(self, r):
        """f(r) e (r f)'(r)."""
        r = np.atleast_1d(np.maximum(np.asarray(r, dtype=float), 1e-12))
        f = np.zeros(r.shape, dtype=complex)
        g = np.zeros(r.shape, dtype=complex)
        m1, m2 = r <= RA, (r > RA) & (r <= RB)
        m3 = r > RB
        f[m1], g[m1] = self.C * r[m1], 2 * self.C * r[m1]
        f[m2] = self.P * self.psip(r[m2]) + self.Q * self.psim(r[m2])
        g[m2] = self.P * self.rpsip(r[m2]) + self.Q * self.rpsim(r[m2])
        f[m3] = self.alpha * r[m3] + self.D / r[m3] ** 2
        g[m3] = 2 * self.alpha * r[m3] - self.D / r[m3] ** 2
        return f, g

    def A(self, x, y):
        r = np.hypot(x, y)
        f, _ = self.f_rfp(r)
        return (f * x / np.maximum(r, 1e-300)).reshape(np.shape(x))

    def gradA(self, x, y):
        r = np.maximum(np.hypot(x, y), 1e-300)
        f, g = self.f_rfp(r)
        f, g = f.reshape(np.shape(x)), g.reshape(np.shape(x))
        fp = (g - f) / r
        gp = (fp * r - f) / r ** 2                  # d(f/r)/dr
        return f / r + x * gp * x / r, x * gp * y / r

    def B_inside(self):
        """Indução (fasor) no interior da cavidade r < a (uniforme): B_z = 2 C."""
        return 2 * self.C

    def losses(self):
        """Perdas Joule médias (W) na casca inteira: (4 pi/3) sigma w^2 int |f|^2 r^2 dr."""
        d = skin_depth(self.mur)
        pts = [RB - s * d for s in (10, 3, 1) if RB - s * d > RA]
        fun = lambda r: abs(self.f_rfp(r)[0][0]) ** 2 * r ** 2
        val, _ = quad(fun, RA, RB, points=pts or None, limit=400, epsabs=0, epsrel=1e-10)
        return 4 * np.pi / 3 * SIGMA * OMEGA ** 2 * val


# ----------------------------------------------------------------------------
# Geração de malha com Gmsh
# ----------------------------------------------------------------------------
def layer_division(t, h1, q=1.25, nmin=4):
    """
    Divisão radial da casca (espessura t) com primeira camada h1 junto à superfície externa.
    Retorna N (nº de camadas) e a razão r (>1: camadas crescem para o interior).
    """
    if t / h1 <= nmin:
        return nmin, 1.0
    N = max(nmin, int(np.ceil(np.log(1 + t * (q - 1) / h1) / np.log(q))))
    S = lambda r: h1 * N if abs(r - 1) < 1e-12 else h1 * (r ** N - 1) / (r - 1)
    lo, hi = 0.5, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if S(mid) < t else (lo, mid)
    return N, 0.5 * (lo + hi)


def _read_mesh(surfs):
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    table = np.zeros((int(np.max(tags)) + 1, 3))
    table[np.asarray(tags, dtype=np.int64)] = np.asarray(xyz).reshape(-1, 3)
    conn_tags, region = [], []
    for reg, s in surfs:
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


def generate_mesh_conductor(mur, refine=1.0, h_t=3e-3, n_delta=4, growth=1.25,
                            lc_far=2.5e-2, lc_in=5e-3):
    """
    Quarto do meridiano (rho>=0, z>=0): cavidade (região 0), casca (região 1), ar externo (região 2).
    A casca usa malha transfinita (camadas) com refino geométrico junto à superfície r = b,
    primeira camada = delta/n_delta. refine>1 refina tudo.
    """
    h_t, lc_far, lc_in, n_delta = h_t / refine, lc_far / refine, lc_in / refine, n_delta * refine
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.add("casca")
    geo = gmsh.model.geo
    O = geo.addPoint(0, 0, 0, lc_in)
    rad, size = (RA, RB, RR), (h_t, h_t, lc_far)
    px = [geo.addPoint(r, 0, 0, s) for r, s in zip(rad, size)]
    py = [geo.addPoint(0, r, 0, s) for r, s in zip(rad, size)]
    arc = [geo.addCircleArc(px[i], O, py[i]) for i in range(3)]
    lx = [geo.addLine(O, px[0]), geo.addLine(px[0], px[1]), geo.addLine(px[1], px[2])]
    ly = [geo.addLine(O, py[0]), geo.addLine(py[0], py[1]), geo.addLine(py[1], py[2])]
    surf = lambda c: geo.addPlaneSurface([geo.addCurveLoop(c)])
    s_in = surf([lx[0], arc[0], -ly[0]])
    s_sh = surf([lx[1], arc[1], -ly[1], -arc[0]])
    s_out = surf([lx[2], arc[2], -ly[2], -arc[1]])

    N, r = layer_division(RB - RA, skin_depth(mur) / n_delta, growth)
    nth = max(8, int(np.ceil(0.5 * np.pi * RB / h_t)))
    for l in (lx[1], ly[1]):                       # de r=a para r=b: elementos diminuem (1/r)
        geo.mesh.setTransfiniteCurve(l, N + 1, "Progression", 1.0 / r)
    for l in (arc[0], arc[1]):
        geo.mesh.setTransfiniteCurve(l, nth + 1)
    geo.mesh.setTransfiniteSurface(s_sh, "Left", [px[0], px[1], py[1], py[0]])
    geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    return _read_mesh([(0, s_in), (1, s_sh), (2, s_out)])


def generate_mesh_sibc(refine=1.0, h_t=3e-3, lc_far=2.5e-2):
    """Só o ar externo (b < r < R): o condutor é substituído pela condição de impedância em r = b."""
    h_t, lc_far = h_t / refine, lc_far / refine
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.add("sibc")
    geo = gmsh.model.geo
    O = geo.addPoint(0, 0, 0, h_t)
    px = [geo.addPoint(r, 0, 0, s) for r, s in ((RB, h_t), (RR, lc_far))]
    py = [geo.addPoint(0, r, 0, s) for r, s in ((RB, h_t), (RR, lc_far))]
    arc = [geo.addCircleArc(px[i], O, py[i]) for i in range(2)]
    lx, ly = geo.addLine(px[0], px[1]), geo.addLine(py[0], py[1])
    s = geo.addPlaneSurface([geo.addCurveLoop([lx, arc[1], -ly, -arc[0]])])
    geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    return _read_mesh([(2, s)])


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
    """Eixo (rho = 0) e fronteira externa (r = R). O plano z = 0 fica livre (Neumann natural)."""
    r = np.hypot(coords[:, 0], coords[:, 1])
    return (coords[:, 0] < tol) | (np.abs(r - RR) < tol)


# ----------------------------------------------------------------------------
# Elemento de referência (triângulo unitário): base de Lagrange P_p, p = 1, 2
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Montagem e solução complexa com eliminação dos nós de Dirichlet
# ----------------------------------------------------------------------------
def material(mur):
    return np.array([1.0, mur, 1.0]), np.array([0.0, SIGMA, 0.0])      # por região 0, 1, 2


def assemble(coords, conn, p, region, mur):
    """
    Sistema multiplicado por mu0 (melhor condicionamento):
      K = int (1/mu_r)[rho grad.grad + d(phi_i phi_j)/drho + phi_i phi_j / rho],  M = int mu0 sigma rho phi_i phi_j
    """
    MUR, SIG = material(mur)
    xq, wq = quad_tri(p + 3)
    phi, dphi = basis(p, xq)
    xp, det, invJ = geometry(coords, conn, xq, p)
    rho = xp[..., 0]
    w = wq[None, :] * np.abs(det)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
    nu = (1.0 / MUR[region])[:, None]
    sg = (MU0 * SIG[region])[:, None]
    dr = G[..., 0]                                                  # d phi / d rho
    Ke = (np.einsum("eq,eqbi,eqci->ebc", w * nu * rho, G, G)
          + np.einsum("eq,qc,eqb->ebc", w * nu, phi, dr)
          + np.einsum("eq,qb,eqc->ebc", w * nu, phi, dr)
          + np.einsum("eq,qb,qc->ebc", w * nu / rho, phi, phi)
          + 1j * OMEGA * np.einsum("eq,qb,qc->ebc", w * sg * rho, phi, phi))
    nb = conn.shape[1]
    n = len(coords)
    rows = np.repeat(conn, nb, axis=1).ravel()
    cols = np.tile(conn, (1, nb)).ravel()
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def circle_edges(coords, conn, p, R, tol=1e-9):
    """Lados de elementos sobre o círculo r = R: nós [vi, vj] (P1) ou [vi, vj, meio] (P2)."""
    r = np.hypot(coords[:, 0], coords[:, 1])
    out = []
    for le, (i, j) in enumerate(LOCAL_EDGES):
        vi, vj = conn[:, i], conn[:, j]
        on = (np.abs(r[vi] - R) < tol) & (np.abs(r[vj] - R) < tol)
        nodes = np.stack([vi, vj], 1) if p == 1 else np.stack([vi, vj, conn[:, 3 + le]], 1)
        out.append(nodes[on])
    return np.vstack(out)


def edge_rule(coords, edges, p, n=4):
    sg, wg = np.polynomial.legendre.leggauss(n)
    s, w = 0.5 * (sg + 1), 0.5 * wg
    if p == 1:
        N = np.stack([1 - s, s], 1)
        dN = np.stack([-np.ones_like(s), np.ones_like(s)], 1)
    else:
        N = np.stack([(1 - s) * (1 - 2 * s), s * (2 * s - 1), 4 * s * (1 - s)], 1)
        dN = np.stack([-3 + 4 * s, 4 * s - 1, 4 - 8 * s], 1)
    X = coords[edges]
    xq = np.einsum("qb,ebj->eqj", N, X)
    dl = np.linalg.norm(np.einsum("qb,ebj->eqj", dN, X), axis=2)
    return N, xq, w[None, :] * dl


def sibc_impedance(mur):
    """Zs = (1+j)/(sigma delta) = sqrt(j w mu / sigma)."""
    return (1 + 1j) / (SIGMA * skin_depth(mur))


def solve(coords, conn, p, region, mur, sibc=False):
    """sibc=True: malha só com o ar externo e condição de impedância de superfície em r = b."""
    K = assemble(coords, conn, p, region, mur)
    if sibc:
        edges = circle_edges(coords, conn, p, RB)
        N, xq, wdl = edge_rule(coords, edges, p)
        Se = np.einsum("eq,eq,qb,qc->ebc", wdl, xq[..., 0], N, N)
        nb = edges.shape[1]
        n = len(coords)
        S = sp.coo_matrix((Se.ravel(), (np.repeat(edges, nb, axis=1).ravel(),
                                        np.tile(edges, (1, nb)).ravel())), shape=(n, n)).tocsr()
        K = K + (1j * OMEGA * MU0 / sibc_impedance(mur)) * S        # j w mu0 / Zs
    dmask = dirichlet_mask(coords)
    free, dirn = np.where(~dmask)[0], np.where(dmask)[0]
    u = np.zeros(len(coords), dtype=complex)
    u[dirn] = g_dirichlet(coords[dirn, 0], coords[dirn, 1])
    # Sistema reduzido (equações dos nós de Dirichlet eliminadas): Kff uf = - Kfd ud
    rhs = -K[free][:, dirn] @ u[dirn]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return u


# ----------------------------------------------------------------------------
# Pós-processamento
# ----------------------------------------------------------------------------
def fields_at_quad(coords, conn, u, p, nq=6):
    xq, wq = quad_tri(nq)
    phi, dphi = basis(p, xq)
    xp, det, invJ = geometry(coords, conn, xq, p)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
    ue = u[conn]
    return (xp, wq[None, :] * np.abs(det), np.einsum("qb,eb->eq", phi, ue),
            np.einsum("eqbi,eb->eqi", G, ue))


def metrics(coords, conn, u, p, region, mur, ana, nq=6):
    """B na cavidade, perdas na casca, erro L2 (peso rho) em relação à solução analítica."""
    xp, w, Ah, gh = fields_at_quad(coords, conn, u, p, nq)
    rho = xp[..., 0]
    Aex = ana.A(xp[..., 0], xp[..., 1])
    out = {"errL2": np.sqrt(np.sum(w * rho * abs(Ah - Aex) ** 2) / np.sum(w * rho * abs(Aex) ** 2))}
    sh = region == 1
    if sh.any():
        out["P"] = 0.5 * SIGMA * OMEGA ** 2 * 2 * np.pi * np.sum(w[sh] * rho[sh] * abs(Ah[sh]) ** 2) * 2
        out["errL2_shell"] = np.sqrt(np.sum((w * rho * abs(Ah - Aex) ** 2)[sh])
                                     / np.sum((w * rho * abs(Aex) ** 2)[sh]))
    cav = region == 0
    if cav.any():
        Bz = gh[..., 0] + Ah / rho
        out["B_in"] = np.sum((w * Bz)[cav]) / np.sum(w[cav])
    return out


def sibc_losses(coords, conn, u, p, mur):
    """P = (1/2) w^2 Re(1/Zs) int |A|^2 dS   (hemisfério x 2 pela simetria)."""
    edges = circle_edges(coords, conn, p, RB)
    N, xq, wdl = edge_rule(coords, edges, p)
    Aq = np.einsum("qb,eb->eq", N, u[edges])
    Zs = sibc_impedance(mur)
    return 0.5 * OMEGA ** 2 * np.real(1 / Zs) * 2 * np.pi * np.sum(wdl * xq[..., 0] * abs(Aq) ** 2) * 2


def equator_profile(coords, u, tol=1e-9):
    m = np.abs(coords[:, 1]) < tol
    idx = np.where(m)[0][np.argsort(coords[m, 0])]
    return coords[idx, 0], u[idx]


def surface_profile(coords, u, R=RB, tol=1e-9):
    r = np.hypot(coords[:, 0], coords[:, 1])
    idx = np.where(np.abs(r - R) < tol)[0]
    th = np.arctan2(coords[idx, 0], coords[idx, 1])          # ângulo polar a partir do eixo z
    o = np.argsort(th)
    return th[o], u[idx][o]


# ----------------------------------------------------------------------------
# Execução de casos
# ----------------------------------------------------------------------------
def run_case(mur, p, refine=1.0, sibc=False, mesh=None):
    t0 = time.time()
    coords6, conn6, region = mesh if mesh is not None else (
        generate_mesh_sibc(refine) if sibc else generate_mesh_conductor(mur, refine))
    coords, conn = build_spaces(coords6, conn6)[p]
    u = solve(coords, conn, p, region, mur, sibc=sibc)
    ana = Analytic(mur, finite=True)
    m = metrics(coords, conn, u, p, region, mur, ana)
    if sibc:
        m["P"] = sibc_losses(coords, conn, u, p, mur)
    m.update(dict(nel=len(conn), ndof=len(coords), time=time.time() - t0, coords=coords,
                  conn=conn, u=u, region=region, p=p, mur=mur, sibc=sibc))
    return m


def print_case(m):
    mur, ana_f, ana_i = m["mur"], Analytic(m["mur"], True), Analytic(m["mur"], False)
    tag = "SIBC" if m["sibc"] else "condutor"
    s = (f"mu_r={mur:6g} {tag:8s} P{m['p']} nel={m['nel']:6d} ndof={m['ndof']:7d} t={m['time']:5.1f}s "
         f"erroL2(A)={m['errL2']:.2e}")
    if "B_in" in m:
        s += (f" | |B_in|/B0={abs(m['B_in'])/B0:.4e} (analit. dom.finito {abs(ana_f.B_inside())/B0:.4e}, "
              f"infinito {abs(ana_i.B_inside())/B0:.4e})")
    s += f" | P={m['P']:.5e} W (analit. {ana_f.losses():.5e})"
    print(s)


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def plot_fields(m, fname="campos_casca.png"):
    coords, conn, u, region, p, mur = (m[k] for k in ("coords", "conn", "u", "region", "p", "mur"))
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], conn[:, :3])
    xp, w, Ah, gh = fields_at_quad(coords, conn, u, p, nq=1)
    Bmag = np.sqrt(abs(gh[..., 1]) ** 2 + abs(gh[..., 0] + Ah / xp[..., 0]) ** 2)[:, 0]   # B = (-dA/dz, dA/drho + A/rho)
    ana = Analytic(mur, True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    z = np.real(u * coords[:, 0])                                   # fluxo rho*A (instante t=0)
    c = ax[0, 0].tricontourf(tri, np.abs(u * coords[:, 0]), 40)
    ax[0, 0].tricontour(tri, z, 25, colors="k", linewidths=.5)
    plt.colorbar(c, ax=ax[0, 0], label=r"$|\rho A_\varphi|$ (Wb/2π)")
    ax[0, 0].set_title(r"Linhas de fluxo ($\mathrm{Re}(\rho A_\varphi)$) e $|\rho A_\varphi|$")

    pc = ax[0, 1].tripcolor(tri, facecolors=np.maximum(Bmag, 1e-4), norm=LogNorm(1e-3, max(Bmag.max(), 1.0)))
    plt.colorbar(pc, ax=ax[0, 1], label="|B| (T)")
    ax[0, 1].set_title("Módulo da indução magnética |B|")

    shell = region == 1
    if shell.any():
        trs = mtri.Triangulation(coords[:, 0], coords[:, 1], conn[shell, :3])
        Jm = SIGMA * OMEGA * np.abs(u)
        c = ax[1, 0].tricontourf(trs, Jm, 40)
        plt.colorbar(c, ax=ax[1, 0], label=r"$|J_\varphi|$ (A/m²)")
        ax[1, 0].set_title("Densidade de corrente induzida na casca")
        ax[1, 0].set_xlim(0, 1.05 * RB); ax[1, 0].set_ylim(0, 1.05 * RB)
    for a in ax[0]:
        a.set_aspect("equal"); a.set_xlabel(r"$\rho$ (m)"); a.set_ylabel("z (m)")
    ax[1, 0].set_aspect("equal"); ax[1, 0].set_xlabel(r"$\rho$ (m)"); ax[1, 0].set_ylabel("z (m)")

    x, ue = equator_profile(coords, u)
    ins = (x >= RA) & (x <= RB)
    xs = np.linspace(RA, RB, 600)
    ax[1, 1].plot((RB - x[ins]) * 1e3, SIGMA * OMEGA * np.abs(ue[ins]), "o", ms=3, label=f"FEM P{p}")
    ax[1, 1].plot((RB - xs) * 1e3, SIGMA * OMEGA * np.abs(ana.A(xs, 0 * xs)), "r-", lw=1, label="analítico")
    ax[1, 1].set_xlabel("profundidade a partir de r = b (mm)")
    ax[1, 1].set_ylabel(r"$|J_\varphi|$ no equador (A/m²)")
    ax[1, 1].set_title(f"Decaimento da corrente (δ = {skin_depth(mur)*1e3:.2f} mm)")
    ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)
    plt.tight_layout(); #plt.savefig(fname, dpi=140)
    #print("Figura salva em", fname)


def plot_sibc_comparison(m_full, m_sibc, fname="comparacao_sibc.png"):
    ana = Analytic(m_full["mur"], True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    xf, uf = equator_profile(m_full["coords"], m_full["u"])
    xs, us = equator_profile(m_sibc["coords"], m_sibc["u"])
    xx = np.linspace(RB, RR, 400)
    ax[0].plot(xx * 1e2, np.abs(ana.A(xx, 0 * xx)) * 1e3, "r-", label="analítico")
    ax[0].plot(xf[xf >= RB] * 1e2, np.abs(uf[xf >= RB]) * 1e3, "b.", ms=4, label="FEM condutor")
    ax[0].plot(xs * 1e2, np.abs(us) * 1e3, "g+", ms=5, label="FEM SIBC")
    ax[0].set_xlabel(r"$\rho$ no equador (cm)"); ax[0].set_ylabel(r"$|A_\varphi|$ (mWb/m)"); ax[0].legend()
    ax[0].set_title("Potencial vetor no ar (equador)"); ax[0].grid(alpha=.3)

    thf, af = surface_profile(m_full["coords"], m_full["u"])
    ths, as_ = surface_profile(m_sibc["coords"], m_sibc["u"])
    tt = np.linspace(0, np.pi / 2, 200)
    ax[1].plot(np.degrees(tt), np.abs(ana.A(RB * np.sin(tt), RB * np.cos(tt))) * 1e3, "r-", label="analítico")
    ax[1].plot(np.degrees(thf), np.abs(af) * 1e3, "b.", ms=5, label="FEM condutor")
    ax[1].plot(np.degrees(ths), np.abs(as_) * 1e3, "g+", ms=6, label="FEM SIBC")
    ax[1].set_xlabel(r"ângulo polar $\theta$ (graus)"); ax[1].set_ylabel(r"$|A_\varphi|$ em r = b (mWb/m)")
    ax[1].set_title("Potencial vetor sobre a superfície"); ax[1].legend(); ax[1].grid(alpha=.3)

    ref = np.interp(xs, xf[xf >= RB], np.abs(uf[xf >= RB]))
    ax[2].semilogy(xs * 1e2, np.abs(np.abs(us) - ref) / np.max(ref), "g-")
    ax[2].set_xlabel(r"$\rho$ no equador (cm)"); ax[2].set_ylabel(r"$||A|_{SIBC}-|A|_{cond}| / \max|A|$")
    ax[2].set_title("Diferença SIBC × condutor"); ax[2].grid(alpha=.3, which="both")
    plt.tight_layout(); #plt.savefig(fname, dpi=140)
    #print("Figura salva em", fname)