"""
Transformação campo próximo -> campo distante e largura de espalhamento (RCS 2D) de um
cilindro PEC (raio a = lambda/2) para as polarizações TM (E_z) e TE (H_z).
FEM em formulação de CAMPO ESPALHADO, elementos P1/P2 (P2 isoparamétrico), malhas do Gmsh.
O domínio é truncado com as condições de MENOR ERRO das partes anteriores:
    (1) BGT:  BGT-2 com R = 1,5 lambda          (fem_bgt_cilindro.py)
    (2) PML retangular: n = 2, x0 = 1,2 lambda, L = 0,4 lambda, sigma0 projetado para R = 1e-4
                                                (fem_pml_cilindro.py)
Se as suas tabelas apontarem outra combinação, altere BGT_CFG / PML_CFG (ou rode com --auto, que
reexecuta os dois estudos e escolhe a combinação de menor erro).

Convenção exp(j w t); onda plana incidente  u_inc = exp(-j k0 x),  d = (1,0),  f = 300 MHz.
Para as duas polarizações o campo espalhado u (u = E_z em TM, u = H_z em TE) satisfaz
    lap(u) + k0^2 u = 0
    TM : u = -u_inc no PEC                       (Dirichlet, equações eliminadas)
    TE : du/dn = -du_inc/dn no PEC               (Neumann: termo de carga int (du_inc/dr) v dl)

Transformação campo próximo -> distante (princípio da equivalência, curva C = circunferência de
raio Rc que envolve o cilindro, contida na malha como curva conforme).  Com J = n x H e M = -n x E:
    TM:  E_z^ff = sqrt(jk0/8pi) e^{-jk0 r}/sqrt(r) * oint [ E_z (r.n) + (j/k0) dE_z/dn ] e^{jk0 r.r'} dl'
    TE:  H_z^ff = sqrt(jk0/8pi) e^{-jk0 r}/sqrt(r) * oint [ H_z (r.n) + (j/k0) dH_z/dn ] e^{jk0 r.r'} dl'
(a forma é a mesma pela dualidade: as integrais do enunciado se reduzem a esta, usando as equações
de Maxwell, J_z = (dE_z/dn)/(j w mu0) em TM e M_z = eta0 (dH_z/dn)/(j k0) em TE.)  Usam-se os campos
ESPALHADOS em C (a contribuição do campo incidente, sem fontes no interior de C, é nula no exterior).
Largura de espalhamento:
    sigma_2D(phi) = lim 2 pi r |u^ff/u^i|^2 = (k0/4) |oint [...] e^{jk0 r.r'} dl'|^2     (|u^i| = 1)

Solução analítica (PEC):
    sigma_2D = (4/k0) | sum_n eps_n [J_n(k0 a)/H_n^(2)(k0 a)] cos(n phi) |^2       TM
    sigma_2D = (4/k0) | sum_n eps_n [J_n'(k0 a)/H_n^(2)'(k0 a)] cos(n phi) |^2     TE
phi é medido a partir da direção de propagação da onda incidente (phi = 0: espalhamento frontal;
phi = 180 graus: retroespalhamento).

Dependências: numpy, scipy, matplotlib, gmsh   (pip install gmsh)
"""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import jv, hankel2, jvp, h2vp
import matplotlib.pyplot as plt
import gmsh

F0 = 300e6
C0 = 299792458.0
LAM = C0 / F0
K0 = 2 * np.pi / LAM
RA = LAM / 2
RC = 1.0 * LAM                    # raio da curva C da transformação (RA < RC < domínio físico)
H_MAX = LAM / 20
R_REFL = 1e-4

BGT_CFG = dict(R=1.5 * LAM, order=2)                       # parte (1)
PML_CFG = dict(x0=1.2 * LAM, L=0.4 * LAM, n=2)             # parte (2)

def _ratio(pol, n):
    if pol == "TM":
        return jv(n, K0 * RA) / hankel2(n, K0 * RA)
    return jvp(n, K0 * RA) / h2vp(n, K0 * RA)

def u_scat_exact(pol, x, y, nmax=60, radial_derivative=False):
    r = np.maximum(np.hypot(x, y), 1e-300)
    phi = np.arctan2(y, x)
    out = np.zeros(np.shape(x), dtype=complex)
    for n in range(nmax + 1):
        eps = 1.0 if n == 0 else 2.0
        cn = -eps * (-1j) ** n * _ratio(pol, n)
        if radial_derivative:
            out += cn * K0 * h2vp(n, K0 * r) * np.cos(n * phi)
        else:
            out += cn * hankel2(n, K0 * r) * np.cos(n * phi)
    return out

def u_inc(x, y):
    return np.exp(-1j * K0 * x)

def du_inc_dr(x, y):
    return -1j * K0 * (x / np.maximum(np.hypot(x, y), 1e-300)) * u_inc(x, y)

def rcs_analytic(pol, phi, nmax=60):
    s = np.zeros(np.shape(phi), dtype=complex)
    for n in range(nmax + 1):
        eps = 1.0 if n == 0 else 2.0
        s += eps * _ratio(pol, n) * np.cos(n * phi)
    return 4.0 / K0 * np.abs(s) ** 2

def _read(surfs):
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
    return table[used][:, :2], orient_ccw(table[used][:, :2], np.searchsorted(used, conn_tags)), region

def _init(h):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.add("rcs")
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
    return geo, circle, square, loop

def generate_mesh_bgt(R, h=H_MAX):
    geo, circle, square, loop = _init(h)
    cyl, cc, cout = circle(RA), circle(RC), circle(R)
    s0 = geo.addPlaneSurface([loop(cc), loop(cyl, True)])
    s1 = geo.addPlaneSurface([loop(cout), loop(cc, True)])
    geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    return _read([(0, s0), (1, s1)])

def generate_mesh_pml(x0, L, h=H_MAX):
    geo, circle, square, loop = _init(h)
    cyl, cc, sq_in, sq_out = circle(RA), circle(RC), square(x0), square(x0 + L)
    s0 = geo.addPlaneSurface([loop(cc), loop(cyl, True)])
    s1 = geo.addPlaneSurface([loop(sq_in), loop(cc, True)])
    s2 = geo.addPlaneSurface([loop(sq_out), loop(sq_in, True)])
    geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    return _read([(0, s0), (1, s1), (2, s2)])

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
    return xp, det, invJ, J

def to_sparse(Ke, conn, n):
    nb = conn.shape[1]
    rows = np.repeat(conn, nb, axis=1).ravel()
    cols = np.tile(conn, (1, nb)).ravel()
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()

def assemble_helmholtz(coords, conn, p):
    xq, wq = quad_tri(p + 3)
    phi, dphi = basis(p, xq)
    _, det, invJ, _ = geometry(coords, conn, xq, p)
    w = wq[None, :] * np.abs(det)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
    Ke = np.einsum("eq,eqbi,eqci->ebc", w, G, G) - K0 ** 2 * np.einsum("eq,qb,qc->ebc", w, phi, phi)
    return to_sparse(Ke, conn, len(coords))

def design_sigma0(n, L, R=R_REFL):
    return -np.log(R) * (n + 1) / (2 * L ** (n + 1))

def sigma_profile(t, x0, n, sigma0):
    d = np.maximum(np.abs(t) - x0, 0.0)
    return np.where(np.abs(t) > x0, sigma0 * d ** n, 0.0)

def assemble_pml(coords, conn, p, x0, n, sigma0):
    xq, wq = quad_tri(p + 4)
    phi, dphi = basis(p, xq)
    xp, det, invJ, _ = geometry(coords, conn, xq, p)
    w = wq[None, :] * np.abs(det)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
    sx = 1 - 1j * sigma_profile(xp[..., 0], x0, n, sigma0) / K0
    sy = 1 - 1j * sigma_profile(xp[..., 1], x0, n, sigma0) / K0
    Ae = (np.einsum("eq,eqb,eqc->ebc", w * sy / sx, G[..., 0], G[..., 0])
          + np.einsum("eq,eqb,eqc->ebc", w * sx / sy, G[..., 1], G[..., 1])
          - K0 ** 2 * np.einsum("eq,qb,qc->ebc", w * sx * sy, phi, phi))
    return to_sparse(Ae, conn, len(coords))

def circle_edges(coords, conn, p, R, tol=1e-8):
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
    return N, dN, xq, dl, w

def bgt_matrix(coords, conn, p, R, order):
    edges = circle_edges(coords, conn, p, R)
    N, dN, _, dl, w = edge_rule(coords, edges, p)
    Me = np.einsum("q,eq,qb,qc->ebc", w, dl, N, N)
    Ke = np.einsum("q,eq,qb,qc->ebc", w, 1.0 / dl, dN, dN)
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

def solve(coords, conn, p, pol, abc):
    nn = len(coords)
    r = np.hypot(coords[:, 0], coords[:, 1])
    pec = np.abs(r - RA) < 1e-8
    dmask = np.zeros(nn, dtype=bool)
    if abc["tipo"] == "bgt":
        A = assemble_helmholtz(coords, conn, p) + bgt_matrix(coords, conn, p, abc["R"], abc["order"])
    else:
        A = assemble_pml(coords, conn, p, abc["x0"], abc["n"], abc["sigma0"])
        xo = abc["x0"] + abc["L"]
        dmask |= (np.abs(np.abs(coords[:, 0]) - xo) < 1e-8) | (np.abs(np.abs(coords[:, 1]) - xo) < 1e-8)
    A = A.tocsr().astype(complex)
    u = np.zeros(nn, dtype=complex)
    b = np.zeros(nn, dtype=complex)
    if pol == "TM":                                                   # PEC: u = -u_inc (Dirichlet)
        dmask |= pec
        u[pec] = -u_inc(coords[pec, 0], coords[pec, 1])
    else:                                                             # PEC: du/dn = -du_inc/dn (Neumann)
        edges = circle_edges(coords, conn, p, RA)
        N, _, xq, dl, w = edge_rule(coords, edges, p)
        fe = np.einsum("q,eq,eq,qb->eb", w, dl, du_inc_dr(xq[..., 0], xq[..., 1]), N)
        b = np.bincount(edges.ravel(), weights=fe.real.ravel(), minlength=nn) \
            + 1j * np.bincount(edges.ravel(), weights=fe.imag.ravel(), minlength=nn)
    free, dirn = np.where(~dmask)[0], np.where(dmask)[0]
    rhs = b[free] - A[free][:, dirn] @ u[dirn]
    u[free] = spla.spsolve(A[free][:, free].tocsc(), rhs)
    return u

def rcs_fem(coords, conn, u, p, region, phis, pol, Rc=RC, nq=6, use_total=False):
    sg, wg = np.polynomial.legendre.leggauss(nq)
    s, w = 0.5 * (sg + 1), 0.5 * wg
    r = np.hypot(coords[:, 0], coords[:, 1])
    rhat = np.stack([np.cos(phis), np.sin(phis)], axis=1)
    F = np.zeros(len(phis), dtype=complex)
    for le, (i, j) in enumerate(LOCAL_EDGES):
        on = ((np.abs(r[conn[:, i]] - Rc) < 1e-8) & (np.abs(r[conn[:, j]] - Rc) < 1e-8) & (region == 0))
        e = np.where(on)[0]
        if len(e) == 0:
            continue
        dxi = REF_VERT[j] - REF_VERT[i]
        xi = REF_VERT[i][None, :] + s[:, None] * dxi[None, :]            # pontos de Gauss no lado de referência
        N, dN = basis(p, xi)
        xp, det, invJ, J = geometry(coords, conn[e], xi, p)               # (ne, q, ...)
        t = np.einsum("eqjm,m->eqj", J, dxi)                              # dx/ds
        tl = np.linalg.norm(t, axis=2)
        nrm = np.stack([t[..., 1], -t[..., 0]], axis=2) / tl[..., None]
        nrm = np.where(((nrm * xp).sum(axis=2) < 0)[..., None], -nrm, nrm)   # normal para fora de C
        ue = u[conn[e]]
        uq = np.einsum("qb,eb->eq", N, ue)
        gq = np.einsum("qbk,eqki,eb->eqi", dN, invJ, ue)
        dn = (gq * nrm).sum(axis=2)
        if use_total:
            x, y = xp[..., 0], xp[..., 1]
            uq = uq + u_inc(x, y)
            dn = dn + (du_inc_dr(x, y))                                    # em C, n = r
        rn = np.einsum("pk,eqk->peq", rhat, nrm)                          # r.n
        phase = np.exp(1j * K0 * np.einsum("pk,eqk->peq", rhat, xp))
        integrand = (uq[None] * rn + (1j / K0) * dn[None]) * phase
        F += np.einsum("peq,q,eq->p", integrand, w, tl)
    return K0 / 4 * np.abs(F) ** 2

def check_transformation(pol, phis):
    s, w = np.polynomial.legendre.leggauss(400)
    ang = np.pi * (s + 1)
    wa = np.pi * w
    x, y = RC * np.cos(ang), RC * np.sin(ang)
    u = u_scat_exact(pol, x, y)
    dn = u_scat_exact(pol, x, y, radial_derivative=True)
    rhat = np.stack([np.cos(phis), np.sin(phis)], axis=1)
    rn = rhat @ np.stack([np.cos(ang), np.sin(ang)])
    phase = np.exp(1j * K0 * RC * rn)
    F = ((u[None] * rn + (1j / K0) * dn[None]) * phase) @ (wa * RC)
    return K0 / 4 * np.abs(F) ** 2

def auto_configs():
    import fem_bgt_cilindro as B
    import fem_pml_cilindro as P
    err = B.run_table((2,), verbose=False)
    (Rl, order), _ = min(err[2].items(), key=lambda kv: kv[1])
    groups, res, res_re = P.experiments()                  # (a) n, (b) x0, (c) L (com sigma0 reprojetado)
    n_b = min(groups["a"], key=lambda t: res_re[t][2])[2]
    x0_b = min(groups["b"], key=lambda t: res_re[t][2])[0]
    L_b = min(groups["c"], key=lambda t: res_re[t][2])[1]
    return dict(R=Rl * B.LAM, order=order), dict(x0=x0_b, L=L_b, n=n_b)

def run(p=2, bgt_cfg=BGT_CFG, pml_cfg=PML_CFG, h=H_MAX, mesh_bgt=None, mesh_pml=None):
    phis = np.radians(np.linspace(0, 180, 361))
    cb = (mesh_bgt or generate_mesh_bgt)(bgt_cfg["R"], h)
    cp = (mesh_pml or generate_mesh_pml)(pml_cfg["x0"], pml_cfg["L"], h)
    sp_b, sp_p = build_spaces(cb[0], cb[1]), build_spaces(cp[0], cp[1])
    abc_b = dict(tipo="bgt", R=bgt_cfg["R"], order=bgt_cfg["order"])
    abc_p = dict(tipo="pml", x0=pml_cfg["x0"], L=pml_cfg["L"], n=pml_cfg["n"],
                 sigma0=design_sigma0(pml_cfg["n"], pml_cfg["L"]))
    out = {}
    for pol in ("TM", "TE"):
        ex = rcs_analytic(pol, phis)
        chk = check_transformation(pol, phis)
        coords, conn = sp_b[p]
        s_bgt = rcs_fem(coords, conn, solve(coords, conn, p, pol, abc_b), p, cb[2], phis, pol)
        coords, conn = sp_p[p]
        s_pml = rcs_fem(coords, conn, solve(coords, conn, p, pol, abc_p), p, cp[2], phis, pol)
        out[pol] = dict(phi=np.degrees(phis), exato=ex, bgt=s_bgt, pml=s_pml)
        e = lambda s: np.max(np.abs(s - ex)) / np.max(ex)
    return out

def plot_rcs(out, bgt_cfg=BGT_CFG, pml_cfg=PML_CFG, fname="rcs_cilindro_PEC.png"):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a, pol in zip(ax, ("TM", "TE")):
        d = out[pol]
        a.plot(d["phi"], d["exato"] / LAM, "k-", lw=2.2, label="analítico")
        a.plot(d["phi"], d["bgt"] / LAM, "r--", lw=1.6, label=f"FEM + BGT-{bgt_cfg['order']} (R = {bgt_cfg['R']/LAM:.2f}λ)")
        a.plot(d["phi"], d["pml"] / LAM, "b:", lw=1.8,
               label=f"FEM + PML (x0 = {pml_cfg['x0']/LAM:.1f}λ, L = {pml_cfg['L']/LAM:.1f}λ, n = {pml_cfg['n']})")
        a.set_xlabel(r"$\phi$ (graus)"); a.set_ylabel(r"$\sigma_{2D}/\lambda$")
        a.set_title(f"Largura de espalhamento - polarização {pol} (cilindro PEC, $a=\\lambda/2$)")
        a.set_xlim(0, 180); a.grid(alpha=.3); a.legend()
    plt.tight_layout()

def plot_results_fem_rcs():
    cfg_b, cfg_p = (auto_configs() if "--auto" in sys.argv else (BGT_CFG, PML_CFG))
    out = run(2, cfg_b, cfg_p)
    plot_rcs(out, cfg_b, cfg_p)
    plt.show()
