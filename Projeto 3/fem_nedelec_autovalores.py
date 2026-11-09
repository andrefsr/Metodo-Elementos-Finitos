"""
Autovalores e autovetores de um guia de onda retangular (a x b) com elementos de Nédélec
(tipo 1, mais baixa ordem) em malhas triangulares.

Problema vetorial 2D:     rot rot u = lambda u   em Omega = [0,a] x [0,b]
    (TE)   n x u = 0            sobre dOmega      (u = campo elétrico transversal)
    (TM)   n x rot u = 0        sobre dOmega      (u = campo magnético transversal; condição NATURAL)

Em 2D, psi = rot u satisfaz  -lap(psi) = lambda psi  (para lambda != 0, u = rot^T psi / lambda), logo
    n x u = 0       <=>  d psi/dn = 0 (Neumann)   ->  lambda = (m pi/a)^2 + (n pi/b)^2,  (m,n) != (0,0)   [TE]
    n x rot u = 0   <=>  psi = 0     (Dirichlet)  ->  lambda = (m pi/a)^2 + (n pi/b)^2,  m,n >= 1        [TM]
Além desses, rot rot tem o autovalor lambda = 0 com multiplicidade enorme (campos gradientes). Eles são
físicos (núcleo do rotacional) e não são "espúrios"; devem ser ignorados. O código usa a técnica
shift-and-invert com um deslocamento logo acima de cada autovalor exato procurado, de modo que o ARPACK
devolve diretamente os modos físicos e nunca o núcleo.

Forma fraca:  (rot u, rot v) = lambda (u, v),  K u = lambda M u,  em H0(rot) (TE: equações das arestas de
contorno ELIMINADAS) ou em H(rot) (TM: todas as arestas livres).

Elemento de Nédélec (aresta i->j): N = lambda_i grad lambda_j - lambda_j grad lambda_i.  Elemento de
referência: N^1 = (1-eta, xi), N^2 = (-eta, xi), N^3 = (-eta, xi-1), rot N^ = 2.  Mapeamento de Piola:
N = J^{-T} N^, rot N = rot^ N^/det J.  Matrizes do elemento a partir do elemento de referência:
    K_ij = 2/|det J|,     M_ij = |det J| sum_ab G_ab int N^_i,a N^_j,b dxi,   G = (J^T J)^{-1}.

Tabela: cinco menores autovalores (exato e FEM com h1, h2, h3, h1/h2 = h2/h3 = 2) e
    alpha = ln( (lambda_h1 - lambda_h2) / (lambda_h2 - lambda_h3) ) / ln 2.

Dependências: numpy, scipy, matplotlib  (e gmsh, só com --gmsh)
Unidades: SI (a = 8 cm, b = 4 cm, ar).
"""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

A_G, B_G = 0.08, 0.04


# ----------------------------------------------------------------------------
# Autovalores exatos
# ----------------------------------------------------------------------------
def exact_eigs(kind, nmax=12):
    """[(lambda, m, n), ...] ordenados (com multiplicidade). TE: (m,n) != (0,0); TM: m,n >= 1."""
    L = []
    for m in range(nmax + 1):
        for n in range(nmax + 1):
            if kind == "TE" and m == 0 and n == 0:
                continue
            if kind == "TM" and (m == 0 or n == 0):
                continue
            L.append(((m * np.pi / A_G) ** 2 + (n * np.pi / B_G) ** 2, m, n))
    L.sort(key=lambda t: (round(t[0], 6), -t[1], t[2]))
    return L


# ----------------------------------------------------------------------------
# Malhas triangulares
# ----------------------------------------------------------------------------
def mesh_rect(nx, ny):
    """nx x ny retângulos em [0,a] x [0,b], cada um dividido em 2 triângulos (anti-horários)."""
    xs, ys = np.linspace(0, A_G, nx + 1), np.linspace(0, B_G, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)
    idx = lambda i, j: i * (ny + 1) + j
    tri = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tri += [(a, b, c), (a, c, d)]
    return nodes, np.array(tri)


def mesh_gmsh(h):
    """Malha não estruturada do Gmsh (import gmsh dentro do Python), tamanho máximo h."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.add("guia")
    geo = gmsh.model.geo
    p = [geo.addPoint(x, y, 0, h) for (x, y) in ((0, 0), (A_G, 0), (A_G, B_G), (0, B_G))]
    ln = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    geo.addPlaneSurface([geo.addCurveLoop(ln)])
    geo.synchronize()
    gmsh.model.mesh.generate(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    etypes, _, enodes = gmsh.model.mesh.getElements(2)
    gmsh.finalize()
    t = np.asarray(enodes[list(etypes).index(2)], dtype=np.int64).reshape(-1, 3)
    used = np.unique(t)
    nodes = np.zeros((int(max(tags)) + 1, 3))
    nodes[np.asarray(tags, dtype=np.int64)] = np.asarray(xyz).reshape(-1, 3)
    nodes, tri = nodes[used][:, :2], np.searchsorted(used, t)
    a, b, c = (nodes[tri[:, k]] for k in range(3))
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    tri[det < 0, 1], tri[det < 0, 2] = tri[det < 0, 2].copy(), tri[det < 0, 1].copy()
    return nodes, tri


# ----------------------------------------------------------------------------
# Elemento de referência, mapeamento, arestas
# ----------------------------------------------------------------------------
LOCAL_EDGES = [(0, 1), (1, 2), (2, 0)]


def ref_basis(xi):
    x, e = xi[:, 0], xi[:, 1]
    return np.stack([np.stack([1 - e, x], axis=1), np.stack([-e, x], axis=1),
                     np.stack([-e, x - 1], axis=1)], axis=1)                          # (nq, 3, 2)


def quad_tri(n):
    xg, wg = np.polynomial.legendre.leggauss(n)
    xg, wg = 0.5 * (xg + 1), 0.5 * wg
    U, V = np.meshgrid(xg, xg, indexing="ij")
    return np.stack([U.ravel(), (V * (1 - U)).ravel()], axis=1), (np.outer(wg, wg) * (1 - U)).ravel()


def mapping(nodes, tri):
    x1, x2, x3 = (nodes[tri[:, k]] for k in range(3))
    J = np.stack([np.stack([x2[:, 0] - x1[:, 0], x3[:, 0] - x1[:, 0]], axis=1),
                  np.stack([x2[:, 1] - x1[:, 1], x3[:, 1] - x1[:, 1]], axis=1)], axis=1)
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    invJ = np.empty_like(J)
    invJ[:, 0, 0], invJ[:, 0, 1] = J[:, 1, 1] / det, -J[:, 0, 1] / det
    invJ[:, 1, 0], invJ[:, 1, 1] = -J[:, 1, 0] / det, J[:, 0, 0] / det
    return x1, J, det, invJ


def edge_structure(tri):
    nel = len(tri)
    e = np.stack([tri[:, [i, j]] for (i, j) in LOCAL_EDGES], axis=1)
    sgn = np.where(e[:, :, 0] < e[:, :, 1], 1.0, -1.0)                               # local x global
    uniq, inv = np.unique(np.sort(e, axis=2).reshape(-1, 2), axis=0, return_inverse=True)
    return inv.reshape(nel, 3), sgn, uniq, np.bincount(inv, minlength=len(uniq)) == 1


def assemble_KM(nodes, tri):
    """Matrizes globais K (rot-rot) e M (massa) a partir do elemento de referência."""
    el_edges, sgn, uniq, boundary = edge_structure(tri)
    x1, J, det, invJ = mapping(nodes, tri)
    ad = np.abs(det)
    Ke = (2.0 / ad)[:, None, None] * sgn[:, :, None] * sgn[:, None, :]               # int rot N_i rot N_j
    xq, wq = quad_tri(3)                                                              # exato até grau 5
    Nh = ref_basis(xq)
    Mhat = np.einsum("q,qia,qjb->ijab", wq, Nh, Nh)
    G = np.einsum("eak,ebk->eab", invJ, invJ)                                         # (J^T J)^{-1}
    Me = ad[:, None, None] * np.einsum("eab,ijab->eij", G, Mhat) * sgn[:, :, None] * sgn[:, None, :]
    nE = len(uniq)
    rows = np.repeat(el_edges, 3, axis=1).ravel()
    cols = np.tile(el_edges, (1, 3)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(nE, nE)).tocsr()
    M = sp.coo_matrix((Me.ravel(), (rows, cols)), shape=(nE, nE)).tocsr()
    return K, M, el_edges, sgn, boundary


# ----------------------------------------------------------------------------
# Problema de autovalores
# ----------------------------------------------------------------------------
def solve_eigen(nodes, tri, kind, nwant=5):
    """
    Retorna (autovalores fisicos ordenados, autovetores nas arestas (M-ortonormais), el_edges, sgn).
    TE: n x u = 0 (arestas de contorno eliminadas);  TM: n x rot u = 0 (natural, todas as arestas).

    O núcleo do rotacional (lambda = 0, campos gradientes) tem multiplicidade enorme e atrapalha o ARPACK
    se o deslocamento ficar longe dos modos de interesse.  Por isso usa-se shift-and-invert com um
    deslocamento sigma LOGO ACIMA de cada autovalor exato (1,003 lambda_exato) e pedem-se só tantos
    autovalores quanto a multiplicidade exata: eles são os mais próximos de sigma (o zero está a ~lambda
    de distância).  Os modos físicos encontrados são então reunidos e ordenados.
    """
    K, M, el_edges, sgn, boundary = assemble_KM(nodes, tri)
    nE = K.shape[0]
    free = np.where(~boundary)[0] if kind == "TE" else np.arange(nE)
    Kf, Mf = K[free][:, free].tocsc(), M[free][:, free].tocsc()
    lista = exact_eigs(kind)[:nwant + 2]
    alvos = []                                                      # [lambda exato, multiplicidade]
    for lam_e, _, _ in lista[:nwant]:
        if not any(abs(lam_e - t[0]) / t[0] < 1e-6 for t in alvos):
            alvos.append([lam_e, sum(1 for l, _, _ in lista if abs(l - lam_e) / lam_e < 1e-6)])
    vals, vecs = [], []
    for lam_e, mult in alvos:
        v, w = spla.eigsh(Kf, k=mult, M=Mf, sigma=1.003 * lam_e, which="LM", tol=1e-12)
        if np.any(np.abs(v - lam_e) > 0.15 * lam_e):
            print(f"  aviso: autovalor FEM {v} longe do exato {lam_e:.2f} (malha muito grossa?)")
        vals += list(v)
        vecs.append(w)
    vals, vecs = np.array(vals), np.hstack(vecs)
    o = np.argsort(vals)[:nwant]
    full = np.zeros((nE, len(o)))
    full[free] = vecs[:, o]
    return vals[o], full, el_edges, sgn


def run(kind, meshes, use_gmsh=False):
    lam = []
    last = None
    for m in meshes:
        nodes, tri = mesh_gmsh(m) if use_gmsh else mesh_rect(m, m // 2)
        vals, vecs, el_edges, sgn = solve_eigen(nodes, tri, kind)
        lam.append(vals[:5])
        last = (nodes, tri, vecs, el_edges, sgn)
    return np.array(lam), last


def convergence_order(l1, l2, l3):
    r = (l1 - l2) / (l2 - l3)
    return np.log(r) / np.log(2) if r > 0 else np.nan


def print_table(kind, lam, meshes, use_gmsh):
    ex = exact_eigs(kind)[:5]
    hs = [(m if use_gmsh else A_G / m) for m in meshes]
    print(f"\n{kind}  (a = {A_G*100:.0f} cm, b = {B_G*100:.0f} cm):  h1 = {hs[0]*100:.3f} cm, "
          f"h2 = {hs[1]*100:.3f} cm, h3 = {hs[2]*100:.3f} cm   (autovalores em 1/m^2)")
    print(f"  {'':4s} {'modo':>6s} {'exato':>12s} {'h1':>12s} {'h2':>12s} {'h3':>12s} {'alpha':>7s}")
    for i, (lex, m, n) in enumerate(ex):
        a = convergence_order(lam[0, i], lam[1, i], lam[2, i])
        print(f"  l{i+1:<3d} {kind}{m}{n:<4d} {lex:12.4f} {lam[0,i]:12.4f} {lam[1,i]:12.4f} {lam[2,i]:12.4f} {a:7.3f}")


# ----------------------------------------------------------------------------
# Autovetores (quiver nos baricentros)
# ----------------------------------------------------------------------------
def plot_eigenvectors(kind, last, fname):
    nodes, tri, vecs, el_edges, sgn = last
    x1, J, det, invJ = mapping(nodes, tri)
    Nh = ref_basis(np.array([[1 / 3, 1 / 3]]))
    N = np.einsum("eki,qjk->eqji", invJ, Nh)[:, 0]                                    # (nel, 3, 2)
    xc = x1 + np.einsum("eij,j->ei", J, np.array([1 / 3, 1 / 3]))
    ex = exact_eigs(kind)
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.5))
    ax.ravel()[-1].axis("off")                                  # 5 autovetores em 6 painéis
    for k, a in enumerate(ax.ravel()[:vecs.shape[1]]):
        u = np.einsum("ej,ejc->ec", sgn * vecs[:, k][el_edges], N)
        sel = np.sort(np.random.default_rng(0).choice(len(xc), min(len(xc), 450), replace=False))
        a.quiver(xc[sel, 0], xc[sel, 1], u[sel, 0], u[sel, 1], np.hypot(u[sel, 0], u[sel, 1]),
                 cmap="viridis", angles="xy")
        a.set_aspect("equal"); a.set_xlim(0, A_G); a.set_ylim(0, B_G)
        lex, m, n = ex[k]
        a.set_title(f"autovetor {k+1} ({kind}{m}{n}),  $\\lambda$ = {lex:.1f}")
    plt.tight_layout()#; plt.savefig(fname, dpi=140)
    #print("Figura salva em", fname)


def plot_results_p2():
    use_gmsh = "--gmsh" in sys.argv
    # malhas com h1/h2 = h2/h3 = 2:  nx = 16, 32, 64 retângulos em x (ny = nx/2), h = a/nx
    meshes = [0.005, 0.0025, 0.00125] if use_gmsh else [16, 32, 64]
    for kind in ("TE", "TM"):
        lam, last = run(kind, meshes, use_gmsh)
        print_table(kind, lam, meshes, use_gmsh)
        plot_eigenvectors(kind, last, f"autovetores_{kind}.png")
    plt.show()
