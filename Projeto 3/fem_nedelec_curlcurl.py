"""
Elementos de Nédélec (tipo 1, mais baixa ordem) em malhas triangulares - problema 2D
    rot rot u + a u = f   em Omega = [-1,1]^2,       n x u = 0 sobre dOmega,      a = 1
    f(x,y) = (2 pi^2 + 1) [ cos(pi x) sin(pi y) , -sin(pi x) cos(pi y) ]^T
Solução exata:  u = [ cos(pi x) sin(pi y) , -sin(pi x) cos(pi y) ]^T

Observação sobre o enunciado: na 1a componente de f e de u, o enunciado traz "sin(pi x)", mas para
que n x u = 0 em y = +-1 (e para que u resolva a equação) a função deve ser sin(pi y):
    rot u = -2 pi cos(pi x) cos(pi y)   =>   rot rot u = 2 pi^2 u   =>   rot rot u + u = (2 pi^2 + 1) u = f.
O código usa sin(pi y).  (Troque em u_exact/f_source/curl_exact se o seu enunciado for outro.)

Forma fraca em H0(rot, Omega):   (rot u, rot v) + a (u, v) = (f, v)   para todo v em H0(rot, Omega).

Elemento de Nédélec de mais baixa ordem (uma função de base por aresta):
    N_e = lambda_i grad(lambda_j) - lambda_j grad(lambda_i)      (aresta e orientada de i para j)
  Elemento de referência (vértices (0,0), (1,0), (0,1); arestas 1:(1,2), 2:(2,3), 3:(3,1)):
    N^1 = (1 - eta, xi),   N^2 = (-eta, xi),   N^3 = (-eta, xi - 1),   rot N^k = 2
  Mapeamento (transformação de Piola covariante), x = x1 + J xi,  J = [x2-x1, x3-x1]:
    N = J^{-T} N^,        rot N = (1/det J) rot^ N^
  Matrizes do elemento a partir do elemento de referência:
    K_ij = int rot N_i rot N_j dOmega = 2 / |det J|                            (rot N^ = 2)
    M_ij = int N_i . N_j dOmega = |det J| sum_ab G_ab int N^_i,a N^_j,b dxi,   G = (J^T J)^{-1}
    b_i  = int f . N_i dOmega = |det J| int f(x(xi)) . (J^{-T} N^_i) dxi
  O grau de liberdade de cada aresta é a circulação tangencial int_e u.t dl (t: do menor para o maior
  índice global de nó), de modo que as arestas de contorno (n x u = 0) têm grau de liberdade nulo e
  suas equações são ELIMINADAS do sistema global.

Estudo de convergência: malhas com h1/h2 = h2/h3 = h3/h4 = 2 (n = 4, 8, 16, 32 quadrados por lado,
cada quadrado dividido em 2 triângulos; h = 2/n).  Também é possível usar malhas do Gmsh (--gmsh).
alpha = ln(e_{h_i}/e_{h_{i+1}}) / ln 2,   e_h = ||u - u_h||_{L2}.

Dependências: numpy, scipy, matplotlib  (e gmsh, só com --gmsh)
"""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

PI = np.pi
A_COEF = 1.0


# ----------------------------------------------------------------------------
# Problema
# ----------------------------------------------------------------------------
def u_exact(x, y):
    return np.stack([np.cos(PI * x) * np.sin(PI * y), -np.sin(PI * x) * np.cos(PI * y)], axis=-1)


def curl_exact(x, y):
    return -2 * PI * np.cos(PI * x) * np.cos(PI * y)


def f_source(x, y):
    return (2 * PI ** 2 + A_COEF) * u_exact(x, y)


# ----------------------------------------------------------------------------
# Malhas triangulares
# ----------------------------------------------------------------------------
def mesh_structured(n):
    """n x n quadrados em [-1,1]^2, cada um dividido em 2 triângulos (anti-horários)."""
    xs = np.linspace(-1.0, 1.0, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)
    idx = lambda i, j: i * (n + 1) + j
    tri = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tri += [(a, b, c), (a, c, d)]
    return nodes, np.array(tri)


def mesh_gmsh(h):
    """Malha não estruturada do Gmsh (import gmsh dentro do Python), tamanho máximo h."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.add("quadrado")
    geo = gmsh.model.geo
    p = [geo.addPoint(x, y, 0, h) for (x, y) in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
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
# Elemento de referência e mapeamento
# ----------------------------------------------------------------------------
LOCAL_EDGES = [(0, 1), (1, 2), (2, 0)]


def ref_basis(xi):
    """Funções de base de Nédélec no elemento de referência: array (nq, 3, 2)."""
    x, e = xi[:, 0], xi[:, 1]
    N1 = np.stack([1 - e, x], axis=1)
    N2 = np.stack([-e, x], axis=1)
    N3 = np.stack([-e, x - 1], axis=1)
    return np.stack([N1, N2, N3], axis=1)


def quad_tri(n):
    """Gauss no triângulo unitário (Duffy), n x n pontos; pesos somam 1/2."""
    xg, wg = np.polynomial.legendre.leggauss(n)
    xg, wg = 0.5 * (xg + 1), 0.5 * wg
    U, V = np.meshgrid(xg, xg, indexing="ij")
    W = np.outer(wg, wg) * (1 - U)
    return np.stack([U.ravel(), (V * (1 - U)).ravel()], axis=1), W.ravel()


def mapping(nodes, tri):
    """x1, J, det J, J^{-1} de cada elemento (mapeamento afim x = x1 + J xi)."""
    x1, x2, x3 = (nodes[tri[:, k]] for k in range(3))
    J = np.stack([np.stack([x2[:, 0] - x1[:, 0], x3[:, 0] - x1[:, 0]], axis=1),
                  np.stack([x2[:, 1] - x1[:, 1], x3[:, 1] - x1[:, 1]], axis=1)], axis=1)   # (nel,2,2)
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    invJ = np.empty_like(J)
    invJ[:, 0, 0], invJ[:, 0, 1] = J[:, 1, 1] / det, -J[:, 0, 1] / det
    invJ[:, 1, 0], invJ[:, 1, 1] = -J[:, 1, 0] / det, J[:, 0, 0] / det
    return x1, J, det, invJ


# ----------------------------------------------------------------------------
# Arestas: numeração global, orientação e arestas de contorno
# ----------------------------------------------------------------------------
def edge_structure(tri):
    nel = len(tri)
    e = np.stack([tri[:, [i, j]] for (i, j) in LOCAL_EDGES], axis=1)                # (nel,3,2)
    sgn = np.where(e[:, :, 0] < e[:, :, 1], 1.0, -1.0)                               # local x global
    uniq, inv = np.unique(np.sort(e, axis=2).reshape(-1, 2), axis=0, return_inverse=True)
    el_edges = inv.reshape(nel, 3)
    boundary = np.bincount(inv, minlength=len(uniq)) == 1
    return el_edges, sgn, uniq, boundary


# ----------------------------------------------------------------------------
# Matrizes e vetor de carga (expressões baseadas no elemento de referência)
# ----------------------------------------------------------------------------
def element_matrices(nodes, tri, sgn):
    x1, J, det, invJ = mapping(nodes, tri)
    ad = np.abs(det)
    # rigidez: int rot N_i rot N_j = 2/|det J|    (rot N^ = 2 em todas as funções de base)
    Ke = (2.0 / ad)[:, None, None] * sgn[:, :, None] * sgn[:, None, :]
    # massa: |det J| sum_ab G_ab Mhat_ab,ij   com  G = J^{-1} J^{-T}
    xq, wq = quad_tri(3)                                                              # exato até grau 5
    Nh = ref_basis(xq)                                                                # (q,3,2)
    Mhat = np.einsum("q,qia,qjb->ijab", wq, Nh, Nh)                                   # integrais de referência
    G = np.einsum("eak,ebk->eab", invJ, invJ)
    Me = ad[:, None, None] * np.einsum("eab,ijab->eij", G, Mhat) * sgn[:, :, None] * sgn[:, None, :]
    return Ke, Me


def load_vector(nodes, tri, sgn, nq=6):
    x1, J, det, invJ = mapping(nodes, tri)
    xq, wq = quad_tri(nq)
    Nh = ref_basis(xq)                                                                # (q,3,2)
    xp = x1[:, None, :] + np.einsum("eij,qj->eqi", J, xq)                             # pontos físicos
    fq = f_source(xp[..., 0], xp[..., 1])                                             # (e,q,2)
    N = np.einsum("eki,qjk->eqji", invJ, Nh)                                          # J^{-T} N^  (e,q,3,2)
    return np.abs(det)[:, None] * sgn * np.einsum("q,eqi,eqji->ej", wq, fq, N)


def assemble(nodes, tri):
    el_edges, sgn, uniq, boundary = edge_structure(tri)
    Ke, Me = element_matrices(nodes, tri, sgn)
    be = load_vector(nodes, tri, sgn)
    nE = len(uniq)
    rows = np.repeat(el_edges, 3, axis=1).ravel()
    cols = np.tile(el_edges, (1, 3)).ravel()
    A = sp.coo_matrix(((Ke + A_COEF * Me).ravel(), (rows, cols)), shape=(nE, nE)).tocsr()
    b = np.bincount(el_edges.ravel(), weights=be.ravel(), minlength=nE)
    return A, b, el_edges, sgn, boundary


def solve(nodes, tri):
    """Sistema reduzido: as equações das arestas de contorno (n x u = 0) são eliminadas."""
    A, b, el_edges, sgn, boundary = assemble(nodes, tri)
    free = np.where(~boundary)[0]
    u = np.zeros(A.shape[0])
    u[free] = spla.spsolve(A[free][:, free].tocsc(), b[free])
    return u, el_edges, sgn


# ----------------------------------------------------------------------------
# Pós-processamento: erro em L2 (e em rot), campo nos baricentros
# ----------------------------------------------------------------------------
def field_at(nodes, tri, u, el_edges, sgn, xi):
    """u_h nos pontos de referência xi (nq,2): retorna posições (e,q,2) e u_h (e,q,2)."""
    x1, J, det, invJ = mapping(nodes, tri)
    Nh = ref_basis(xi)
    N = np.einsum("eki,qjk->eqji", invJ, Nh)
    coef = sgn * u[el_edges]                                                          # (e,3)
    uh = np.einsum("ej,eqji->eqi", coef, N)
    xp = x1[:, None, :] + np.einsum("eij,qj->eqi", J, xi)
    return xp, uh, det


def errors(nodes, tri, u, el_edges, sgn, nq=6):
    xq, wq = quad_tri(nq)
    xp, uh, det = field_at(nodes, tri, u, el_edges, sgn, xq)
    ex = u_exact(xp[..., 0], xp[..., 1])
    w = wq[None, :] * np.abs(det)[:, None]
    eL2 = np.sqrt(np.sum(w * np.sum((ex - uh) ** 2, axis=2)))
    curl_h = (2.0 / det) * np.sum(sgn * u[el_edges], axis=1)                          # rot u_h (constante)
    ecurl = np.sqrt(np.sum(w * (curl_exact(xp[..., 0], xp[..., 1]) - curl_h[:, None]) ** 2))
    return eL2, ecurl


def self_check():
    """Verificações do elemento: circulação unitária e reprodução de campos do espaço de Nédélec."""
    nodes, tri = mesh_structured(3)
    nodes = nodes + 0.03 * np.sin(5 * nodes[:, ::-1])               # distorce (malha não uniforme)
    el_edges, sgn, uniq, _ = edge_structure(tri)
    # interpolante de v = (c1 - b y, c2 + b x): graus de liberdade = circulação int_e v.t dl (exata, v linear)
    c1, c2, bb = 0.7, -0.4, 1.3
    t = nodes[uniq[:, 1]] - nodes[uniq[:, 0]]                                         # do menor para o maior índice
    mid = 0.5 * (nodes[uniq[:, 0]] + nodes[uniq[:, 1]])
    v = np.stack([c1 - bb * mid[:, 1], c2 + bb * mid[:, 0]], axis=1)
    dof = np.sum(v * t, axis=1)
    xq = np.array([[1 / 3, 1 / 3], [0.2, 0.5]])
    xp, uh, det = field_at(nodes, tri, dof, el_edges, sgn, xq)
    vex = np.stack([c1 - bb * xp[..., 1], c2 + bb * xp[..., 0]], axis=-1)
    curl_h = (2.0 / det) * np.sum(sgn * dof[el_edges], axis=1)
    print(f"[teste] reprodução de v = (c1 - b y, c2 + b x): erro = {np.max(abs(uh - vex)):.2e};  "
          f"rot v = 2b: erro = {np.max(abs(curl_h - 2 * bb)):.2e}")


# ----------------------------------------------------------------------------
# Execução: tabela de convergência e quiver
# ----------------------------------------------------------------------------
def run(use_gmsh=False, ns=(4, 8, 16, 32)):
    results = []
    for n in ns:
        h = 2.0 / n
        nodes, tri = mesh_gmsh(h) if use_gmsh else mesh_structured(n)
        u, el_edges, sgn = solve(nodes, tri)
        eL2, ecurl = errors(nodes, tri, u, el_edges, sgn)
        results.append(dict(h=h, nodes=nodes, tri=tri, u=u, el_edges=el_edges, sgn=sgn, eL2=eL2,
                            ecurl=ecurl, ndof=len(u), nel=len(tri)))
    return results


def print_table(results):
    print(f"\n{'h':>9s} {'#arestas':>9s} {'#elem.':>8s} {'||u-u_h||_L2':>14s} {'alpha':>7s} "
          f"{'||rot(u-u_h)||':>15s} {'alpha_rot':>10s}")
    for i, r in enumerate(results):
        a = "-" if i == 0 else f"{np.log(results[i-1]['eL2'] / r['eL2']) / np.log(2):.3f}"
        ac = "-" if i == 0 else f"{np.log(results[i-1]['ecurl'] / r['ecurl']) / np.log(2):.3f}"
        print(f"{r['h']:9.5f} {r['ndof']:9d} {r['nel']:8d} {r['eL2']:14.6e} {a:>7s} {r['ecurl']:15.6e} {ac:>10s}")


def plot_quiver(results, fname="campo_u_quiver.png"):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    for a, r, nome in ((ax[0], results[0], "h1"), (ax[1], results[-1], "h4")):
        xi = np.array([[1 / 3, 1 / 3]])
        xp, uh, _ = field_at(r["nodes"], r["tri"], r["u"], r["el_edges"], r["sgn"], xi)
        x, y, uh = xp[:, 0, 0], xp[:, 0, 1], uh[:, 0, :]
        ue = u_exact(x, y)
        st = max(1, len(x) // 600)                                      # subamostra se a malha for fina
        mag = np.hypot(uh[::st, 0], uh[::st, 1])
        a.quiver(x[::st], y[::st], ue[::st, 0], ue[::st, 1], color="lightgray", angles="xy",
                 scale_units="xy", scale=None, width=.006, label="exato")
        q = a.quiver(x[::st], y[::st], uh[::st, 0], uh[::st, 1], mag, angles="xy", cmap="viridis", width=.004)
        plt.colorbar(q, ax=a, label=r"$|u_h|$")
        a.triplot(r["nodes"][:, 0], r["nodes"][:, 1], r["tri"], color="k", lw=.2, alpha=.4) \
            if len(r["tri"]) <= 200 else None
        a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y")
        a.set_title(f"Campo $u_h$ (Nédélec, malha {nome}: h = {r['h']:.4f}, {r['nel']} elementos)")
    plt.tight_layout()#; plt.savefig(fname, dpi=140)
    # print("Figura salva em", fname)


def plot_results_p1():
    use_gmsh = "--gmsh" in sys.argv
    self_check()
    results = run(use_gmsh)
    print_table(results)
    plot_quiver(results)
    plt.show()
