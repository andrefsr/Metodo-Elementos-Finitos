"""
Modos TE e TM de um guia de onda retangular (a x b) pelo Método dos Elementos Finitos
Elementos triangulares de Lagrange P1 e P2 (malhas do Gmsh, import gmsh dentro do Python).

Problema de autovalores escalar (k_c^2 = autovalor):
    TM :  -lap(E_z) = k_c^2 E_z       E_z = 0  nas paredes (Dirichlet, eliminado do sistema)
    TE :  -lap(H_z) = k_c^2 H_z       dH_z/dn = 0 nas paredes (Neumann natural)
  forma fraca:  K u = k_c^2 M u,   K = int grad(phi_i).grad(phi_j),  M = int phi_i phi_j

 - TM: as equações dos nós de Dirichlet são ELIMINADAS (só os nós livres entram no sistema).
 - TE: o autovalor nulo (m = n = 0) é descartado.
 - Autovalores analíticos: k_c^2 = (m pi/a)^2 + (n pi/b)^2  (TE: m=n=0 excluído; TM: mn = 0 excluído).
 - Gera: tabela dos 5 menores autovalores com o erro relativo (TE e TM, P1 e P2),
   campos dos modos TE10 e TE20 (H_z e campo elétrico transversal) e a curva de
   dispersão k0 a x beta/k0 para TE10 e TE20 em 1,9-5,0 GHz (passo 100 MHz).

Dependências: numpy, scipy, matplotlib, gmsh   (pip install gmsh)
Unidades: SI.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import gmsh

A_G, B_G = 0.08, 0.04            # a = 8,0 cm, b = 4,0 cm (guia preenchido com ar)
C0 = 299792458.0
H_MAX = A_G / 20                 # tamanho máximo de elemento escolhido (4 mm)


# ----------------------------------------------------------------------------
# Malha (Gmsh): retângulo [0,a] x [0,b], triângulos de 6 nós (ordem 2)
# ----------------------------------------------------------------------------
def generate_mesh(hmax=H_MAX):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)
    gmsh.model.add("guia")
    geo = gmsh.model.geo
    p = [geo.addPoint(x, y, 0, hmax) for (x, y) in [(0, 0), (A_G, 0), (A_G, B_G), (0, B_G)]]
    ln = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    s = geo.addPlaneSurface([geo.addCurveLoop(ln)])
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
    """Espaços P1 (vértices) e P2 (vértices + pontos médios dos lados)."""
    verts = np.unique(conn6[:, :3])
    remap = -np.ones(len(coords), dtype=np.int64)
    remap[verts] = np.arange(len(verts))
    return {1: (coords[verts], remap[conn6[:, :3]]), 2: (coords, conn6)}


def boundary_mask(coords, tol=1e-9):
    x, y = coords[:, 0], coords[:, 1]
    return (x < tol) | (x > A_G - tol) | (y < tol) | (y > B_G - tol)


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


def assemble(coords, conn, p):
    xq, wq = quad_tri(p + 3)
    phi, dphi = basis(p, xq)
    _, det, invJ = geometry(coords, conn, xq, p)
    w = wq[None, :] * np.abs(det)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)
    Ke = np.einsum("eq,eqbi,eqci->ebc", w, G, G)
    Me = np.einsum("eq,qb,qc->ebc", w, phi, phi)
    nb = conn.shape[1]
    n = len(coords)
    rows = np.repeat(conn, nb, axis=1).ravel()
    cols = np.tile(conn, (1, nb)).ravel()
    mk = lambda D: sp.coo_matrix((D.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    return mk(Ke), mk(Me)


# ----------------------------------------------------------------------------
# Autovalores / autovetores
# ----------------------------------------------------------------------------
def analytic_eigs(kind, nmax=12):
    """Lista ordenada [(k_c^2, m, n), ...] (TE: sem (0,0); TM: m,n >= 1)."""
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


def solve_eigen(coords, conn, p, kind, nev=8):
    """
    Retorna (k_c^2 ordenados, autovetores nos nós (M-ortonormais)).
    TM: nós de Dirichlet eliminados.  TE: Neumann natural, autovalor nulo descartado.
    """
    K, M = assemble(coords, conn, p)
    n = len(coords)
    if kind == "TM":
        free = np.where(~boundary_mask(coords))[0]
    else:
        free = np.arange(n)
    Kf, Mf = K[free][:, free].tocsc(), M[free][:, free].tocsc()
    extra = 1 if kind == "TE" else 0
    vals, vecs = spla.eigsh(Kf, k=nev + extra, M=Mf, sigma=-100.0, which="LM")
    o = np.argsort(vals)
    vals, vecs = vals[o], vecs[:, o]
    if kind == "TE":
        vals, vecs = vals[1:], vecs[:, 1:]                    # descarta o autovalor nulo (m = n = 0)
    full = np.zeros((n, len(vals)))
    full[free] = vecs
    return vals, full


def eigen_table(coords, conn, p, kind, nshow=5, nev=8):
    vals, _ = solve_eigen(coords, conn, p, kind, nev)
    an = analytic_eigs(kind)[:nshow]
    rows = []
    for i, (kex, m, n) in enumerate(an):
        rows.append((f"{kind}{m}{n}", kex, vals[i], abs(vals[i] - kex) / kex))
    return rows


def print_table(rows, kind, p, hmax):
    print(f"\n{kind} - P{p} (h_max = {hmax*1e3:.2f} mm)")
    print(f"  {'modo':6s} {'k_c^2 analítico (1/m²)':>24s} {'k_c^2 FEM (1/m²)':>20s} {'erro relativo':>15s}")
    for nome, kex, kh, er in rows:
        print(f"  {nome:6s} {kex:24.6f} {kh:20.6f} {er:15.3e}")


# ----------------------------------------------------------------------------
# Campos dos modos TE10 e TE20
# ----------------------------------------------------------------------------
def mode_field(coords, conn, p, kind, m, n, nev=8):
    """
    Autovetor FEM do modo (m,n), com sinal/mistura ajustados ao modo analítico: em autovalores
    degenerados (TE20 e TE01 têm o mesmo k_c com a = 2b) projeta-se o modo analítico no
    subespaço dos autovetores degenerados.  Retorna (campo normalizado, k_c^2 do modo).
    """
    vals, vecs = solve_eigen(coords, conn, p, kind, nev)
    kex = (m * np.pi / A_G) ** 2 + (n * np.pi / B_G) ** 2
    sel = np.where(np.abs(vals - kex) < 0.03 * kex)[0]            # cluster em torno do autovalor
    _, M = assemble(coords, conn, p)
    fan = np.cos(m * np.pi * coords[:, 0] / A_G) * np.cos(n * np.pi * coords[:, 1] / B_G)
    coef = vecs[:, sel].T @ (M @ fan)                             # vetores M-ortonormais
    u = vecs[:, sel] @ coef
    u /= np.max(np.abs(u))
    return u, vals[sel[np.argmax(np.abs(coef))]]


def grad_at_barycenters(coords, conn, u, p):
    xq = np.array([[1 / 3, 1 / 3]])
    _, dphi = basis(p, xq)
    xp, det, invJ = geometry(coords, conn, xq, p)
    G = np.einsum("qbk,eqki->eqbi", dphi, invJ)[:, 0]
    return xp[:, 0, :], np.einsum("ebi,eb->ei", G, u[conn])


def plot_modes(coords, conn, p, fname="modos_TE10_TE20.png"):
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], conn[:, :3])
    fig, ax = plt.subplots(2, 2, figsize=(13, 6.5))
    for r, (m, n) in enumerate(((1, 0), (2, 0))):
        u, kh = mode_field(coords, conn, p, "TE", m, n)
        kex = (m * np.pi / A_G) ** 2 + (n * np.pi / B_G) ** 2
        xc, g = grad_at_barycenters(coords, conn, u, p)
        # campo elétrico transversal de um modo TE:  E_t ~ z x grad(H_z) = (-dH/dy, dH/dx)
        Ex, Ey = -g[:, 1], g[:, 0]
        c = ax[r, 0].tricontourf(tri, u, 21, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(c, ax=ax[r, 0], label=r"$H_z$ (normalizado)")
        st = max(1, len(xc) // 350)                                  # subamostra as setas
        ax[r, 0].quiver(xc[::st, 0], xc[::st, 1], Ex[::st], Ey[::st], color="k", width=.003)
        ax[r, 0].set_title(f"TE{m}{n} - FEM P{p}:  $k_c^2$ = {kh:.2f} (analítico {kex:.2f})")
        # analítico
        X, Y = np.meshgrid(np.linspace(0, A_G, 200), np.linspace(0, B_G, 100))
        Hz = np.cos(m * np.pi * X / A_G) * np.cos(n * np.pi * Y / B_G)
        c2 = ax[r, 1].contourf(X, Y, Hz, 21, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(c2, ax=ax[r, 1], label=r"$H_z$")
        xs, ys = np.meshgrid(np.linspace(0.004, A_G - 0.004, 20), np.linspace(0.004, B_G - 0.004, 10))
        dHx = -(m * np.pi / A_G) * np.sin(m * np.pi * xs / A_G) * np.cos(n * np.pi * ys / B_G)
        dHy = -(n * np.pi / B_G) * np.cos(m * np.pi * xs / A_G) * np.sin(n * np.pi * ys / B_G)
        ax[r, 1].quiver(xs, ys, -dHy, dHx, color="k", width=.003)
        ax[r, 1].set_title(f"TE{m}{n} - analítico")
        for a_ in ax[r]:
            a_.set_aspect("equal"); a_.set_xlabel("x (m)"); a_.set_ylabel("y (m)")
    plt.tight_layout(); #plt.savefig(fname, dpi=140)
    # print("Figura salva em", fname)


# ----------------------------------------------------------------------------
# Curva de dispersão  k0 a  x  beta/k0   (1,9 - 5,0 GHz, passo de 100 MHz)
# ----------------------------------------------------------------------------
def dispersion(kc2):
    f = np.arange(1.9e9, 5.0e9 + 1, 0.1e9)
    k0 = 2 * np.pi * f / C0
    beta = np.sqrt((k0 ** 2 - kc2).astype(complex))                # abaixo do corte: beta = -j alpha
    return f, k0, beta


def plot_dispersion(kc2_fem, fname="dispersao_guia.png"):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cores = {"TE10": "tab:blue", "TE20": "tab:red"}
    for nome, (m, n) in (("TE10", (1, 0)), ("TE20", (2, 0))):
        kc2_an = (m * np.pi / A_G) ** 2 + (n * np.pi / B_G) ** 2
        f, k0, b_an = dispersion(np.array(kc2_an))
        _, _, b_h = dispersion(np.array(kc2_fem[nome]))
        ax.plot(k0 * A_G, np.real(b_an) / k0, "-", color=cores[nome], lw=1.5, label=f"{nome} analítico")
        ax.plot(k0 * A_G, np.real(b_h) / k0, "o", color=cores[nome], ms=4, mfc="none", label=f"{nome} FEM")
        ax.plot(k0 * A_G, -np.imag(b_an) / k0, "--", color=cores[nome], lw=1, label=f"{nome} atenuação $\\alpha/k_0$ (abaixo do corte)")
    ax.plot(k0 * A_G, np.ones_like(k0), ":", color="gray", label="$\\beta = k_0$ (linha de luz)")
    ax.set_xlabel(r"$k_0 a$"); ax.set_ylabel(r"$\beta/k_0$")
    ax.set_title("Curva de dispersão normalizada (1,9 - 5,0 GHz)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout(); #plt.savefig(fname, dpi=140)
    # print("Figura salva em", fname)


def plot_results_fem_guia():
    print(f"Guia a = {A_G*100:.1f} cm, b = {B_G*100:.1f} cm; h_max = {H_MAX*1e3:.1f} mm; "
          f"f_c(TE10) = {C0/(2*A_G)/1e9:.3f} GHz, f_c(TE20) = {C0/A_G/1e9:.3f} GHz")

    # --- escolha do h_max: erro máximo entre os 5 menores autovalores
    print("\nEstudo de h_max (maior erro relativo entre os 5 menores autovalores):")
    for hm in (A_G / 10, A_G / 20, A_G / 40):
        c6, cn6 = generate_mesh(hm)
        sp_ = build_spaces(c6, cn6)
        for p in (1, 2):
            e = {k: max(r[3] for r in eigen_table(*sp_[p], p, k)) for k in ("TE", "TM")}
            print(f"  h_max = {hm*1e3:5.2f} mm  P{p}: TE = {e['TE']:.2e}   TM = {e['TM']:.2e}   (nós = {len(sp_[p][0])})")

    # --- tabelas com o h_max escolhido
    coords6, conn6 = generate_mesh(H_MAX)
    spaces = build_spaces(coords6, conn6)
    # for p in (1, 2):
    #     for kind in ("TE", "TM"):
    #         print_table(eigen_table(*spaces[p], p, kind), kind, p, H_MAX)

    # --- campos TE10 e TE20 e curva de dispersão (P2)
    plot_modes(*spaces[2], 2)
    kc2 = {"TE10": mode_field(*spaces[2], 2, "TE", 1, 0)[1], "TE20": mode_field(*spaces[2], 2, "TE", 2, 0)[1]}
    plot_dispersion(kc2)
    plt.show()
    return
