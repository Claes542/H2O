#!/usr/bin/env python3
"""
Does H2 still bind when the electron-electron interface carries the surface tension
the electron gas requires?

RealQM gives each electron its own domain. In H2 the two domains meet at the MIDPLANE
between the nuclei, fixed there by symmetry -- so the free-boundary machinery is not
needed and the interface condition can be imposed exactly. That also makes H2 the right
molecule for this test: in helium the two domains meet at a plane THROUGH the nucleus,
where interface error and Coulomb-cusp error superimpose and cannot be separated.

The interface condition is Robin,

    d(psi)/dn + beta*psi = 0        (outward normal of the z>0 domain points in -z,
                                     so this reads  d(psi)/dz = beta*psi  at z=0)

which is the natural boundary condition of a surface energy (beta/2) * Int psi^2 dS.
beta is therefore a SURFACE TENSION of the electron-electron interface. beta=0 is
Neumann, the framework's present choice and zero surface tension; beta->infinity is
Dirichlet, a hard node between the atoms.

    beta = 0        H2 as RealQM currently computes it
    beta > 0        density pushed off the midplane -- the bond region is drained
    beta = 1.146/a  the value the degenerate electron gas requires, with a the cell size

Since "a" is not uniquely defined for a molecule, we scan beta and report where binding
dies, then convert that to the cell size a_crit = 1.146/beta_crit at which the gas value
would be reached. Comparing a_crit with molecular length scales is the result.

Geometry: cylindrical (r,z), axially symmetric. Electron 1 occupies z >= 0, electron 2 is
its mirror image. Nuclei on the axis at z = +/- R/2. Self-consistent: each electron feels
both nuclei plus the Hartree potential of the other electron, and NO self-interaction --
which is RealQM's own structure, not an approximation.

E(H2) = 2*[T + Int V_ne rho] + Int V_H[rho2] rho1 + 1/R   [+ surface energy if counted]
E(H)  = same grid, full space, one nucleus, no interface -- so discretisation error
        largely cancels in E_bind = 2 E(H) - E(H2).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import sys

# ----------------------------------------------------------------- grid
#
# BOTH axes are half-integer (cell centres), so every physical plane -- the interface
# at z=0 and the nuclear planes at z=+/-R/2 -- lies on a cell FACE. Three things follow,
# and the first attempt at this calculation failed for want of all three:
#   * the mirror z -> -z is exact, with no layer shared between the two domains;
#   * the Robin condition is applied at a face, where a flux condition belongs;
#   * every nucleus sits at the SAME offset relative to the grid, in the atom and in
#     the molecule alike, so the cusp error is the same in both and cancels in
#     E_bind = 2 E(H) - E(H2).
# R must therefore be an even multiple of H.
H     = 0.05          # spacing, a0
RMAX  = 7.0           # radial extent
ZMAX  = 8.0           # axial half-extent

NR = int(round(RMAX / H))
rr = (np.arange(NR) + 0.5) * H          # cell centres: r=0 is a face, never sampled


def zgrid(zlo, zhi):
    """cell centres of [zlo, zhi]; the endpoints are faces."""
    n = int(round((zhi - zlo) / H))
    return zlo + (np.arange(n) + 0.5) * H


def coulomb(rgrid, zg, z0):
    """-1/|x - (0,0,z0)| on the (r,z) mesh, regularised at the cell scale."""
    R2 = rgrid[None, :] ** 2 + (zg[:, None] - z0) ** 2
    return -1.0 / np.sqrt(R2 + (0.35 * H) ** 2)


def volume(zg):
    """cell volumes 2*pi*r*dr*dz. All cells are full: the boundaries are faces."""
    return np.broadcast_to((2 * np.pi * rr * H * H)[None, :], (len(zg), NR)).copy()


# ----------------------------------------------------------------- operators
def laplacian(zg, beta=None):
    """
    Cylindrical Laplacian on (z, r).  Dirichlet at r = RMAX and at the far z faces.
    If beta is not None, the z = zg[0] face carries the Robin condition
    d(psi)/dz = beta*psi, implemented with the ghost point psi_{-1} = psi_1 - 2 h beta psi_0.
    """
    NZ = len(zg)
    N = NZ * NR
    idx = lambda k, j: k * NR + j
    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    for k in range(NZ):
        for j in range(NR):
            i = idx(k, j)
            diag = 0.0
            # ---- radial:  d2/dr2 + (1/r) d/dr, conservative form
            rp, rm = rr[j] + 0.5 * H, rr[j] - 0.5 * H
            if j + 1 < NR:
                add(i, idx(k, j + 1), rp / (rr[j] * H * H)); diag -= rp / (rr[j] * H * H)
            else:
                diag -= rp / (rr[j] * H * H)                     # psi = 0 outside
            if j - 1 >= 0:
                add(i, idx(k, j - 1), rm / (rr[j] * H * H)); diag -= rm / (rr[j] * H * H)
            else:
                diag -= 0.0                                       # r=0: no flux by symmetry
            # ---- axial
            if k + 1 < NZ:
                add(i, idx(k + 1, j), 1.0 / (H * H)); diag -= 1.0 / (H * H)
            else:
                diag -= 1.0 / (H * H)                             # psi = 0 far away
            if k - 1 >= 0:
                add(i, idx(k - 1, j), 1.0 / (H * H)); diag -= 1.0 / (H * H)
            else:
                if beta is None:
                    diag -= 1.0 / (H * H)                         # psi = 0 (full-space atom case)
                else:
                    # Robin at the z=0 FACE.  d(psi)/dz = beta*psi there, so with the
                    # face value (psi_0+psi_g)/2 and slope (psi_0-psi_g)/H,
                    #     psi_g = gamma * psi_0,  gamma = (1 - beta H/2)/(1 + beta H/2).
                    # gamma=1 is Neumann (zero flux); gamma=-1 is Dirichlet.
                    gamma = (1.0 - 0.5 * beta * H) / (1.0 + 0.5 * beta * H)
                    diag += gamma / (H * H)                        # the ghost folds into the diagonal
                    diag -= 1.0 / (H * H)
            add(i, i, diag)
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def poisson_solver(zg):
    """Factorise -laplacian once; reuse for every Hartree solve."""
    L = laplacian(zg, beta=None)
    return spl.factorized((-L).tocsc()), L


def hartree(solve, zg, rho, vol):
    """
    V_H with  lap V = -4 pi rho, Dirichlet V = Q/|x| on the closed faces.
    The boundary contribution is folded into the right-hand side.
    """
    NZ = len(zg); N = NZ * NR
    Q = float((rho * vol).sum())
    b = 4 * np.pi * rho.reshape(N).copy()
    # far-field monopole on the Dirichlet faces
    bd = np.zeros((NZ, NR))
    rad = lambda z, r: np.sqrt(r * r + z * z)
    bd[:, NR - 1] += Q / rad(zg, rr[NR - 1] + H) / (H * H) * ((rr[NR - 1] + 0.5 * H) / rr[NR - 1])
    bd[NZ - 1, :] += Q / rad(zg[NZ - 1] + H, rr) / (H * H)
    bd[0, :] += Q / rad(zg[0] - H, rr) / (H * H)
    b += bd.reshape(N)
    return solve(b).reshape(NZ, NR)


def ground_state(V, zg, beta):
    """Lowest eigenpair of -1/2 lap + V, with the interface condition."""
    L = laplacian(zg, beta=beta)
    Hm = (-0.5 * L + sp.diags(V.reshape(-1))).tocsc()
    w, v = spl.eigsh(Hm, k=1, sigma=-3.0, which='LM', maxiter=5000)
    psi = np.abs(v[:, 0]).reshape(len(zg), NR)
    return float(w[0]), psi


# ----------------------------------------------------------------- systems
def hydrogen():
    """One electron, one nucleus, full space, no interface. Same grid -> errors cancel."""
    zg = zgrid(-ZMAX, ZMAX)
    vol = volume(zg)
    V = coulomb(rr, zg, 0.0)
    solve, L = poisson_solver(zg)
    _, psi = ground_state(V, zg, beta=None)
    rho = psi ** 2
    rho /= (rho * vol).sum()
    p = np.sqrt(rho).reshape(-1)
    T = 0.5 * float(-(p * (L @ p) * vol.reshape(-1)).sum())
    Ene = float((V * rho * vol).sum())
    return T + Ene


def h2(R, beta, verbose=False):
    """Electron 1 on z >= 0; electron 2 its mirror image. Robin at z = 0."""
    assert abs(round(R / (2 * H)) - R / (2 * H)) < 1e-9, "R must be an even multiple of H"
    zg = zgrid(0.0, ZMAX)                    # the domain of electron 1
    zfull = zgrid(-ZMAX, ZMAX)               # for the Hartree solve
    vol = volume(zg); volf = volume(zfull)
    nz1 = len(zg); k0 = len(zfull) // 2      # zfull[k0] is the first cell with z > 0

    Vne = coulomb(rr, zg, +R / 2) + coulomb(rr, zg, -R / 2)
    solvef, Lf = poisson_solver(zfull)
    L1 = laplacian(zg, beta=beta)

    # start from an atomic guess on the right nucleus
    rho = np.exp(-2 * np.sqrt(rr[None, :] ** 2 + (zg[:, None] - R / 2) ** 2))
    rho /= (rho * vol).sum()

    E = None
    for it in range(200):
        # electron 2 = mirror of electron 1, laid on the full grid
        # exact mirror: zfull[k0-1-k] = -zg[k], no shared layer
        rho2 = np.zeros_like(volf)
        rho2[k0 - 1::-1, :] = rho[:k0, :] if nz1 >= k0 else rho
        rho2 /= (rho2 * volf).sum()
        VH_full = hartree(solvef, zfull, rho2, volf)
        VH = VH_full[k0:k0 + nz1, :]

        mu, psi = ground_state(Vne + VH, zg, beta=beta)
        rnew = psi ** 2
        rnew /= (rnew * vol).sum()
        d = float(np.abs(rnew - rho).sum() * H * H)
        rho = 0.25 * rnew + 0.75 * rho
        rho /= (rho * vol).sum()
        if d < 1e-9:
            break

    p = np.sqrt(rho).reshape(-1)
    T = 0.5 * float(-(p * (L1 @ p) * vol.reshape(-1)).sum())
    Ene = float((Vne * rho * vol).sum())
    Eee = float((VH * rho * vol).sum())
    # psi^2 at the z=0 FACE, from the cell value and its ghost
    gam = (1.0 - 0.5 * beta * H) / (1.0 + 0.5 * beta * H)
    psif = 0.5 * (1.0 + gam) * np.sqrt(rho[0, :])
    Esurf = beta * float((psif ** 2 * (2 * np.pi * rr * H)).sum())   # 2 * (beta/2) * Int psi^2 dS
    Etot = 2 * (T + Ene) + Eee + 1.0 / R
    if verbose:
        print(f"      T={T:+.5f} Vne={Ene:+.5f} Vee={Eee:+.5f} Vnn={1/R:+.5f} "
              f"Esurf={Esurf:+.5f} iters={it+1} drho={d:.2e}")
    return Etot, Etot + Esurf


if __name__ == '__main__':
    print(__doc__.split('Geometry:')[0])
    print(f"grid h={H}  RMAX={RMAX}  ZMAX={ZMAX}  ({NR} radial points)\n")

    EH = hydrogen()
    print(f"E(H) on this grid = {EH:+.5f} Ha   (exact -0.50000, error {EH+0.5:+.5f})\n")

    R = float(sys.argv[1]) if len(sys.argv) > 1 else 1.40
    betas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.146, 1.5, 2.0]

    print(f"R = {R} a0        E_bind = 2 E(H) - E(H2),  positive = bound\n")
    print(f"{'beta':>6} {'a=1.146/b':>10} {'E(H2)':>10} {'E_bind':>10} {'eV':>8} "
          f"{'+surf':>10} {'eV':>8}")
    print('-' * 68)
    for b in betas:
        E, Es = h2(R, b, verbose=('-v' in sys.argv))
        bind = 2 * EH - E
        binds = 2 * EH - Es
        acrit = ('%10.2f' % (1.146 / b)) if b > 0 else '       inf'
        print(f"{b:6.2f} {acrit} {E:10.5f} {bind:10.5f} {bind*27.2114:8.3f} "
              f"{binds:10.5f} {binds*27.2114:8.3f}")
