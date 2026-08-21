#!/usr/bin/env python3
"""
Does H2 still bind when the electron-electron interface carries the surface tension
the degenerate electron gas requires?

RealQM gives each electron its own domain. In H2 the two domains meet at the MIDPLANE
between the nuclei, fixed there by symmetry -- so the free-boundary machinery is not
needed and the interface condition can be imposed exactly. That is also why H2 is the
right molecule: in helium the domains meet at a plane THROUGH the nucleus, where
interface error and Coulomb-cusp error superimpose and cannot be separated.

VARIATIONAL FORMULATION.  Everything is built as a quadratic form and minimised, which
is both the correct discretisation and a direct instantiation of the claim being tested.
For one electron on its domain,

    E[u] = 1/2 Int |grad u|^2 dV  +  Int V u^2 dV  +  beta/2 Surf u^2 dS,
    subject to Int u^2 dV = 1,

whose stationarity condition is the generalised eigenproblem

    ( 1/2 A  +  M_V  +  beta/2 S ) u  =  mu  M u

with A the (symmetric) stiffness matrix, M the diagonal mass matrix of cell volumes,
M_V = diag(V_j * vol_j), and S the diagonal surface matrix on the interface layer.
NO GHOST POINTS ARE NEEDED: the Robin condition d(u)/dn + beta u = 0 is the natural
boundary condition of the surface term, so writing the energy down is enough. beta is
therefore literally a SURFACE TENSION of the electron-electron interface, and beta = 0,
the framework's present choice, is zero surface tension.

    beta = 0        H2 as RealQM currently computes it (Neumann, free interface)
    beta = 1.146/a  the value the degenerate electron gas requires, a = cell size
    beta -> inf     Dirichlet, a hard node between the atoms

Since "a" is not uniquely defined for a molecule we scan beta, find where binding dies,
and report the cell size a_crit = 1.146/beta_crit at which the gas value would be
reached. Comparing a_crit with molecular length scales is the result.

GRID.  Both axes are cell-centred, so every physical plane -- the interface at z=0 and
the nuclear planes at z = +/- R/2 -- lies on a cell FACE. Hence the mirror z -> -z is
exact with no shared layer; the surface term sits on a face, where it belongs; and every
nucleus has the SAME position relative to the grid in the atom as in the molecule, so the
cusp error is common to both and largely cancels in E_bind = 2 E(H) - E(H2). R must be an
even multiple of the spacing.

WHY THE EARLIER ATTEMPTS FAILED.  The first used a non-symmetric finite-difference
cylindrical Laplacian -- the radial coefficients r_{j+1/2}/(r_j h^2) differ between
neighbours -- while eigsh assumes symmetry, so the eigenvalues were unreliable and the
self-consistency never converged (drho stalled near 7e-2, and the binding trend
zigzagged where it must fall monotonically). Building the energy as a quadratic form
fixes this by construction: A is symmetric because it is assembled face by face.

SECOND GATE.  V_ee must be near 1/d for centroid separation d. An earlier version dropped
the monopole term from the Hartree boundary condition, giving V_ee = 0.333 where ~0.55 was
due, and since hydrogen carries no Hartree term the error did not cancel: it lowered E(H2)
alone and inflated E_bind to 9.4 eV. The term is now included.

VALIDATION GATE.  E(H) must come out near -0.5 Ha on the same grid. Nothing else in the
output means anything until it does, and the script says so.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import sys

H    = 0.05        # 1.40 / (2H) = 14, so the equilibrium bond lands on the grid
RMAX = 8.0
ZMAX = 8.0
NR   = int(round(RMAX / H))
rr   = (np.arange(NR) + 0.5) * H          # cell centres; r = 0 is a face, never sampled


def zgrid(zlo, zhi):
    n = int(round((zhi - zlo) / H))
    return zlo + (np.arange(n) + 0.5) * H


def volumes(zg):
    return np.broadcast_to((2 * np.pi * rr * H * H)[None, :], (len(zg), NR)).copy()


def stiffness(zg, dirichlet_z=False):
    """
    Symmetric A with  u^T A u = Int |grad u|^2 dV, axisymmetric cylindrical.

    Radial face between j and j+1 carries weight 2 pi r_{j+1/2};
    axial face between k and k+1 carries weight 2 pi r_j.
    A face to the outside is included only where a Dirichlet wall is wanted (u_ghost = 0);
    every omitted face is a natural zero-flux (Neumann) boundary -- which is exactly what
    the interface at z = 0 must be before the surface term is added.
    """
    NZ = len(zg)
    idx = lambda k, j: k * NR + j
    rows, cols, vals = [], [], []
    diag = np.zeros(NZ * NR)

    def face(i1, i2, w):
        diag[i1] += w; diag[i2] += w
        rows.append(i1); cols.append(i2); vals.append(-w)
        rows.append(i2); cols.append(i1); vals.append(-w)

    for k in range(NZ):
        for j in range(NR):
            i = idx(k, j)
            if j + 1 < NR:
                face(i, idx(k, j + 1), 2 * np.pi * (rr[j] + 0.5 * H))
            else:
                diag[i] += 2 * np.pi * (rr[j] + 0.5 * H)       # u = 0 outside
            if k + 1 < NZ:
                face(i, idx(k + 1, j), 2 * np.pi * rr[j])
            else:
                diag[i] += 2 * np.pi * rr[j]                    # u = 0 far away (+z)
            if k == 0 and dirichlet_z:
                diag[i] += 2 * np.pi * rr[j]                    # u = 0 far away (-z)
    rows.extend(range(NZ * NR)); cols.extend(range(NZ * NR)); vals.extend(diag)
    return sp.csr_matrix((vals, (rows, cols)), shape=(NZ * NR, NZ * NR))


def surface_matrix(zg):
    """diag carrying the face area 2 pi r dr on the k = 0 layer: the interface at z = 0."""
    d = np.zeros((len(zg), NR))
    d[0, :] = 2 * np.pi * rr * H
    return sp.diags(d.reshape(-1))


def vnuc(zg, positions):
    """-1/|x - x_a| point-sampled at cell centres. No softening is applied: each nucleus
    sits on a grid face/edge and never coincides with a sample point."""
    V = np.zeros((len(zg), NR))
    for z0 in positions:
        V -= 1.0 / np.sqrt(rr[None, :] ** 2 + (zg[:, None] - z0) ** 2)
    return V


def ground_state(A, M, Vcell, vol, S=None, beta=0.0, sigma=-4.0):
    Hm = 0.5 * A + sp.diags((Vcell * vol).reshape(-1))
    if S is not None and beta != 0.0:
        Hm = Hm + 0.5 * beta * S
    w, v = spl.eigsh(Hm.tocsc(), k=1, M=M.tocsc(), sigma=sigma, which='LM', maxiter=8000)
    u = np.abs(v[:, 0])
    u /= np.sqrt(float(u @ (M @ u)))
    return float(w[0]), u


def hydrogen():
    zg = zgrid(-ZMAX, ZMAX)
    vol = volumes(zg); M = sp.diags(vol.reshape(-1))
    A = stiffness(zg, dirichlet_z=True)
    V = vnuc(zg, [0.0])
    mu, u = ground_state(A, M, V, vol)
    T = 0.5 * float(u @ (A @ u))
    Ene = float((V * (u.reshape(len(zg), NR) ** 2) * vol).sum())
    return T + Ene, mu


def h2(R, beta, verbose=False):
    assert abs(round(R / (2 * H)) - R / (2 * H)) < 1e-9, "R must be an even multiple of H"
    zg = zgrid(0.0, ZMAX)                     # electron 1 occupies z >= 0
    zf = zgrid(-ZMAX, ZMAX)
    nz, nzf = len(zg), len(zf)
    k0 = nzf // 2                             # zf[k0] is the first cell with z > 0
    vol = volumes(zg); volf = volumes(zf)
    M = sp.diags(vol.reshape(-1)); Mf = sp.diags(volf.reshape(-1))

    A  = stiffness(zg, dirichlet_z=False)     # z = 0 left natural: that is the interface
    Af = stiffness(zf, dirichlet_z=True).tocsc()
    solve_poisson = spl.factorized(Af)

    # Monopole boundary term for the Hartree solve.  The walls in Af impose V = 0, but the
    # true potential of a unit charge is Q/|x| there -- about 1/8 Ha for this box. That error
    # does NOT cancel in E_bind, because hydrogen has no Hartree term at all, so leaving it
    # out lowers E(H2) alone and inflates the binding. Each Dirichlet face contributes its
    # stiffness weight times the prescribed ghost value to the right-hand side.
    bdry = np.zeros((nzf, NR))
    bdry[:, NR - 1] += (2 * np.pi * (rr[NR - 1] + 0.5 * H)) / np.sqrt(
        (rr[NR - 1] + H) ** 2 + zf ** 2)
    bdry[nzf - 1, :] += (2 * np.pi * rr) / np.sqrt(rr ** 2 + (zf[nzf - 1] + H) ** 2)
    bdry[0, :] += (2 * np.pi * rr) / np.sqrt(rr ** 2 + (zf[0] - H) ** 2)
    bdry = bdry.reshape(-1)
    S  = surface_matrix(zg)
    Vne = vnuc(zg, [+R / 2, -R / 2])

    u = np.exp(-np.sqrt(rr[None, :] ** 2 + (zg[:, None] - R / 2) ** 2)).reshape(-1)
    u /= np.sqrt(float(u @ (M @ u)))
    rho = (u ** 2).reshape(nz, NR)
    VH = np.zeros((nz, NR)); d = 1.0; it = 0

    for it in range(60):
        rho2 = np.zeros((nzf, NR))
        rho2[k0 - 1::-1, :] = rho[:k0, :]     # exact mirror, no shared layer
        rho2 /= float((rho2 * volf).sum())
        VH = solve_poisson(4 * np.pi * (Mf @ rho2.reshape(-1)) + bdry
                           ).reshape(nzf, NR)[k0:k0 + nz, :]

        mu, u = ground_state(A, M, Vne + VH, vol, S=S, beta=beta)
        rnew = (u ** 2).reshape(nz, NR)
        d = float(np.abs(rnew - rho).max() / max(rnew.max(), 1e-30))
        rho = 0.3 * rnew + 0.7 * rho
        rho /= float((rho * vol).sum())
        if d < 1e-8:
            break

    u = np.sqrt(rho).reshape(-1)
    u /= np.sqrt(float(u @ (M @ u)))
    rho = (u ** 2).reshape(nz, NR)
    T = 0.5 * float(u @ (A @ u))
    Ene = float((Vne * rho * vol).sum())
    Eee = float((VH * rho * vol).sum())
    Esurf = beta * float(u @ (S @ u))          # 2 * (beta/2) * Int u^2 dS
    Etot = 2 * (T + Ene) + Eee + 1.0 / R
    if verbose:
        print(f"        T={T:+.5f} Vne={Ene:+.5f} Vee={Eee:+.5f} Vnn={1/R:+.5f} "
              f"Esurf={Esurf:+.5f} mu={mu:+.4f} it={it+1} d={d:.1e}", flush=True)
    return Etot, Etot + Esurf


if __name__ == '__main__':
    R = float(sys.argv[1]) if len(sys.argv) > 1 else 1.40
    verb = '-v' in sys.argv
    print(f"grid h={H} RMAX={RMAX} ZMAX={ZMAX} "
          f"({NR} radial x {int(2*ZMAX/H)} axial)", flush=True)

    EH, muH = hydrogen()
    err = EH + 0.5
    print(f"\nVALIDATION  E(H) = {EH:+.6f} Ha   exact -0.500000   error {err:+.6f}"
          f"   mu = {muH:+.5f}", flush=True)
    if abs(err) > 0.02:
        print("\n*** GATE FAILED: E(H) is not within 0.02 Ha of exact.", flush=True)
        print("*** Nothing below can be trusted; the grid does not resolve the cusp.\n",
              flush=True)
    else:
        print("*** gate passed\n", flush=True)

    print(f"R = {R} a0     E_bind = 2 E(H) - E(H2);  positive = bound\n", flush=True)
    print(f"{'beta':>6} {'a=1.146/b':>10} {'E(H2)':>11} {'E_bind':>10} {'eV':>8}"
          f" {'+surf':>10} {'eV':>8}", flush=True)
    print('-' * 70, flush=True)
    for b in [0.0, 0.1, 0.2, 0.4, 0.8, 1.146, 2.0]:
        E, Es = h2(R, b, verbose=verb)
        bind, binds = 2 * EH - E, 2 * EH - Es
        acrit = ('%10.2f' % (1.146 / b)) if b > 0 else '       inf'
        print(f"{b:6.3f} {acrit} {E:11.5f} {bind:10.5f} {bind*27.2114:8.3f}"
              f" {binds:10.5f} {binds*27.2114:8.3f}", flush=True)
