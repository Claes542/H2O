"""
Does one interface parameter serve atoms as well as the electron gas?

The uniform gas needs a soft (Robin) interface, psi' + beta*psi = 0, with the
single dimensionless value beta*a = 1.146 reproducing Thomas-Fermi exactly. The
framework currently uses beta = 0 (free plane), which is the C = 0 case. The
question that decides whether beta is a constant of the theory or a per-system
fit is whether atoms tolerate the gas's value.

An earlier attempt at this failed on numerics, not physics: it used a fixed step
count while dt scales as h^2, so finer meshes received less relaxation and the
sensitivity reversed sign. Two fixes here:

  1. equal relaxation TIME across meshes -- steps scale as 1/h^2, not fixed;
  2. the nuclear cusp treated by softening at a FIXED physical length rather than
     at the grid scale, so refining the mesh does not change the potential.

Test system: helium, the smallest case with an electron-electron interface. Two
electrons on half-spaces meeting at a plane through the nucleus, no
self-interaction, Coulomb repulsion between the domains. Exact: -2.9037 Ha.

The interface condition at z = 0, outward normal -z:

    dpsi/dz = beta * psi      beta = 0 : Neumann, the present choice
                              beta -> inf : Dirichlet, a node

usage: python3 robin_test.py [--N 80] [--L 12] [--time 40] [--betas 0,0.5,1.146,2]
"""
import argparse

import numpy as np


def solve(N, L, beta, tau_total, eps_phys=0.15, Z=2.0):
    """Relax helium's z>0 domain for a fixed imaginary TIME tau_total."""
    h = L / N
    ax = (np.arange(N) - N // 2 + 0.5) * h          # cell centres, plane at z=0
    X, Y, Zc = np.meshgrid(ax, ax, ax, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Zc**2 + eps_phys**2)  # FIXED physical softening
    UP = Zc > 0

    k = 2 * np.pi * np.fft.fftfreq(N, d=h)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0

    def hartree(rho):
        pk = 4 * np.pi * np.fft.fftn(rho) / K2
        pk[0, 0, 0] = 0.0
        return np.real(np.fft.ifftn(pk))

    k0 = N // 2                                      # first live layer above the plane

    def lap(psi):
        out = -6.0 * psi
        for axis, sh in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
            out = out + np.roll(psi, sh, axis=axis)
        # interface: replace the wrapped-in value below k0 by the ghost the BC implies.
        # psi'(0) = beta*psi  =>  psi_ghost = psi(k0) * (1 - beta*h)
        wrapped = np.roll(psi, 1, axis=2)[:, :, k0]
        out[:, :, k0] += psi[:, :, k0] * (1.0 - beta * h) - wrapped
        return out / h**2

    Vn = -Z / R
    psi = np.exp(-1.7 * np.sqrt(X**2 + Y**2 + Zc**2)) * UP
    psi /= np.sqrt((psi**2).sum() * h**3)

    dt = 0.2 * h**2
    nsteps = max(200, int(tau_total / dt))           # EQUAL TIME, not equal steps
    for _ in range(nsteps):
        rho_self = psi**2
        Vee = hartree(rho_self[:, :, ::-1])          # field of the mirror electron
        Hpsi = -0.5 * lap(psi) + (Vn + Vee) * psi
        E = (psi * Hpsi).sum() * h**3
        psi = psi - dt * (Hpsi - E * psi)
        psi = np.maximum(psi, 0.0) * UP
        nrm = np.sqrt((psi**2).sum() * h**3)
        if not np.isfinite(nrm) or nrm == 0:
            return float('nan'), nsteps
        psi /= nrm

    rho_self = psi**2
    Vee = hartree(rho_self[:, :, ::-1])
    T = 0.5 * ((np.gradient(psi, h, axis=0)**2
                + np.gradient(psi, h, axis=1)**2
                + np.gradient(psi, h, axis=2)**2) * UP).sum() * h**3
    Vnuc = (Vn * rho_self).sum() * h**3
    Eee = (Vee * rho_self).sum() * h**3              # counted once for the pair
    return 2 * (T + Vnuc) + Eee, nsteps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=72)
    ap.add_argument('--L', type=float, default=12.0)
    ap.add_argument('--time', type=float, default=40.0, help='imaginary time per run')
    ap.add_argument('--betas', default='0,0.5,1.146,2,20')
    ap.add_argument('--meshes', default='')
    args = ap.parse_args()
    betas = [float(b) for b in args.betas.split(',')]

    if args.meshes:
        print("Mesh convergence at fixed imaginary time (the check the last attempt failed)\n")
        print(f"{'N':>5} {'h':>7} " + " ".join(f"beta={b:<8g}" for b in betas))
        for N in [int(x) for x in args.meshes.split(',')]:
            row = []
            for b in betas:
                E, ns = solve(N, args.L, b, args.time)
                row.append(E)
            print(f"{N:>5} {args.L/N:>7.3f} " + " ".join(f"{v:>12.4f}" for v in row), flush=True)
        return

    print(f"Helium, two half-space domains. N={args.N}, L={args.L}, "
          f"imaginary time {args.time} (steps scale as 1/h^2)")
    print("exact -2.9037 Ha; the gas requires beta*a = 1.146\n")
    print(f"{'beta':>8} {'E (Ha)':>12} {'shift vs Neumann':>18} {'steps':>8}")
    base = None
    for b in betas:
        E, ns = solve(args.N, args.L, b, args.time)
        if base is None:
            base = E
        print(f"{b:>8.3f} {E:>12.4f} {E - base:>18.4f} {ns:>8}", flush=True)
    print("\nA small shift at beta = 1.146 means atoms TOLERATE the gas's value, so one")
    print("number serves both and the framework gains solids. A large shift means bound")
    print("and free matter need different interface conditions.")


if __name__ == '__main__':
    main()
