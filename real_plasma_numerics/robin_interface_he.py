"""
Do atoms tolerate the Robin interface that the electron gas requires?

The uniform-gas analysis (realqm_screening_note.md) shows RealQM's Neumann
interface gives zero localisation cost, hence no Fermi pressure, no screening and
no plasmon dispersion; and that a Robin condition psi' + beta psi = 0 with the
single dimensionless value beta*a = 1.146 reproduces Thomas-Fermi exactly.

beta is a parameter, and the framework claims none -- so the question is whether
the SAME condition is compatible with atoms, where the interface currently uses
beta = 0 and works. If atomic energies barely move across the range, the number
is determined once by the gas and used everywhere. If they move strongly, bound
and free matter need different interface conditions, which is a structural
inconsistency rather than a gap.

Test system: helium as RealQM treats it -- two electrons, each a non-negative
density on its own half-space, meeting at a midplane through the nucleus, with
no self-interaction and the Coulomb repulsion taken between the two domains.
By symmetry only the z > 0 electron is solved; the other is its mirror image.

Interface condition at z = 0, outward normal -z:

    dpsi/dn + beta psi = 0   <=>   dpsi/dz = beta psi

    beta = 0        Neumann  (flat at the midplane; RealQM's present choice)
    beta -> inf     Dirichlet (node at the midplane)

Relaxation is imaginary-time gradient flow with the Hartree field of the mirror
density from an FFT Poisson solve. Energies in Hartree; the experimental helium
ground state is -2.9037.

usage: python3 robin_interface_he.py [N] [L]
"""
import sys

import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 72
L = float(sys.argv[2]) if len(sys.argv) > 2 else 14.0
Z = 2.0

h = L / N
ax = (np.arange(N) - N // 2 + 0.5) * h          # cell centres, no point on the plane
X, Y, Zc = np.meshgrid(ax, ax, ax, indexing='ij')
R = np.sqrt(X**2 + Y**2 + Zc**2)
R = np.maximum(R, 0.5 * h)                       # soften only inside one cell
UP = Zc > 0                                      # the z > 0 domain

k = 2 * np.pi * np.fft.fftfreq(N, d=h)
KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2
K2[0, 0, 0] = 1.0


def hartree(rho):
    """Potential of a charge density by FFT Poisson (periodic; box is large)."""
    pk = 4 * np.pi * np.fft.fftn(rho) / K2
    pk[0, 0, 0] = 0.0
    return np.real(np.fft.ifftn(pk))


def laplacian_robin(psi, beta):
    """7-point Laplacian on the z>0 half, with dpsi/dz = beta*psi at the midplane.

    The plane sits between the cells k = N//2-1 and k = N//2, so the first live
    layer is k0 = N//2. Its neighbour below is a ghost value fixed by the
    interface condition: with a one-sided difference across the half-cell gap,
    psi_ghost = psi(k0) * (1 - beta*h).
    """
    k0 = N // 2
    lap = -6.0 * psi
    for ax_, sh in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        lap = lap + np.roll(psi, sh, axis=ax_)
    # correct the layer adjacent to the interface: the rolled-in value from below
    # is whatever wrapped around; replace it by the ghost value
    wrapped = np.roll(psi, 1, axis=2)[:, :, k0]
    ghost = psi[:, :, k0] * (1.0 - beta * h)
    lap[:, :, k0] += (ghost - wrapped)
    return lap / h**2


def solve(beta, steps=4000, dt=None):
    psi = np.exp(-1.7 * R) * UP
    psi /= np.sqrt((psi**2).sum() * h**3)
    dt = dt or 0.25 * h**2
    Vn = -Z / R
    for it in range(steps):
        rho_self = psi**2
        rho_other = rho_self[:, :, ::-1]                 # mirror image in z
        Vee = hartree(rho_other)
        Hpsi = -0.5 * laplacian_robin(psi, beta) + (Vn + Vee) * psi
        E = (psi * Hpsi).sum() * h**3
        psi = psi - dt * (Hpsi - E * psi)
        psi = np.maximum(psi, 0.0) * UP
        nrm = np.sqrt((psi**2).sum() * h**3)
        if not np.isfinite(nrm) or nrm == 0:
            return float('nan'), float('nan')
        psi /= nrm

    rho_self = psi**2
    rho_other = rho_self[:, :, ::-1]
    Vee = hartree(rho_other)
    T = 0.5 * ((np.gradient(psi, h, axis=0)**2
                + np.gradient(psi, h, axis=1)**2
                + np.gradient(psi, h, axis=2)**2) * UP).sum() * h**3
    Vnuc = (Vn * rho_self).sum() * h**3
    Eee = 0.5 * (Vee * rho_self).sum() * h**3 * 2      # both cross terms, counted once
    E_tot = 2 * (T + Vnuc) + Eee
    return E_tot, T


def main():
    print(f"Helium, two half-space domains, N = {N}, L = {L} a0, h = {h:.3f}")
    print("experimental ground state: -2.9037 Ha\n")
    print(f"{'beta (1/a0)':>12} {'E_total (Ha)':>14} {'T per electron':>15} {'shift vs Neumann':>18}")
    base = None
    for beta in (0.0, 0.25, 0.5, 1.146, 2.0, 20.0):
        E, T = solve(beta)
        if base is None:
            base = E
        print(f"{beta:>12.3f} {E:>14.4f} {T:>15.4f} {E - base:>18.4f}")
    print("\nIf the shift across 0 -> 1.146 is small, atoms TOLERATE the value the")
    print("electron gas requires, and one determined number serves both. If it is")
    print("large, bound and free matter need different interface conditions.")


if __name__ == '__main__':
    main()
