"""
Does RealQM bind H-?  The electron affinity of hydrogen, from the 1+1 shell.

The transport question is downstream of this one. If the framework does not bind
an extra electron to H, there is no carrier and nothing can be transported --
which would also explain why an excess electron was expelled from a chain, and
why no configuration would hold one.

RealQM's H- is a +1 kernel with TWO non-overlapping electron domains in a 1+1
radial shell: a tight inner electron on 0 < r < R and a diffuse outer one on
r > R, with R a FREE BOUNDARY found by minimising the energy. Each domain carries
exactly one unit of charge and each meets the interface with the free (Neumann)
condition psi'(R) = 0, which is what the solver uses between like domains.

Spherical symmetry makes this a radial problem, so it can be done properly:

    -1/2 u_i'' - u_i / r + V_j(r) u_i = E_i u_i ,   u = r psi ,
    V_j = Hartree potential of the OTHER electron (no self-interaction).

Measured: E(H) = -0.5 Ha, E(H-) = -0.5277 Ha, affinity 0.0277 Ha = 0.754 eV.

usage: python3 hminus_radial.py [--N 4000] [--Rmax 40]
"""
import argparse

import numpy as np

HA_EV = 27.211386245988


def solve_domain(r, dr, V, lo_idx, hi_idx, bc_lo, bc_hi):
    """Lowest eigenvalue of -1/2 u'' + V u = E u on [lo_idx, hi_idx].

    bc = 'zero'      u = 0 at that end (origin, or infinity)
       = 'neumann'   psi'(r) = 0, i.e. u' = u/r  (the free interface)
    Finite differences plus bisection on the node count.
    """
    n = hi_idx - lo_idx + 1
    if n < 8:
        return np.inf, None

    def shoot(E):
        u = np.zeros(n)
        # start from the low end
        if bc_lo == 'zero':
            u[0], u[1] = 0.0, 1e-8
        else:                                   # Neumann at the low end
            u[0] = 1.0
            u[1] = u[0] * (1.0 + dr / r[lo_idx])
        nodes = 0
        for i in range(1, n - 1):
            k = 2.0 * (V[lo_idx + i] - E)
            u[i + 1] = 2.0 * u[i] - u[i - 1] + dr * dr * k * u[i]
            if u[i + 1] * u[i] < 0.0:
                nodes += 1
        if bc_hi == 'zero':
            resid = u[-1]
        else:                                   # Neumann at the high end
            resid = (u[-1] - u[-2]) / dr - u[-1] / r[hi_idx]
        return nodes, resid, u

    lo, hi = -2.0, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        nodes, resid, _ = shoot(mid)
        if nodes > 0:
            hi = mid
        else:
            _, r_lo, _ = shoot(lo)
            if r_lo * resid < 0.0:
                hi = mid
            else:
                lo = mid
    E = 0.5 * (lo + hi)
    _, _, u = shoot(E)
    nrm = np.sqrt(np.sum(u * u) * dr)
    if nrm <= 0 or not np.isfinite(nrm):
        return np.inf, None
    return E, u / nrm


def hartree_from(u, r, dr, lo, hi):
    """Potential of a unit spherical charge with radial density u^2 on [lo, hi]."""
    V = np.zeros_like(r)
    dens = np.zeros_like(r)
    dens[lo:hi + 1] = u * u                       # u^2 dr is the charge element
    q_in = np.cumsum(dens) * dr                   # charge inside r
    tail = np.cumsum((dens / np.maximum(r, 1e-12))[::-1])[::-1] * dr
    V = q_in / np.maximum(r, 1e-12) + tail
    return V


def energy_at(R, r, dr, N, iters=60):
    """Total energy of H- with the interface at radius R."""
    iR = int(np.searchsorted(r, R))
    if iR < 20 or iR > N - 60:
        return np.inf
    Vn = -1.0 / np.maximum(r, 1e-12)

    # start with no mutual repulsion
    V2 = np.zeros_like(r)
    V1 = np.zeros_like(r)
    u1 = u2 = None
    for _ in range(iters):
        E1, u1n = solve_domain(r, dr, Vn + V2, 0, iR, 'zero', 'neumann')
        if u1n is None:
            return np.inf
        E2, u2n = solve_domain(r, dr, Vn + V1, iR, N - 1, 'neumann', 'zero')
        if u2n is None:
            return np.inf
        u1, u2 = u1n, u2n
        V1n = hartree_from(u1, r, dr, 0, iR)
        V2n = hartree_from(u2, r, dr, iR, N - 1)
        V1, V2 = 0.5 * V1 + 0.5 * V1n, 0.5 * V2 + 0.5 * V2n     # damped mixing

    # total energy: sum of eigenvalues minus the double-counted repulsion
    Erep = np.sum(u1 * u1 * V2[0:iR + 1]) * dr
    return E1 + E2 - Erep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=4000)
    ap.add_argument('--Rmax', type=float, default=40.0)
    args = ap.parse_args()

    r = np.linspace(args.Rmax / args.N, args.Rmax, args.N)
    dr = r[1] - r[0]

    # reference: hydrogen itself, one electron, whole space
    Vn = -1.0 / np.maximum(r, 1e-12)
    EH, _ = solve_domain(r, dr, Vn, 0, args.N - 1, 'zero', 'zero')
    print(f"reference  E(H)  = {EH:.5f} Ha      (exact -0.5)")
    print(f"measured   E(H-) = -0.52770 Ha      affinity 0.0277 Ha = 0.754 eV\n")

    print(f"{'R (a0)':>8} {'E(H-) (Ha)':>13} {'vs E(H)':>12} {'affinity (eV)':>15}")
    best = (np.inf, None)
    for R in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0):
        E = energy_at(R, r, dr, args.N)
        if not np.isfinite(E):
            print(f"{R:>8.2f} {'--':>13}")
            continue
        aff = (EH - E) * HA_EV
        print(f"{R:>8.2f} {E:>13.5f} {E - EH:>12.5f} {aff:>15.3f}")
        if E < best[0]:
            best = (E, R)
    if best[1] is not None:
        aff = (EH - best[0]) * HA_EV
        print(f"\nbest interface R = {best[1]:.2f} a0,  E(H-) = {best[0]:.5f} Ha")
        print(f"electron affinity = {aff:.3f} eV   (measured 0.754 eV)")
        print("\n" + ("BOUND: RealQM holds the extra electron, so a carrier exists."
                      if aff > 0 else
                      "UNBOUND: the framework does not hold an extra electron on hydrogen.\n"
                      "Then there is no carrier, and no transport question to ask."))


if __name__ == '__main__':
    main()
