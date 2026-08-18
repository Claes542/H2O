"""
Time-dependent RealQM on a chain: does charge flow under a field, or only polarise?

The static solver cannot answer this. Domain ownership there is assigned by kernel
proximity, so an electron cannot change owner; and more fundamentally an energy
minimisation returns the equilibrium partition, not a current. Nuclei move in that
scheme and electrons relax, which is why Grotthuss conduction works (the carrier is
a nucleus) and why the electron chains returned nothing.

The time-dependent formulation is where the question belongs: a real density
carrying a complex current, with free boundaries that move. This is a 1D chain
version of realqm_freeboundary.py -- n nuclei, n electron domains separated by
moving interfaces, Neumann at each interface, evolved in real time with a uniform
applied field.

The diagnostic is how far the interfaces travel, measured in lattice spacings:

    displacement >> a   charge is moving through the lattice   -- metallic
    displacement << a   domains distort and stop               -- insulating

A finite chain polarises and saturates either way, so the magnitude is the signal,
not the eventual halt.

BASELINE FIRST. With --baseline the script drops to one nucleus and two electrons,
the case realqm_freeboundary.py treats, and reports the ground-state energy and the
symmetric interface position. If that is wrong, nothing downstream is worth reading.

STATUS (2026-08-18): BASELINE FAILS. Do not use the chain mode.

The --baseline case (one nucleus Z=2, two domains) relaxes to E = +0.76 Ha --
unbound, where a two-electron system in that well must be negative. It is
converged, not under-relaxed: 20k, 60k and 120k steps all give +0.7623.
Normalisation is exact (domain charges 1.0000) and the interface sits on the
nucleus by symmetry, so the geometry and the constraint machinery work; it is the
bound state that does not form. The electrons spread over the half-line instead of
localising: a soft nucleus at eps=0.6 reaches only -3.3 over ~1 a0, while each
electron is smeared over 20 a0, so the attraction it feels is weak against the
electron-electron repulsion.

Unresolved whether that is a defect of this setup (box too large at L=40, eps too
soft, cusped initial guess) or the same interface effect that makes C = 0 in the
uniform gas, now showing up in a bound-state problem. Either way the chain mode
cannot be read until the baseline binds.

Kept because the scheme is the right one for the conduction question -- real-time
evolution with moving free boundaries is what the static solver lacks -- and
because the baseline check did its job.

usage: python3 chain_td.py [--baseline] [--n 4] [--a 2.0] [--field 0.02]
"""
import argparse

import numpy as np


def build(args):
    N, L = args.N, args.L
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    if args.baseline:
        xs = np.array([L / 2])
        Z = np.array([2.0])
        ne = 2
    else:
        span = (args.n - 1) * args.a
        xs = L / 2 + (np.arange(args.n) - (args.n - 1) / 2) * args.a
        Z = np.ones(args.n)
        ne = args.n
    Vnuc = np.zeros(N)
    for xi, zi in zip(xs, Z):
        Vnuc -= zi / np.sqrt((x - xi) ** 2 + args.eps ** 2)
    return x, dx, Vnuc, xs, ne


def coul_kernel(x, dx, eps):
    ker = 1.0 / np.sqrt(x ** 2 + eps ** 2)
    return np.fft.fft(np.fft.ifftshift(ker)), dx


def hartree(nd, kerF, dx):
    return np.real(np.fft.ifft(np.fft.fft(nd) * kerF)) * dx


def masks_from(bnds, N):
    """bnds: interior interface indices, ascending. Returns one mask per domain."""
    edges = [0] + list(bnds) + [N - 1]
    ms = []
    for d in range(len(edges) - 1):
        m = np.zeros(N)
        lo = edges[d] if d == 0 else edges[d] + 1
        m[lo:edges[d + 1] + 1] = 1.0
        ms.append(m)
    return ms


def lap_neumann(u, m, dx):
    """u'' with zero flux across the domain's own faces, Dirichlet at the box ends."""
    up = np.empty_like(u); um = np.empty_like(u)
    up[:-1] = np.where(m[1:] > 0.5, u[1:], u[:-1]); up[-1] = 0.0
    um[1:] = np.where(m[:-1] > 0.5, u[:-1], u[1:]); um[0] = 0.0
    return (up + um - 2 * u) / dx ** 2


def normalise(u, m, dx):
    s = np.sum(np.abs(u) ** 2 * m) * dx
    return u / np.sqrt(s) if s > 0 else u


def relax(psis, ms, Vnuc, kerF, dx, steps, dt):
    """Imaginary time to the ground state, no self-interaction."""
    for _ in range(steps):
        dens = [np.abs(p) ** 2 * m for p, m in zip(psis, ms)]
        tot = sum(dens)
        for i, (p, m) in enumerate(zip(psis, ms)):
            Vee = hartree(tot - dens[i], kerF, dx)
            H = -0.5 * lap_neumann(p, m, dx) + (Vnuc + Vee) * p
            p = p - dt * H
            psis[i] = normalise(np.maximum(p, 0.0) * m, m, dx)
    return psis


def interface_positions(psis, ms, x, dx, bnds):
    """Where each interior interface sits, by density balance of its two neighbours."""
    return [x[b] for b in bnds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=800)
    ap.add_argument('--L', type=float, default=40.0)
    ap.add_argument('--eps', type=float, default=0.6)
    ap.add_argument('--n', type=int, default=4, help='number of nuclei')
    ap.add_argument('--a', type=float, default=2.0, help='lattice spacing (a0)')
    ap.add_argument('--field', type=float, default=0.02, help='uniform field (a.u.)')
    ap.add_argument('--relax', type=int, default=4000)
    ap.add_argument('--T', type=float, default=40.0, help='real-time duration')
    ap.add_argument('--baseline', action='store_true')
    args = ap.parse_args()

    x, dx, Vnuc, xs, ne = build(args)
    kerF, _ = coul_kernel(x, dx, args.eps)

    # interfaces start midway between nuclei; for the baseline, at the nucleus
    if args.baseline:
        bnds = [int(round(xs[0] / dx))]
    else:
        mids = 0.5 * (xs[:-1] + xs[1:])
        bnds = [int(round(m / dx)) for m in mids]
    ms = masks_from(bnds, args.N)

    psis = []
    for i, m in enumerate(ms):
        c = xs[min(i, len(xs) - 1)]
        p = np.exp(-np.abs(x - c)) * m
        psis.append(normalise(p, m, dx))

    dt = 0.2 * dx ** 2
    psis = relax(psis, ms, Vnuc, kerF, dx, args.relax, dt)

    dens = [np.abs(p) ** 2 * m for p, m in zip(psis, ms)]
    tot = sum(dens)
    E = 0.0
    for i, (p, m) in enumerate(zip(psis, ms)):
        Vee = hartree(tot - dens[i], kerF, dx)
        T = 0.5 * np.sum(np.gradient(p, dx) ** 2 * m) * dx
        E += T + np.sum((Vnuc + 0.5 * Vee) * dens[i]) * dx

    if args.baseline:
        print(f"BASELINE  1 nucleus Z=2, 2 domains, soft-Coulomb eps={args.eps}")
        print(f"  ground-state energy   E = {E:.4f} Ha")
        print(f"  interface at x = {x[bnds[0]]:.3f} (nucleus at {xs[0]:.3f})")
        print(f"  domain charges: " + "  ".join(f"{np.sum(d)*dx:.4f}" for d in dens))
        print("\n  A soft-Coulomb 1D 'helium' is not the physical atom; what this checks is that")
        print("  the domains stay normalised, the interface sits where symmetry says, and the")
        print("  energy is finite and stable. If those hold the chain run below is worth reading.")
        return

    print(f"CHAIN  n={args.n} nuclei, spacing a={args.a} a0, field={args.field} a.u.")
    print(f"  relaxed energy E = {E:.4f} Ha")
    print(f"  domain charges: " + "  ".join(f"{np.sum(d)*dx:.3f}" for d in dens))
    x0 = np.array(interface_positions(psis, ms, x, dx, bnds))
    print(f"  interfaces at t=0: " + "  ".join(f"{v:.2f}" for v in x0))

    # real time with a uniform field: track the centroid of each domain, which is
    # what moves if charge flows; the interfaces follow it
    Vfield = args.field * (x - x.mean())
    nsteps = int(args.T / (0.4 * dx))
    dtr = args.T / nsteps
    cents0 = np.array([np.sum(x * d) * dx / max(np.sum(d) * dx, 1e-12) for d in dens])
    psis_c = [p.astype(complex) for p in psis]
    for step in range(nsteps):
        dens = [np.abs(p) ** 2 * m for p, m in zip(psis_c, ms)]
        tot = sum(dens)
        for i, (p, m) in enumerate(zip(psis_c, ms)):
            Vee = hartree(tot - dens[i], kerF, dx)
            H = -0.5 * lap_neumann(p, m, dx) + (Vnuc + Vee + Vfield) * p
            psis_c[i] = p - 1j * dtr * H

    dens = [np.abs(p) ** 2 * m for p, m in zip(psis_c, ms)]
    cents = np.array([np.sum(x * d) * dx / max(np.sum(d) * dx, 1e-12) for d in dens])
    shift = cents - cents0
    print(f"  domain centroid shifts after T={args.T}: "
          + "  ".join(f"{v:+.3f}" for v in shift))
    print(f"  mean |shift| = {np.mean(np.abs(shift)):.3f} a0, "
          f"lattice spacing a = {args.a} a0  -> {np.mean(np.abs(shift))/args.a:.2f} a")
    print(f"  charge drift: " + "  ".join(f"{np.sum(d)*dx-1:+.1e}" for d in dens))
    print("\n  >> 1 lattice spacing: charge moving through the lattice, metallic")
    print("  << 1 lattice spacing: domains distort and stop, insulating")
    print("  charge drift far from zero: the scheme is not conserving, read nothing from it")


if __name__ == '__main__':
    main()
