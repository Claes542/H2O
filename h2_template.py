#!/usr/bin/env python3
"""
A literal transcription of the p5 template, runnable from the command line.

This exists for arbitration. h2_p5_original.html and essence_solver.html both produce
correct energies and molecule.js does not: at R=6 the exact V_ee is 1/R = 0.16667, the
direct sum alone gives it, and every relaxation walks it down (p5 gradient step 0.0712,
damped Jacobi 0.042). Debugging that inside the GPU solver costs a browser run per
hypothesis. Here the algorithm can be checked directly against answers that are known
exactly -- E(H) = -0.5 and V_ee = 1/R -- before anything is ported back.

The template advances BOTH fields by the SAME gradient step with the SAME dt:

    P[m] += dt * ( lap(P[m])/h^2 + 2*pi*rho_other )
    u[m] += 0.5*d*lap(u[m]) + dt*(K - 2*P[m])*u[m]

P is HALF the Coulomb potential: lap(P) = -2*pi*rho gives P = 0.5/r for a unit charge,
which is why the wavefunction equation carries the factor 2 on P. Summing Int rho_m * P_m
over BOTH electrons counts each pair twice and recovers V_ee = 1/R.

    python3 h2_template.py H         -> hydrogen atom, exact -0.5
    python3 h2_template.py H2 6.0    -> H2 at R=6, exact V_ee = 1/6 = 0.16667
"""
import os
import sys

import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else 'H'
R = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
BOX = float(os.environ.get('BOX', 16.0))
N = int(os.environ.get('N', 48))
STEPS = int(os.environ.get('STEPS', 3000))
DV = float(os.environ.get('DV', 0.12))          # dt = DV*h^2, stable below 1/6
# Offset of the dividing plane from the midpoint, in a0. This models a free boundary that
# has drifted: SHIFT > 0 gives electron 1 more of the box than electron 0. molecule.js
# evolves its boundary freely and at beta=0, curv=0 there is no confinement cost opposing
# a drift (the C=0 defect), so the question is how far V_ee falls when the split is lopsided.
SHIFT = float(os.environ.get('SHIFT', 0.0))
# PBC=0 pins P to zero on the box wall instead of seeding it with the exact monopole tail
# 0.5/r. This is what a ping-pong pair whose scratch half was never seeded would do: the
# zero boundary is read back in on alternate sweeps and propagates inward. The predicted
# cost is 0.5/L per electron, so at L=8 with two electrons V_ee should fall by ~0.125.
PBC = os.environ.get('PBC', '1')
# RCUT models molecule.js's Voronoi labelling, where the trial function is cut off at
# initRcut: beyond that radius from BOTH nuclei no atom wins and the cells fall to the
# default label. Electron 1 then owns only its near field while electron 0 inherits all
# the far field. RCUT=0 disables (clean half-space split).
RCUT = float(os.environ.get('RCUT', 0.0))
# KERNEL=clamp reproduces molecule.js's nuclear kernel: Z/max(|r|, 2h) -- a hard clamp two
# grid spacings from the nucleus (R_SING = 2*hGrid) rather than the sqrt(r^2+h^2) softening
# used above. It gives a far better single atom (molecule.js: E(H) = -0.49861), which is
# exactly why it needs checking across TWO domains: a kernel tuned on one atom need not
# carry over, and E(H2) - 2E(H) is where any discrepancy would show.
KERNEL = os.environ.get('KERNEL', 'soft')
# BETA: the Robin surface tension. The energy carries (beta/2) * Int psi^2 dS on every face
# of a domain that borders a non-domain cell, whose natural boundary condition is
# dn(psi) + beta*psi = 0. It supplies the confinement energy a free interface otherwise
# lacks (the C=0 defect). molecule.js implements the same pair: an extra -beta*nface/h in
# the psi update, and 0.5*beta*psi^2*nface*h^2 in the energy sum.
BETA = float(os.environ.get('BETA', 0.0))

S = N + 1
h = BOX / N
h2, h3 = h * h, h ** 3
dt, half_d = DV * h2, 0.5 * DV
TWO_PI = 2.0 * np.pi

ax = np.arange(S) * h - BOX / 2.0
X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')

nuc = ([(0.0, 0.0, -R / 2, 1.0), (0.0, 0.0, +R / 2, 1.0)] if MODE == 'H2'
       else [(0.0, 0.0, 0.0, 1.0)])
NE = len(nuc)

# nuclear attraction, softened at the nucleus by sqrt(r^2 + h^2) as the GPU code does
K = np.zeros((S, S, S))
for (nx, ny, nz, nZ) in nuc:
    _r = np.sqrt((X - nx) ** 2 + (Y - ny) ** 2 + (Z - nz) ** 2)
    if KERNEL == 'clamp':
        K += nZ / np.maximum(_r, 2.0 * h)
    else:
        K += nZ / np.sqrt(_r ** 2 + h2)

# Domains: one electron per nucleus, split at the midplane z = 0 for H2 -- the hard-Neumann
# midplane of the essence solver. The outermost shell is excluded from every domain, so the
# box wall is a zero-flux edge where psi has already decayed.
interior = np.zeros((S, S, S), dtype=bool)
interior[1:-1, 1:-1, 1:-1] = True
own = []
for m in range(NE):
    if NE == 1:
        o = interior.copy()
    else:
        o = interior & ((Z < SHIFT) if m == 0 else (Z >= SHIFT))
        if RCUT > 0.0:
            nx1, ny1, nz1, _ = nuc[1]
            near1 = np.sqrt((X - nx1) ** 2 + (Y - ny1) ** 2 + (Z - nz1) ** 2) < RCUT
            o = (interior & near1 & (Z >= SHIFT)) if m == 1 else (interior & ~(near1 & (Z >= SHIFT)))
    own.append(o)

# psi: a 1s on its own nucleus, normalised over its own domain
u = []
for m in range(NE):
    nx, ny, nz, _ = nuc[m]
    a = np.exp(-np.sqrt((X - nx) ** 2 + (Y - ny) ** 2 + (Z - nz) ** 2 + h2))
    a = np.where(own[m], a, 0.0)
    a /= np.sqrt((a[own[m]] ** 2).sum() * h3)
    u.append(a)

# P[m]: the potential felt by electron m from the OTHERS, seeded with the exact point-charge
# solution 0.5/r over the FULL grid including the boundary. The boundary is never updated,
# so it holds the correct monopole tail instead of 0.
P = []
for m in range(NE):
    a = np.zeros((S, S, S))
    for n in range(NE):
        if n == m:
            continue
        nx, ny, nz, _ = nuc[n]
        a += 0.5 / np.sqrt((X - nx) ** 2 + (Y - ny) ** 2 + (Z - nz) ** 2 + h2)
    if PBC == '0':
        b = np.zeros_like(a)
        b[1:-1, 1:-1, 1:-1] = a[1:-1, 1:-1, 1:-1]
        a = b
    P.append(a)

SHIFTS = [(1, 0), (-1, 0), (1, 1), (-1, 1), (1, 2), (-1, 2)]


def exposed_faces(o):
    """Number of faces of each cell that border a cell outside the domain. This is the
    discrete surface measure: sum(psi^2 * nface) * h^2 approximates Int psi^2 dS."""
    n = np.zeros(o.shape)
    for sh, axis in SHIFTS:
        n += np.where(np.roll(o, sh, axis=axis), 0.0, 1.0)
    return np.where(o, n, 0.0)


nface = [exposed_faces(o) for o in own]


def lap_own(a, o):
    """Zero-flux laplacian: a neighbour outside the domain contributes the centre value,
    which is the reflecting condition at a free interface. The outermost shell is never in
    a domain, so np.roll's wraparound is always masked out."""
    tot = -6.0 * a
    for s, axis in SHIFTS:
        tot += np.where(np.roll(o, s, axis=axis), np.roll(a, s, axis=axis), a)
    return tot


def lap_full(a):
    out = np.zeros_like(a)
    out[1:-1, 1:-1, 1:-1] = (
        a[2:, 1:-1, 1:-1] + a[:-2, 1:-1, 1:-1] +
        a[1:-1, 2:, 1:-1] + a[1:-1, :-2, 1:-1] +
        a[1:-1, 1:-1, 2:] + a[1:-1, 1:-1, :-2] -
        6.0 * a[1:-1, 1:-1, 1:-1])
    return out


for step in range(STEPS):
    for m in range(NE):
        rho_other = np.zeros((S, S, S))
        for n in range(NE):
            if n != m:
                rho_other += np.where(own[n], u[n] ** 2, 0.0)
        # interior only; the boundary stays frozen at the monopole tail
        upd = dt * (lap_full(P[m]) / h2 + TWO_PI * rho_other)
        P[m][1:-1, 1:-1, 1:-1] += upd[1:-1, 1:-1, 1:-1]
    for m in range(NE):
        a = u[m]
        vbeta = BETA * nface[m] / h if BETA != 0.0 else 0.0
        nxt = a + half_d * lap_own(a, own[m]) + dt * (K - 2.0 * P[m] - vbeta) * a
        nxt = np.where(own[m], nxt, 0.0)
        nxt /= np.sqrt((nxt[own[m]] ** 2).sum() * h3)
        u[m] = nxt

T = V_eK = V_ee = 0.0
for m in range(NE):
    o = own[m]
    T += float((-0.5 * u[m] * lap_own(u[m], o) / h2)[o].sum()) * h3
    V_eK += float((-K * u[m] ** 2)[o].sum()) * h3
    V_ee += float((P[m] * u[m] ** 2)[o].sum()) * h3
S_beta = 0.0
if BETA != 0.0:
    for m in range(NE):
        S_beta += 0.5 * BETA * float((u[m] ** 2 * nface[m])[own[m]].sum()) * h2
V_KK = (1.0 / R) if MODE == 'H2' else 0.0
E = T + V_eK + V_ee + V_KK + S_beta

tag = f" R={R:.2f}" if MODE == 'H2' else ""
print(f"{MODE}{tag}  box={BOX} N={N} h={h:.4f} steps={STEPS}")
print(f"  T     {T:+.5f}")
print(f"  V_eK  {V_eK:+.5f}")
if MODE == 'H2':
    print(f"  V_ee  {V_ee:+.5f}   exact 1/R = {1/R:.5f}   ratio {V_ee/(1/R):.3f}")
else:
    print(f"  V_ee  {V_ee:+.5f}")
print(f"  V_KK  {V_KK:+.5f}")
if BETA != 0.0:
    print(f"  S_beta{S_beta:+.5f}   (beta = {BETA})")
print(f"  E     {E:+.5f}" + ("   exact -0.50000" if MODE == 'H' else ""))
