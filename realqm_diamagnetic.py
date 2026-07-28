"""
Diamagnetic susceptibility of closed-shell atoms from a RealQM radial SCF.
==========================================================================
RealQM signature (as in he_two_shell_radial.py): every electron moves in the
kernel potential -Z/r PLUS the Hartree field of all the OTHER electrons, with
NO self-repulsion (kinetic energy takes the confining role). No exchange.
This is a radial, spherically-averaged mean-field approximation to RealQM: it
captures the no-self-interaction rule but averages over the non-overlap
(domain) geometry, so it is a FIRST PASS, not the full 3D multiphase solver.

For each closed-shell atom we self-consistently solve, per subshell (n,l),
    -1/2 u'' + [ -Z/r + l(l+1)/(2 r^2) + V_H[n_tot] - V_H[u_(nl)^2] ] u = E u,
build the total density n_tot(r) = sum_s occ_s u_s(r)^2, and read off
    Sum<r^2> = sum_s occ_s * int u_s(r)^2 r^2 dr   (a0^2, since u=rR).

Langevin/Larmor molar diamagnetic susceptibility (Gaussian, cm^3/mol):
    chi_M = -(N_A/6) r_e Sum<r^2>  =  -0.7920e-6 * Sum<r^2>[a0^2]   cm^3/mol.
Compared against experimental molar susceptibilities.
Units: Hartree atomic (hbar=m_e=e=1); lengths a0, energies Ha.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal

Ha_eV = 27.211386
COEF  = 0.79197  # chi_M[1e-6 cm^3/mol] = -COEF * Sum<r^2>[a0^2]

# radial grid u(r)=r R(r) on (0,Rmax]; u(0)=0 built in, u(Rmax)->0
Rmax, N = 35.0, 8000
r = np.linspace(Rmax / N, Rmax, N)
h = r[1] - r[0]

def solve(V, l, node):
    """Eigenpair with `node` radial nodes in the l-channel: -1/2 u'' + (V+cent) u = E u."""
    cent = l * (l + 1) / (2.0 * r * r)
    diag = 1.0 / h**2 + V + cent
    off  = -0.5 / h**2 * np.ones(N - 1)
    E, U = eigh_tridiagonal(diag, off, select='i', select_range=(node, node))
    u = U[:, 0]
    u /= np.sqrt(np.sum(u**2) * h)              # int u^2 dr = 1
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    return E[0], u

def hartree(dens):
    """Monopole Hartree V_H(r) of a radial density dens(r) (int dens dr = q)."""
    cum  = np.cumsum(dens) * h                          # charge within r
    tail = np.cumsum((dens / r)[::-1])[::-1] * h        # int_r^inf dens/r' dr'
    return cum / r + tail

# atom -> list of subshells (n, l, occupancy)
ATOMS = {
    "He": (2.0,  [(1, 0, 2)]),
    "Ne": (10.0, [(1, 0, 2), (2, 0, 2), (2, 1, 6)]),
    "Ar": (18.0, [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 6)]),
}

# experimental molar diamagnetic susceptibilities, 1e-6 cm^3/mol (CRC / Pascal)
CHI_EXP = {"He": -1.9, "Ne": -7.2, "Ar": -19.6}

def run_atom(Z, shells):
    Vbare = -Z / r
    orbs = []                                   # (n,l,occ,node,u)
    for (n, l, occ) in shells:
        node = n - l - 1
        _, u = solve(Vbare + 0.5 * occ / r, l, node)   # screened hydrogenic start
        orbs.append([n, l, occ, node, u])

    for it in range(400):
        ntot = np.zeros(N)
        for (_, _, occ, _, u) in orbs:
            ntot += occ * u**2
        VH_tot = hartree(ntot)
        maxdE = 0.0
        new = []
        for (n, l, occ, node, u) in orbs:
            VH_self = hartree(u**2)                     # remove ONE electron's self-field
            Veff = Vbare + VH_tot - VH_self
            e, un = solve(Veff, l, node)
            new.append([n, l, occ, node, un, e])
        # density mixing
        mix = 0.3
        for i, (n, l, occ, node, un, e) in enumerate(new):
            uold = orbs[i][4]
            um = np.sqrt((1 - mix) * uold**2 + mix * un**2)
            um /= np.sqrt(np.sum(um**2) * h)
            orbs[i][4] = um
        if it > 0:
            maxdE = max(abs(new[i][5] - prevE[i]) for i in range(len(new)))
        prevE = [new[i][5] for i in range(len(new))]
        if it > 2 and maxdE < 1e-6:
            break

    # observables
    sum_r2 = 0.0
    rows = []
    for (n, l, occ, node, u) in orbs:
        r1 = np.sum(u**2 * r) * h
        r2 = np.sum(u**2 * r * r) * h
        sum_r2 += occ * r2
        name = f"{n}{'spdf'[l]}"
        rows.append((name, occ, r1, r2))
    return sum_r2, rows, it

print(f"{'atom':4} {'Sum<r2> (a0^2)':>16} {'chi_calc':>10} {'chi_exp':>9} {'ratio':>7}   (1e-6 cm^3/mol)")
print("-" * 66)
for name, (Z, shells) in ATOMS.items():
    sum_r2, rows, it = run_atom(Z, shells)
    chi_calc = -COEF * sum_r2
    chi_exp  = CHI_EXP[name]
    ratio    = chi_calc / chi_exp
    print(f"{name:4} {sum_r2:16.3f} {chi_calc:10.2f} {chi_exp:9.2f} {ratio:7.2f}   (SCF {it} it)")
    for (sh, occ, r1, r2) in rows:
        print(f"      {sh:>3} x{occ:<2d}  <r>={r1:6.3f}  <r^2>={r2:7.3f}  occ*<r^2>={occ*r2:8.3f}")
    print()
