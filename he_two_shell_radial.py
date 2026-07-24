"""
Orthohelium (He 2^3S) two-shell eigenvalues by radial self-consistent field.
=============================================================================
Two electrons around a +2 kernel: inner 1s, outer 2s. Each electron solves a
radial Schrodinger equation in the TOTAL Coulomb potential = kernel (-2/r) +
the Hartree field of the OTHER electron (no self-repulsion). Self-consistent.

Gives the two per-shell eigenvalues E_1 (1s) and E_2 (2s), hence
  BREATHING (para)  = |E_1 - E_2|          (inner<->outer, ground not involved)
  SLOSHING  (ortho) = E_tot(2^3S) - E(1s^2)  (uses known totals; ~19.8 eV)
Units: Hartree atomic (hbar=m=e=1).
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal

Ha_eV = 27.211386

# radial grid u(r)=r R(r) on (0, Rmax]; u(0)=0, u(Rmax)=0
Rmax, N = 60.0, 6000
r = np.linspace(Rmax / N, Rmax, N)
h = r[1] - r[0]
Z = 2.0

def solve_l0(V, which):
    """Lowest (which=0) or 2nd (which=1) l=0 eigenpair of -1/2 u'' + V u = E u."""
    diag = 1.0 / h**2 + V
    off = -0.5 / h**2 * np.ones(N - 1)
    E, U = eigh_tridiagonal(diag, off, select='i', select_range=(which, which))
    u = U[:, 0]
    u /= np.sqrt(np.sum(u**2) * h)          # normalize: int u^2 dr = 1
    return E[0], u

def hartree(u):
    """V_H(r) from radial density u^2 (one electron): enclosed/r + outside tail."""
    dens = u**2                              # int dens dr = 1
    cum = np.cumsum(dens) * h                # charge within r
    tail = np.cumsum((dens / r)[::-1])[::-1] * h  # int_r^inf dens/r' dr'
    return cum / r + tail

# --- initial hydrogenic guesses ---
Vbare = -Z / r
e1, u1 = solve_l0(Vbare, 0)                  # 1s
e2, u2 = solve_l0(Vbare + 0.75 / r, 1)       # 2s (screened start)

# --- SCF ---
for it in range(200):
    VH_from_2 = hartree(u2)                  # field of outer felt by inner
    VH_from_1 = hartree(u1)                  # field of inner felt by outer
    e1n, u1n = solve_l0(Vbare + VH_from_2, 0)          # 1s in kernel + outer
    e2n, u2n = solve_l0(Vbare + VH_from_1, 1)          # 2s in kernel + inner
    mix = 0.35                                # density mixing for stability
    u1 = np.sqrt((1 - mix) * u1**2 + mix * u1n**2); u1 /= np.sqrt(np.sum(u1**2) * h)
    u2 = np.sqrt((1 - mix) * u2**2 + mix * u2n**2); u2 /= np.sqrt(np.sum(u2**2) * h)
    if abs(e1n - e1) < 1e-8 and abs(e2n - e2) < 1e-8:
        e1, e2 = e1n, e2n
        break
    e1, e2 = e1n, e2n

# --- total energy: E = e1 + e2 - J12  (subtract double-counted Hartree) ---
J12 = np.sum(u1**2 * hartree(u2)) * h
E_tot = e1 + e2 - J12

# --- ground state 1s^2 (both electrons in 1s) at the SAME radial Hartree level ---
ug = np.exp(-2 * r); ug /= np.sqrt(np.sum(ug**2) * h)
eg = 0.0
for itg in range(300):
    egn, ugn = solve_l0(Vbare + hartree(ug), 0)   # 1s in kernel + the other 1s electron
    ug = np.sqrt(0.65 * ug**2 + 0.35 * ugn**2); ug /= np.sqrt(np.sum(ug**2) * h)
    if abs(egn - eg) < 1e-9:
        eg = egn; break
    eg = egn
Jg = np.sum(ug**2 * hartree(ug)) * h          # 1s-1s repulsion
E_ground_c = 2 * eg - Jg                       # computed ground total, same level as E_tot

# --- expectation-value cross-check of each orbital energy ---
def orb_energy(u, Vother):
    T = 0.5 * np.sum(np.gradient(u, h)**2) * h
    V = np.sum(u**2 * (Vbare + Vother)) * h
    return T + V

print(f"SCF iterations: {it}")
print(f"  inner 1s eigenvalue  E_1 = {e1:+.4f} Ha = {e1*Ha_eV:+.2f} eV")
print(f"  outer 2s eigenvalue  E_2 = {e2:+.4f} Ha = {e2*Ha_eV:+.2f} eV  (observed outer binding -0.175)")
print(f"  1s <r> = {np.sum(u1**2*r)*h:.3f} a0    2s <r> = {np.sum(u2**2*r)*h:.3f} a0")
print(f"  total E (Hartree-level) = {E_tot:+.4f} Ha  (observed 2^3S -2.1748)")
print()
gap = abs(e1 - e2)
slosh_c = E_tot - E_ground_c        # COMPUTED sloshing: both totals at radial Hartree level
slosh_obs = -2.1748 - (-2.9037)     # observed totals, for comparison
print(f"  ground 1s^2 (radial Hartree): E = {E_ground_c:+.4f} Ha  (exact -2.9037; Hartree omits correlation)")
print()
print(f"  BREATHING / para  = |E_1 - E_2|        = {gap:.4f} Ha = {gap*Ha_eV:.2f} eV   (two shell rates, computed)")
print(f"  SLOSHING  / ortho = E(1s2s) - E(1s^2)  = {slosh_c:.4f} Ha = {slosh_c*Ha_eV:.2f} eV   (both computed, radial Hartree)")
print(f"                      observed totals      = {slosh_obs:.4f} Ha = {slosh_obs*Ha_eV:.2f} eV   (NIST 2^3S excitation 19.82)")
print(f"  ratio breathing/sloshing = {gap/slosh_c:.2f}")
