#!/usr/bin/env python3
"""
The hole in a hydrogen chain, computed exactly: what standard theory says the carrier does.

RealQM's hop barrier is measured on a chain of n cores carrying n-1 electrons -- one hole --
with the nuclei clamped. This computes the same object the ordinary way, and the point of it
is not to produce a competing number but to establish that in standard theory THERE IS NO
BARRIER TO COMPETE WITH.

At clamped nuclei a hole in a periodic chain is a Bloch state: delocalised over every site,
dispersing with a bandwidth 4t, carrying no activation energy at all. A barrier appears only
once the carrier localises, and localisation needs either lattice relaxation (a polaron) or
disorder. The RealQM chain has neither, so the comparison is not 1.9 eV against 0.3 eV -- it
is a barrier against a band.

Three quantities are computed, all in a minimal basis where FCI is exact:

  bandwidth      E(k=pi) - E(k=0) for the hole, from the Hueckel/tight-binding limit of the
                 same chain -- the energy scale a hopping barrier must exceed for hopping to
                 be the mechanism rather than band motion
  hole delocalisation
                 the Mulliken hole population per site in the exact ground state. A hopping
                 carrier sits on one site; a band carrier is spread over all of them. This
                 is the direct test, and it needs no model
  E_corr         FCI - HF, which measures how far the exact state is from a single
                 determinant -- large means the Mott physics RealQM builds in is real here

Usage:  python3 h_chain_hole.py [n] [basis]      default 7 atoms, sto-6g
"""
import sys

import numpy as np
from pyscf import fci, gto, scf

N = int(sys.argv[1]) if len(sys.argv) > 1 else 7
BASIS = sys.argv[2] if len(sys.argv) > 2 else 'sto-6g'
SPACINGS = [3.0, 4.0, 4.5, 6.0]        # bohr; 4.5 is where RealQM's corrugation was measured
HA_EV = 27.211386


def chain(a, n, charge):
    pos = [('H', (0.0, 0.0, (i - (n - 1) / 2) * a)) for i in range(n)]
    nelec = n - charge
    return gto.M(atom=pos, basis=BASIS, unit='Bohr', verbose=0,
                 charge=charge, spin=nelec % 2)


print(f"Hole in a hydrogen chain, n = {N}, basis {BASIS}")
print("H_n+ : n protons, n-1 electrons, nuclei clamped -- the same object RealQM hops.\n")
print(f"{'a':>5} {'HF':>11} {'FCI':>11} {'E_corr':>8} {'bandwidth':>10} "
      f"{'max hole/site':>14} {'verdict':>12}")
print(f"{'(a0)':>5} {'':>11} {'(exact)':>11} {'(eV)':>8} {'4t (eV)':>10} {'(of 1.0)':>14}")
print('-' * 78)

for a in SPACINGS:
    m = chain(a, N, charge=1)
    mf = scf.RHF(m) if (N - 1) % 2 == 0 else scf.ROHF(m)
    mf.kernel()
    ehf = mf.e_tot

    cis = fci.FCI(mf)
    efci, civec = cis.kernel()

    # Hole population per site: 1 - (electrons on that site). With one basis function per
    # atom in a minimal basis, the diagonal of the AO density matrix in a Loewdin-orthogonal
    # basis is the site occupancy, and no population-analysis arbitrariness enters.
    dm1 = cis.make_rdm1(civec, mf.mo_coeff.shape[1], m.nelec)
    dm_ao = mf.mo_coeff @ dm1 @ mf.mo_coeff.T
    s = m.intor('int1e_ovlp')
    w, v = np.linalg.eigh(s)
    # Extended bases on a chain are near linearly dependent -- the smallest overlap
    # eigenvalues run to 1e-8 and below, and 1/sqrt of those overflows. Drop that null space
    # rather than inverting through it; it carries no density.
    keep = w > 1e-7
    w, v = w[keep], v[:, keep]
    s_half = v @ np.diag(np.sqrt(w)) @ v.T
    ao_occ = np.diag(s_half @ dm_ao @ s_half)       # Loewdin per-BASIS-FUNCTION occupancy
    # With more than one function per atom the site occupancy is their sum. In a minimal
    # basis this is the identity and the two definitions agree; in cc-pVDZ they do not, and
    # taking the per-function maximum instead would report a spurious localisation.
    occ = np.array([ao_occ[p0:p1].sum() for _, _, p0, p1 in m.aoslice_by_atom()])
    hole = 1.0 - occ
    max_hole = hole.max()

    # Tight-binding bandwidth: t is the nearest-neighbour hopping integral, meaningful only
    # where there is one function per site, so it is reported for the minimal basis alone.
    if m.nao == N:
        h1_ao = m.intor('int1e_kin') + m.intor('int1e_nuc')
        s_inv_half = v @ np.diag(1.0 / np.sqrt(w)) @ v.T
        h1 = s_inv_half @ h1_ao @ s_inv_half
        bandwidth = 4 * abs(h1[0, 1]) * HA_EV
    else:
        bandwidth = float('nan')

    verdict = 'DELOCALISED' if max_hole < 2.0 / N else 'localised'
    print(f"{a:5.1f} {ehf:11.5f} {efci:11.5f} {(efci-ehf)*HA_EV:8.2f} "
          f"{bandwidth:10.2f} {max_hole:14.3f} {verdict:>12}")

print(f"\nA hole spread evenly over {N} sites has {1.0/N:.3f} per site; one sitting on a")
print("single site has 1.000. The exact ground state is the arbiter, and no barrier can be")
print("defined for a carrier that is not localised in the first place.")
print("\nBandwidth 4t is the scale a hopping barrier must EXCEED for hopping to be the")
print("transport mechanism rather than band motion. RealQM gives ~1.9 eV at a = 4.5 with")
print("clamped nuclei; standard theory at the same geometry gives a band and no barrier at")
print("all. The difference is the U -> infinity limit, not a disagreement about a number.")
