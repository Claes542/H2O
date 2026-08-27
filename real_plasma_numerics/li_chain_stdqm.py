#!/usr/bin/env python3
"""
The standard-quantum-mechanics reference for the lithium chain.

`chain_slide.html` computes a chain of Li-like pseudopotential cores, one valence electron
each, in RealQM: non-overlapping unit charge densities on domains meeting at free boundaries.
This computes the same chain the ordinary way -- Kohn-Sham DFT with a real basis, delocalised
orbitals, all electrons -- so the two can be set side by side on quantities both can produce.

What is compared, and why each is a fair test:

  equilibrium spacing   a geometry, no interpretation needed on either side
  cohesive energy       E(chain)/n - E(atom); both frameworks compute both terms
  HOMO-LUMO gap         the conduction verdict in standard theory. A 1-D chain with one
                        electron per site is a half-filled band, hence a metal in tight
                        binding -- but Peierls' theorem says that band is unstable to
                        dimerisation, which opens a gap. Which one wins at a given spacing
                        is exactly the question RealQM answers with its registry preference
                        and its depinning transition.

RealQM's answers, for the comparison (a in bohr, n = 5 cores unless stated):

  equilibrium           a ~ 4.5      (uniform-chain energies -1.29985 / -1.334 / -1.348 /
                                      -1.320 at a = 6.0 / 5.0 / 4.5 / 4.0)
  registry              bond-centred lies 0.064 Ha below atom-centred
  sliding barrier       +184 meV at a=6.0, +136 at 5.0, ~0 at 4.5, -190 at 4.0
                        i.e. tunable through zero: a depinning transition near a ~ 4.6
  wall defect           179 meV, length-independent to 5%

Usage:  python3 li_chain_stdqm.py [n] [basis]      default 7 atoms, 6-31g
"""
import sys

import numpy as np
from pyscf import dft, gto, scf

N = int(sys.argv[1]) if len(sys.argv) > 1 else 7
BASIS = sys.argv[2] if len(sys.argv) > 2 else '6-31g'
XC = 'pbe'
SPACINGS = [4.0, 4.5, 5.0, 6.0, 7.0]          # bohr, matching the RealQM scan
HARTREE_EV = 27.211386


def atom_energy():
    """Isolated Li, spin 1 (2s^1). The reference for the cohesive energy."""
    m = gto.M(atom='Li 0 0 0', basis=BASIS, spin=1, unit='Bohr', verbose=0)
    mf = dft.UKS(m)
    mf.xc = XC
    return mf.kernel()


def chain(a, n):
    """n Li atoms in a line at spacing a (bohr). n odd -> one unpaired electron, spin 1."""
    pos = [(0.0, 0.0, (i - (n - 1) / 2) * a) for i in range(n)]
    m = gto.M(atom=[('Li', p) for p in pos], basis=BASIS, spin=(n % 2),
              unit='Bohr', verbose=0)
    mf = dft.UKS(m) if (n % 2) else dft.RKS(m)
    mf.xc = XC
    e = mf.kernel()
    # HOMO-LUMO from the occupied/virtual split. For a chain this is the finite-size stand-in
    # for the band gap: it closes as n grows if the system is metallic, and stays open if a
    # gap has opened.
    if isinstance(mf.mo_energy, np.ndarray) and mf.mo_energy.ndim == 1:
        occ, mo = mf.mo_occ, mf.mo_energy
        homo = mo[occ > 0].max()
        lumo = mo[occ == 0].min()
    else:
        homo = max(mo[oc > 0].max() for mo, oc in zip(mf.mo_energy, mf.mo_occ))
        lumo = min(mo[oc == 0].min() for mo, oc in zip(mf.mo_energy, mf.mo_occ))
    return e, (lumo - homo), mf.converged


print(f"Li chain, standard QM: {XC.upper()}/{BASIS}, n = {N} atoms, all electrons\n")
e_atom = atom_energy()
print(f"E(Li atom) = {e_atom:.6f} Ha\n")

print(f"{'a (bohr)':>9} {'E(chain) Ha':>14} {'cohesive eV/atom':>18} {'HOMO-LUMO eV':>14}  conv")
print('-' * 62)
rows = []
for a in SPACINGS:
    e, gap, ok = chain(a, N)
    coh = (N * e_atom - e) / N * HARTREE_EV        # positive = bound
    rows.append((a, e, coh, gap * HARTREE_EV))
    print(f"{a:9.2f} {e:14.6f} {coh:18.3f} {gap*HARTREE_EV:14.3f}  {'yes' if ok else 'NO'}")

best = min(rows, key=lambda r: r[1])
print(f"\nequilibrium spacing (of those sampled): a = {best[0]:.2f} bohr, "
      f"cohesive {best[2]:.3f} eV/atom, gap {best[3]:.3f} eV")
print("\nfor comparison:")
print("  RealQM (chain_slide, pseudopotential cores)  a_eq ~ 4.5 bohr")
print("  experiment, Li metal (bcc)                   nearest neighbour 5.7 bohr, "
      "cohesive 1.63 eV/atom")
