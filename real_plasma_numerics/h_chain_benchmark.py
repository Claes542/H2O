#!/usr/bin/env python3
"""
The hydrogen chain: the right test, and one with an exact answer.

The lithium chain was the wrong system to compare on. It was chosen because a
pseudopotential core costs one electron per site, which kept the RealQM solver on its
verified path -- a computational reason. But a simple metal is precisely what the scope
section of the article excludes, and precisely where density-functional theory is at its
best, since its delocalisation bias is harmless when the system really is metallic.

A chain of hydrogen atoms is better on every count:

  * one electron per site with no pseudopotential, so no free radius to argue about --
    the parameter-choice objection disappears entirely
  * stretched, it is the canonical Mott insulator: each atom holds its electron and the
    gap is on-site repulsion, which is the U -> infinity regime the construction is for
  * DFT is known to FAIL there, delocalisation error being at its worst, so the standard
    column sits in its own worst case rather than its best
  * and in a minimal basis the chain is small enough for FULL CONFIGURATION INTERACTION,
    which is exact within that basis. The comparison is then against an exact answer
    rather than against another model, and "choose the approximation to suit the problem"
    has nothing left to choose

The scan spans both regimes in one sweep. Compressed, the chain is metallic and DFT should
do well. Stretched, it is Mott and DFT should fail while a localising construction should
not. The crossover is the test.

Usage:  python3 h_chain_benchmark.py [n] [basis]     default 6 atoms, sto-6g
"""
import sys

import numpy as np
from pyscf import ci, cc, dft, fci, gto, scf

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BASIS = sys.argv[2] if len(sys.argv) > 2 else 'sto-6g'
SPACINGS = [1.4, 1.8, 2.4, 3.0, 4.0, 6.0]        # bohr: bonded -> stretched -> Mott
HA_EV = 27.211386


def chain(a, n):
    pos = [('H', (0.0, 0.0, (i - (n - 1) / 2) * a)) for i in range(n)]
    return gto.M(atom=pos, basis=BASIS, unit='Bohr', verbose=0, spin=n % 2)


# The atomic reference must be computed in the SAME basis as the chain, not taken as the
# exact -0.5. STO-6G is a six-Gaussian fit to a Slater 1s, not a 1s: it gives -0.47117, so
# using -0.5 charges every cohesive energy 0.029 Ha per atom -- about 0.8 eV -- and the sign
# of the error makes chains look less bound than they are. The tell is the a = 6.0 row, where
# FCI returns exactly N times the basis-set atom and the chain is fully dissociated.
_mh = gto.M(atom='H 0 0 0', basis=BASIS, spin=1, unit='Bohr', verbose=0)
e_atom_fci = scf.UHF(_mh).run().e_tot        # one electron: HF is exact in any basis
print(f"Hydrogen chain, n = {N}, basis {BASIS}")
print(f"E(H) in this basis = {e_atom_fci:.6f} Ha (exact value -0.5; the deficit is the basis)")
print("Cohesive energies below are referred to that, not to -0.5.\n")
print(f"{'a':>5} {'HF':>11} {'PBE':>11} {'CCSD':>11} {'FCI':>11} "
      f"{'E_corr':>9} {'PBE-FCI':>9}")
print(f"{'(a0)':>5} {'':>11} {'':>11} {'':>11} {'(exact)':>11} {'(eV)':>9} {'(eV)':>9}")
print('-' * 74)

rows = []
for a in SPACINGS:
    m = chain(a, N)
    mf = scf.RHF(m).run()
    ehf = mf.e_tot

    ks = dft.RKS(m); ks.xc = 'pbe'; ks.kernel()
    eks = ks.e_tot

    try:
        eccsd = cc.CCSD(mf).run().e_tot
    except Exception:
        eccsd = float('nan')

    # FCI: exact within the basis. Minimal basis keeps this tractable for n <= 10.
    try:
        cisolver = fci.FCI(mf)
        efci = cisolver.kernel()[0]
    except Exception:
        efci = float('nan')

    ecorr = (efci - ehf) * HA_EV if np.isfinite(efci) else float('nan')
    dft_err = (eks - efci) * HA_EV if np.isfinite(efci) else float('nan')
    rows.append((a, ehf, eks, eccsd, efci, ecorr, dft_err))
    print(f"{a:5.1f} {ehf:11.5f} {eks:11.5f} {eccsd:11.5f} {efci:11.5f} "
          f"{ecorr:9.2f} {dft_err:9.2f}")

print("\nE_corr = FCI - HF: the correlation energy. It GROWS on stretching -- that growth")
print("is the Mott regime announcing itself, and it is what a single determinant cannot")
print("represent.")
print("PBE - FCI: the standard column's error against the exact answer, in eV. If it grows")
print("with spacing, DFT is failing exactly where the construction of this article is aimed.")

fin = [r for r in rows if np.isfinite(r[4])]
if fin:
    best = min(fin, key=lambda r: r[4])
    print(f"\nequilibrium (FCI, of those sampled): a = {best[0]:.1f} bohr, "
          f"E = {best[4]:.5f} Ha, cohesive {(N*e_atom_fci - best[4])/N*HA_EV:.3f} eV/atom")
    print("\nRealQM runs the same geometries with bare protons (rc = 0) in chain_slide.html.")
