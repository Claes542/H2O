#!/usr/bin/env python3
"""
Alanine dipeptide (Ace-Ala-NMe) — PySCF HF reference calculation.

Coordinates taken directly from ala_dipeptide.html addAtom() calls.
The HTML uses au_x, au_y, au_z in atomic units (bohr), with r = A = 1.8897.
We convert bohr -> Angstroms for PySCF input (divide by 1.8897259886).
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Extract coordinates from the HTML (au values computed as factor * r)
#    where r = A = 1.8897 bohr/Angstrom
# ---------------------------------------------------------------------------
A = 1.8897  # bohr per Angstrom, same as in the HTML

# (au_x, au_y, au_z, Z, element_symbol)
# Directly from the addAtom calls: addAtom(coeff_x * r, coeff_y * r, coeff_z * r, Z, el)
atoms_au = [
    # Ace methyl
    (-5.5*A,    0.0,       0.0,      6, 'C'),   # 0  Cme
    (-6.1*A,    0.9*A,     0.0,      1, 'H'),   # 1
    (-6.1*A,   -0.9*A,     0.0,      1, 'H'),   # 2
    (-6.1*A,    0.0,       0.9*A,    1, 'H'),   # 3
    # Ace carbonyl C
    (-4.0*A,    0.0,       0.0,      6, 'C'),   # 4  Cace
    # Ace O
    (-3.7*A,    0.0,       1.2*A,    8, 'O'),   # 5  Oace
    # Peptide bond N1
    (-3.0*A,    0.0,      -0.7*A,    7, 'N'),   # 6  N1
    (-3.3*A,    0.0,      -1.7*A,    1, 'H'),   # 7  HN1
    # Ca
    (-1.5*A,    0.0,      -0.5*A,    6, 'C'),   # 8  Ca
    # Ha
    (-1.3*A,    0.0,      -1.6*A,    1, 'H'),   # 9  Ha
    # Cb (methyl sidechain)
    (-1.2*A,    1.5*A,     0.0,      6, 'C'),   # 10 Cb
    (-0.5*A,    2.1*A,     0.5*A,    1, 'H'),   # 11
    (-2.1*A,    2.1*A,     0.0,      1, 'H'),   # 12
    (-1.0*A,    1.5*A,    -1.0*A,    1, 'H'),   # 13
    # Ala carbonyl C
    (-0.2*A,   -0.5*A,     0.3*A,    6, 'C'),   # 14 Cala
    # Ala O
    (-0.3*A,   -0.5*A,     1.5*A,    8, 'O'),   # 15 Oala
    # Peptide bond N2
    ( 1.0*A,   -0.8*A,    -0.3*A,    7, 'N'),   # 16 N2
    ( 1.0*A,   -0.8*A,    -1.3*A,    1, 'H'),   # 17 HN2
    # NMe methyl
    ( 2.3*A,   -1.0*A,     0.2*A,    6, 'C'),   # 18 Cnme
    ( 2.9*A,   -0.1*A,     0.2*A,    1, 'H'),   # 19
    ( 2.9*A,   -1.8*A,    -0.2*A,    1, 'H'),   # 20
    ( 2.2*A,   -1.2*A,     1.2*A,    1, 'H'),   # 21
]

# Convert bohr -> Angstrom for PySCF
BOHR_TO_ANG = 0.529177210903  # 1 bohr in Angstroms

labels = [
    'C(Ace-Me)', 'H', 'H', 'H',
    'C(Ace-CO)', 'O(Ace)',
    'N1', 'H(N1)',
    'C(alpha)', 'H(alpha)',
    'C(beta)', 'H', 'H', 'H',
    'C(Ala-CO)', 'O(Ala)',
    'N2', 'H(N2)',
    'C(NMe)', 'H', 'H', 'H',
]

print("=" * 70)
print("Alanine Dipeptide (Ace-Ala-NMe) — PySCF HF Reference")
print("=" * 70)

# Build PySCF geometry string
coords_ang = []
geom_lines = []
for i, (ax, ay, az, Z, el) in enumerate(atoms_au):
    x_ang = ax * BOHR_TO_ANG
    y_ang = ay * BOHR_TO_ANG
    z_ang = az * BOHR_TO_ANG
    coords_ang.append((x_ang, y_ang, z_ang))
    geom_lines.append(f'{el}  {x_ang:12.6f}  {y_ang:12.6f}  {z_ang:12.6f}')

coords_ang = np.array(coords_ang)

print(f"\nNumber of atoms: {len(atoms_au)}")
print("\nCoordinates (Angstrom):")
print(f"{'#':>3s}  {'Label':>12s}  {'X':>10s}  {'Y':>10s}  {'Z':>10s}")
for i, (ax, ay, az, Z, el) in enumerate(atoms_au):
    x_a, y_a, z_a = coords_ang[i]
    print(f"{i:3d}  {labels[i]:>12s}  {x_a:10.4f}  {y_a:10.4f}  {z_a:10.4f}")

geom_str = '; '.join(geom_lines)

# ---------------------------------------------------------------------------
# 2. PySCF calculations
# ---------------------------------------------------------------------------
from pyscf import gto, scf, grad

def run_hf(basis_name):
    """Run HF calculation and return mol, energy, and nuclear gradients."""
    mol = gto.M(
        atom=geom_str,
        basis=basis_name,
        unit='Angstrom',
        verbose=3,
    )
    print(f"\nBasis: {basis_name}")
    print(f"  Number of basis functions: {mol.nao_nr()}")
    print(f"  Number of electrons:       {mol.nelectron}")

    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.conv_tol = 1e-9
    e_tot = mf.kernel()

    print(f"\n  HF/{basis_name} Total Energy: {e_tot:.10f} Hartree")
    print(f"  HF/{basis_name} Total Energy: {e_tot * 27.2114:.6f} eV")

    # Compute forces (negative of gradient)
    g = grad.RHF(mf)
    de = g.kernel()  # gradient dE/dR in Hartree/Bohr
    forces = -de      # forces = -gradient

    print(f"\n  Forces on atoms (Hartree/Bohr):")
    print(f"  {'#':>3s}  {'Label':>12s}  {'Fx':>12s}  {'Fy':>12s}  {'Fz':>12s}  {'|F|':>12s}")
    for i in range(len(atoms_au)):
        fx, fy, fz = forces[i]
        fmag = np.sqrt(fx**2 + fy**2 + fz**2)
        print(f"  {i:3d}  {labels[i]:>12s}  {fx:12.6f}  {fy:12.6f}  {fz:12.6f}  {fmag:12.6f}")

    max_f = np.max(np.linalg.norm(forces, axis=1))
    rms_f = np.sqrt(np.mean(np.sum(forces**2, axis=1)))
    print(f"\n  Max |force|: {max_f:.6f} Ha/Bohr")
    print(f"  RMS |force|: {rms_f:.6f} Ha/Bohr")

    return mol, e_tot, forces

# ---------------------------------------------------------------------------
# 3. Run both basis sets
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("HF/STO-3G (minimal basis — matches simple quantum model)")
print("=" * 70)
mol_sto, e_sto, f_sto = run_hf('sto-3g')

print("\n" + "=" * 70)
print("HF/6-31G* (polarized double-zeta — better reference)")
print("=" * 70)
mol_631g, e_631g, f_631g = run_hf('6-31g*')

# ---------------------------------------------------------------------------
# 4. Bond lengths
# ---------------------------------------------------------------------------
# Define bonded pairs (0-indexed, from molecular topology)
bonds = [
    # Ace methyl C-H bonds
    (0, 1), (0, 2), (0, 3),
    # Ace methyl C - Ace carbonyl C
    (0, 4),
    # Ace C=O
    (4, 5),
    # Ace C - N1 (peptide bond)
    (4, 6),
    # N1 - H
    (6, 7),
    # N1 - Ca
    (6, 8),
    # Ca - Ha
    (8, 9),
    # Ca - Cb
    (8, 10),
    # Cb - H (methyl)
    (10, 11), (10, 12), (10, 13),
    # Ca - Cala (carbonyl)
    (8, 14),
    # Cala = O
    (14, 15),
    # Cala - N2 (peptide bond)
    (14, 16),
    # N2 - H
    (16, 17),
    # N2 - Cnme
    (16, 18),
    # Cnme - H (methyl)
    (18, 19), (18, 20), (18, 21),
]

print("\n" + "=" * 70)
print("Bond Lengths (Angstrom)")
print("=" * 70)
print(f"{'Bond':>30s}  {'Distance':>10s}")
for (a, b) in bonds:
    dx = coords_ang[a] - coords_ang[b]
    dist = np.linalg.norm(dx)
    bond_label = f"{labels[a]}({a})-{labels[b]}({b})"
    print(f"{bond_label:>30s}  {dist:10.4f}")

# Also print in bohr for comparison with the HTML grid code
print(f"\n{'Bond':>30s}  {'Dist(Bohr)':>10s}")
for (a, b) in bonds:
    ax1, ay1, az1 = atoms_au[a][:3]
    ax2, ay2, az2 = atoms_au[b][:3]
    dist_bohr = np.sqrt((ax1-ax2)**2 + (ay1-ay2)**2 + (az1-az2)**2)
    bond_label = f"{labels[a]}({a})-{labels[b]}({b})"
    print(f"{bond_label:>30s}  {dist_bohr:10.4f}")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  HF/STO-3G  energy: {e_sto:16.10f} Ha  ({e_sto*27.2114:12.6f} eV)")
print(f"  HF/6-31G*  energy: {e_631g:16.10f} Ha  ({e_631g*27.2114:12.6f} eV)")
print(f"  Basis set lowering: {(e_631g - e_sto)*27.2114:.6f} eV")
print(f"                      {(e_631g - e_sto)*627.509:.4f} kcal/mol")
