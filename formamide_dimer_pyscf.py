#!/usr/bin/env python3
"""
Formamide dimer — PySCF HF reference.
Computes dimer energy, monomer energy, and binding energy.
Also computes at the same geometry used in the WebGPU simulation.
"""
import numpy as np

A = 1.8897  # bohr per Angstrom
bohr2ang = 0.529177  # Angstrom per bohr
csep = 5.5  # C-C separation in Angstroms

# Coordinates in Angstroms
# Formamide 1 (left, NH2 donor)
x1 = -csep / 2
dimer_atoms = [
    ('C', [x1, 0, 0]),
    ('O', [x1 - 1.22, 0, 0]),
    ('H', [x1, 0, 1.09]),
    ('N', [x1 + 1.34, 0.3, 0]),
    ('H', [x1 + 1.34 + 0.87, -0.2, 0]),
    ('H', [x1 + 1.34 + 0.3, 1.2, 0]),
    # Formamide 2 (right, C=O acceptor)
    ('C', [csep/2, 0, 0]),
    ('O', [csep/2 - 1.22, 0, 0]),
    ('H', [csep/2, 0, -1.09]),
    ('N', [csep/2 + 1.34, -0.3, 0]),
    ('H', [csep/2 + 1.34 + 0.87, 0.2, 0]),
    ('H', [csep/2 + 1.34 + 0.3, -1.2, 0]),
]

# Single formamide monomer (isolated, same internal geometry as molecule 1)
monomer_atoms = [
    ('C', [0, 0, 0]),
    ('O', [-1.22, 0, 0]),
    ('H', [0, 0, 1.09]),
    ('N', [1.34, 0.3, 0]),
    ('H', [1.34 + 0.87, -0.2, 0]),
    ('H', [1.34 + 0.3, 1.2, 0]),
]

try:
    from pyscf import gto, scf, grad

    for basis in ['sto-3g', '6-31g*']:
        print(f"\n{'='*70}")
        print(f"HF/{basis}")
        print(f"{'='*70}")

        # Dimer
        mol_d = gto.M(atom=dimer_atoms, basis=basis, verbose=0)
        mf_d = scf.RHF(mol_d).run()
        E_dimer = mf_d.e_tot
        print(f"  Dimer energy:   {E_dimer:.6f} Ha")

        # Monomer
        mol_m = gto.M(atom=monomer_atoms, basis=basis, verbose=0)
        mf_m = scf.RHF(mol_m).run()
        E_mono = mf_m.e_tot
        print(f"  Monomer energy: {E_mono:.6f} Ha")

        # Binding energy
        E_bind = E_dimer - 2 * E_mono
        E_bind_kcal = E_bind * 627.509
        E_bind_eV = E_bind * 27.2114
        print(f"  Binding energy: {E_bind:.6f} Ha = {E_bind_kcal:.2f} kcal/mol = {E_bind_eV:.4f} eV")
        print(f"  (negative = bound)")

        # Dimer forces
        g_d = mf_d.nuc_grad_method().run()
        forces = -g_d.de  # force = -gradient
        print(f"\n  Dimer forces (Ha/Bohr):")
        labels = ['C1','O1','HC1','N1','HN1a','HN1b','C2','O2','HC2','N2','HN2a','HN2b']
        for i, (lbl, f) in enumerate(zip(labels, forces)):
            print(f"    {lbl:6s}  Fx={f[0]:+.4f}  Fy={f[1]:+.4f}  Fz={f[2]:+.4f}  |F|={np.linalg.norm(f):.4f}")

        # Key distances
        coords = np.array([a[1] for a in dimer_atoms])
        d_HO = np.linalg.norm(coords[4] - coords[7])  # HN1a - O2
        d_NO = np.linalg.norm(coords[3] - coords[7])  # N1 - O2
        d_CC = np.linalg.norm(coords[0] - coords[6])  # C1 - C2
        print(f"\n  Key distances:")
        print(f"    H···O (H-bond): {d_HO:.3f} A")
        print(f"    N···O (donor-acceptor): {d_NO:.3f} A")
        print(f"    C···C (inter-monomer): {d_CC:.3f} A")

    # Summary
    print(f"\n{'='*70}")
    print("Expected from literature:")
    print("  H-bond distance (O···H): ~1.9 A")
    print("  Donor-acceptor (N···O):  ~2.9 A")
    print("  Binding energy:          ~-14 kcal/mol (double H-bond dimer)")
    print("  Note: single H-bond formamide dimer ~-7 kcal/mol")
    print(f"{'='*70}")

except ImportError:
    print("PySCF not installed. Run: pip3 install pyscf")
