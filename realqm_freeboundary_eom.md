# RealQM: complex family + atomic spectra (working note)

Supersedes the earlier free-boundary/sloshing-breathing drafts. Final coordinated state.

## Corpus split: REAL = energies, COMPLEX = motion
- **Real-valued RealQM** (flagship RealQMarXiv4 + atoms/nucleus/chemistry): ground states/energies from
  non-overlapping charge densities, no self-interaction. Non-overlap = a real correlation hole => beats HF
  (He ground -2.90 vs HF -2.86, exact -2.9037).
- **Complex RealQM = MOTION (current).** The complex phase means ONE thing: flow. j = Im(psi* grad psi) =
  rho grad S, zero for real psi. Two motions: oscillating current -> radiation; circulating current ->
  magnetism. Plus spin (internal circulation).

## Complex family (Gallery: "Currents, radiation, magnetism, spin")
1. **ComplexRealQM.tex** -- foundation (complex law, current, free-boundary charge conservation).
2. **RealQMAction.tex = "RealQM Atomic Spectra: Radiation as Real Charge Oscillation"** -- radiation
   (oscillating current). NEW; cites Complex, Magnetism, Spinor.
3. **MagnetismRealQM.tex** -- magnetism (circulating current) + NuclearMoment.tex.
4. **SpinorResidue.tex** -- spin (g=2, Stern-Gerlach, non-relativistic).
All cross-referenced; all in the gallery Articles block.

## Atomic spectra -- the CORRECTED story
- A line = real charge density oscillating between two configs at DeltaE = E_n - E_m; a dipole resonating
  with radiation. Schrodinger's antenna in real 3D, deterministic, no collapse, no abstract superposition.
- **BUT radiation needs the complex form.** The moving density IS a current: charge conservation
  d_t rho + div j = 0, and j = Im(psi* grad psi) = 0 for real psi => a real density is STATIC and DARK. So
  the wavefunction is complex DURING the transition. REAL = the levels + observed dipole; COMPLEX = the
  current carrying the motion. (Corrects the earlier "spectra are real, no complex needed" -- that was WRONG;
  a real density cannot oscillate.) Only DIFFERENCES observable (absolute clocks cancel).
- **Selection rule = geometry:** swing to one side (l-change, s->p) => dipole => radiates; spherical breath
  (s->s) => no dipole => DARK => metastability (He 2s).
- vs standard QM: line = formal superposition (= measurement question); RealQM = physical real charge motion.
  Same structure/numbers (one-active-electron, screened core); RealQM adds the physical mechanism + its own
  correlation-improved energies.

## Helium
- Optical spectrum = one active (outer) electron in the screened field of the 1s core; screening lifts the
  l-degeneracy (2s < 2p).
- Lines captured (allowed, l-changing): 2p->2s 1083 nm; 3s->2p 706.5 nm; 3d->2p 587.6 nm (D3, solar
  discovery line); 1s^2->1s2p 58.4 nm resonance.
- Dark: 1s^2->1s2s (s->s) = the 2s metastability (2^1S ~20 ms, 2^3S ~2 h).
- Split-1s ground is spherical (mirror halves sum) => invisible to radiation. Excited state is
  nested/spherical (NOT lopsided half-space), which is exactly what the metastability requires.
- **ortho/para SPLITTING = spin/exchange (~0.8 eV), BEYOND real RealQM** (non-overlap = spatial antisymmetry,
  not spin). Single-electron spin reachable (SpinorResidue g=2); collective/exchange not. Earlier
  sloshing/breathing mapping RETIRED (breathing was dipole-dark; both 1s2s->ground are s->s).

## Numbers computed
he_two_shell_radial.py (radial SCF, Hartree): E_1(1s)=-1.73, E_2(2s)=-0.15 Ha; E(1s2s)=-2.15, E(1s^2)=-2.85
(exact -2.9037). Full 3D domain solver atom_He.html: He ground -2.90 (beats HF -2.86 -> correlation via
non-overlap). atom.js: per-shell eigenvalue readout res.eig[m]. atom_shells.html: 3D two-shell (drifts off
the excited state -- use the radial SCF for excited eigenvalues).
