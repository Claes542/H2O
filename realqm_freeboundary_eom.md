# RealQM Atomic Spectra (working note)

Article: **RealQMAction.tex** — "RealQM Atomic Spectra: Radiation as Real Charge Oscillation — Sloshing and
Breathing" (7 pp). Arc: Real-valued RealQM → Complex-valued RealQM → stationary states → coexistence of time
scales → two sources of spectra → fast free boundary → conclusion.

## Framing (the thesis)
Atomic spectra were the STARTING POINT of QM and are the atom's most VISIBLE feature, yet standard QM hides
the physics: levels = eigenvalues, a line = beat of a SUPERPOSITION of two eigenstates — a FORMAL device
whose meaning is left open (= the measurement question). RealQM gives physics: radiation = REAL
charge-density oscillation.

## §1 Real-valued RealQM
N electron densities psi_j^2 on non-overlapping domains Omega_j; nuclei = fixed kernel V_K. Unit charge
int_{Omega_j} psi_j^2 = 1; finite kinetic energy: psi_j in H^1(Omega_j), total Psi = sum psi_j in H^1(R^3)
(=> psi_j AGREE on interdomain boundaries = condition (i)). Energy (2):
  E = sum_j 1/2 int|grad psi_j|^2 + int V_K rho + sum_{j<k} int int psi_j^2 psi_k^2/|x-y|,  rho = sum psi_j^2.
NO self-repulsion (e-e sum over distinct j<k only); kinetic energy takes over the anti-collapse role.
Minimised over the WAVE FUNCTIONS ALONE — the partition {Omega_j} is an OUTCOME, not a separate variable:
  (ii) Neumann d_n psi_j = 0 = the NATURAL boundary condition of the psi-minimisation (part of each domain's
       own Poisson problem);
  (i) density continuity = the H^1 matching that LOCATES the interface.
Potential felt by electron j = V_K + phi_j, phi_j = field of the OTHER electrons (no self).

## §2 Complex-valued RealQM — WHY complex
The complex phase e^{-iEt} is PERIODIC MOTION (a clock). Its point: radiation = RESONANCE = agreement of
periodic motions. Two frequencies E_n, E_m => beat E_n-E_m = real periodic oscillation resonating with the EM
field = radiation. A static real density has phases frozen, no periodic motion, cannot resonate; complexifying
restores it. Extension is DIRECT (density phase-blind, so geometry+energy unchanged): i d_t psi_j = -1/2 lap
psi_j + (V_K + phi_j) psi_j. Current j_j = Im(psi_j* grad psi_j) = -|psi_j|^2 grad theta_j.

## §3 Stationary states = stationary points of the global energy E
Ground state = MINIMUM of E. Stationary states = STATIONARY POINTS of E (Schrödinger spirit), each a total
energy E_n (the eigenvalue), E_0 < E_1 < ... = the FULL STATES. NO separate eigenvalue problems over Omega_j:
the local E-L condition -1/2 lap u_j + (V_K+phi_j)u_j = E_j u_j is the local face of one global stationarity;
E_j = Lagrange multiplier / clock rate of domain j. Total E != sum_j E_j (shared interactions). Notation:
j,k = domains (rates E_j); n = global states (total energies E_n). Radiation between full states at E_n - E_m.

## §4-5 Two time scales / coexistence
psi_j = u_j(x; Gamma(t)) e^{-i theta_j}: SLOW quasi-static geometry Gamma(t) (from densities) + FAST
per-domain phase clocks theta_j = int E_j dt. Uniform clock carries no current -> boundary quasi-static, no
fast boundary EOM. The phase becomes physical through DIFFERENCES, in two ways — NEITHER a superposition:
- SLOSHING: charge physically MOVES between two full states Psi_n, Psi_m -> real transition, radiates at
  E_n - E_m. (Standard QM writes Psi = a Psi_n e^{-iE_n t} + b Psi_m e^{-iE_m t}, a formal superposition with
  unclear/missing physics; RealQM carries no such object, only real charge in motion.)
- BREATHING: two domains of ONE state (inner+outer shell) COEXIST, relative phase theta_j-theta_k at E_j-E_k;
  radiates only if the shells couple.

## §6 Two sources of spectra (COMPUTED for helium)
- SLOSHING = between two full states, E_n - E_m (total-energy difference). Clear, primary; includes single
  electron (H sloshes 1s<->2p).
- BREATHING = between two shells of one state, E_j - E_k (shell rates). Exotic, less clear (dipole-dark for
  spherical s-s; needs coupling).
Helium (he_two_shell_radial.py, radial SCF, ALL at one Hartree level, all computed):
  E_1=-1.73, E_2=-0.15 Ha (inner <r>=0.76, outer 4.9 a0);  E(1s2s)=-2.15, E(1s^2)=-2.85 Ha (exact -2.9037).
  ortho, SLOSHING = E(1s2s)-E(1s^2) = 0.70 Ha = 19.0 eV  (observed 19.8; Hartree error cancels in difference).
  para,  BREATHING = |E_1-E_2|      = 1.58 Ha = 43 eV.  ratio 2.26.
Caveat: standard ortho/para is a SPIN label, and there para (1s^2) owns the ground state (opposite
attachment). Method: use the RADIAL SCF, NOT the 3D relaxation solver (drifts off the excited state).

## Radiation observability (honest guardrail)
Only the ATOM's net dipole radiates (Maxwell couples to total rho). A single DOMAIN's beat is not separately
observable as radiation; a spherical breathing is dipole-dark (why 2s is metastable). Atom line = total-energy
difference (relaxation included); one-electron atom (H): atom = the one domain. Domain radiation is NOT a
distinguishing observable of RealQM vs standard QM (both radiate the total density).

## §7 Fast free boundary (open)
If the boundary must move on the FAST scale with the flow (V_n = j_j.n/rho_j), then density-continuity +
Neumann + flow conflict (Neumann => j.n=0 => V_n=0). Quasi-static scheme sidesteps by NOT posing it. Needed
only for bound-to-bound charge crossing in ~1 electronic period: attosecond charge migration (flagship),
sudden bond cleavage, fast ion-atom charge exchange, conical intersections. NOT for thermal chemistry (slow)
or ionization/decay (charge to continuum = OUTER absorbing boundary, single domain leaking).
