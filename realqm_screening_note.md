# Screening in RealQM: the continuum limit gives Thomas–Fermi exactly

Working note, 2026-08-17. Atomic units throughout (ħ = m = e = 1, 4πε₀ = 1).

## Why this is worth doing

A plasma is a Coulomb many-body system, so RealQM models it in its own terms
rather than displacing an established theory. And screening has a *known
analytic answer* in two limits, so the test cannot be tuned: either the length
comes out right or it does not. No mesh, no configuration selection, no solver —
the whole calculation is a linearisation.

## The functional

Partition an electron gas of density *n* into non-overlapping cells, one per
electron, of linear size a = n^(−1/3) — the RealQM postulate. The localisation
cost per electron scales as ħ²/2m · a^(−2) ∝ n^(2/3), so the kinetic energy
*density* is

    t(n) = C n^(5/3)

with C fixed by the cell's shape and boundary condition. **This is the
Thomas–Fermi form**, obtained here from non-overlap rather than from Fermi
statistics.

## Linearisation

Minimise E[n] = ∫ t(n) + electrostatics at fixed chemical potential. With a test
charge Q at the origin and a neutralising background,

    (5/3) C n^(2/3) − φ = μ

Linearising n = n₀ + δn about the uniform state gives

    δn = (9 n₀^(1/3) / 10C) φ

and substituting into ∇²φ = −4π[Q δ(r) − δn] gives a Helmholtz equation,

    ∇²φ − k²φ = −4π Q δ(r),     **k² = 18π n₀^(1/3) / (5C)**

hence the Yukawa potential φ = Q e^(−kr)/r. Screening, with λ = 1/k ∝ n^(−1/6).

## The comparison

Thomas–Fermi for a free electron gas gives k²_TF = (4/π)(3π²n)^(1/3).

Setting C to the Thomas–Fermi coefficient, C_TF = (3/10)(3π²)^(2/3) = 2.8712,
the formula above reproduces k²_TF **exactly**, at every density:

| n (a.u.) | λ, RealQM with C_TF | λ, Thomas–Fermi |
|---|---|---|
| 0.01 | 1.086 a₀ | 1.086 a₀ |
| 0.1 | 0.740 a₀ | 0.740 a₀ |
| 1.0 | 0.504 a₀ | 0.504 a₀ |

So the **form is exact** — Yukawa, with k² ∝ n^(1/3) — and everything rests on
the single constant C.

## But C = 0 for a uniform gas, and that is the real finding

The step above assumed the cell has a nonzero localisation cost. Check it against
the actual functional and it does not.

RealQM's kinetic term is ½∫_{Ω_i}|∇ψ_i|², integrated over each electron's **own**
domain, subject to ∫_{Ω_i}ψ_i² = 1, with the interfaces free to move and **no
node imposed** — the atom calibration found the free (Neumann) plane decisive
against a Dirichlet node.

Now take ψ_i constant = 1/√V on each cell. It satisfies the normalisation, the
total density Σψ_i² is uniform, the Coulomb energy is minimal against the
neutralising background, and **∇ψ_i = 0 everywhere, so the kinetic energy is
exactly zero.** The discontinuity at the cell face costs nothing: the integral
runs over Ω_i only, and no node is imposed there.

The uniform state is therefore the global minimum, with **C = 0** at every
density. No localisation energy, no degeneracy pressure, no quantum screening;
λ → ∞.

**This is confirmed in the implementation, not only in the formulation.** In
`molecule_nucleus.js` the U-update carries the comment *"Neumann BC at domain
boundaries; Dirichlet (ψ=0) at e-P boundaries if enabled"*, and the Dirichlet
branch is guarded by `myC * atoms[l].charge < 0.0` — it fires only between
*opposite* charges. Every electron–electron interface is Neumann. For a pure
electron gas there are no other interfaces, so the code agrees with the analysis.

### What this means

In an atom RealQM's kinetic energy is nonzero because nuclear attraction
**shapes** the density: the gradient comes from the profile, not from the
partition. Non-overlap by itself never produces kinetic energy. In a uniform gas
nothing shapes the density, so the mechanism that stands in for Pauli in atoms
supplies nothing at all.

Taken literally the model has **no Fermi pressure**: no degeneracy support for
metals, dense plasmas, or white dwarfs, and no Thomas–Fermi screening at any
density. Screening in a RealQM plasma would have to be entirely thermal.

That is a definite, falsifiable difference from standard physics, and it is
better found here than by a referee. It is also narrow: it says nothing against
RealQM's atomic and molecular results, where an attractor is always present.
The honest statement is that non-overlap reproduces shell structure where
something shapes the density and supplies nothing where nothing does.

## The plasma frequency, which RealQM gets exactly

Collective oscillation is the counter-case, and it is worth putting beside the
failure because it isolates the defect.

Displace the electron distribution rigidly by ξ against the ion background. The
resulting surface charge gives a restoring field E = 4πnξ, so ξ̈ = −4πnξ and

    ω_p² = 4πn

— temperature-free, degeneracy-free, depending only on charge density and
inertia.

**RealQM gives this exactly.** Under a rigid displacement each ψ_i is translated
unchanged, so ∇ψ_i is unchanged and the localisation energy is untouched. The
kinetic term contributes nothing, and nothing is left but Coulomb and inertia —
which is what the answer depends on. C = 0 does not bite here at all.

## Where it returns: the dispersion

The correction to the plasma frequency,

    ω²(k) = ω_p² + (3/5)k²v_F²   (degenerate)
    ω²(k) = ω_p² + 3k²v_th²      (classical)

is a **compressibility** term: it costs kinetic energy to squeeze the gas. With
C = 0 there is no such cost, so RealQM predicts

    ω²(k) = ω_p² ,  flat — no dispersion at all.

That is measurable. Plasmon dispersion in simple metals is a standard inelastic
X-ray scattering and EELS observable, and sodium and aluminium have
well-determined positive dispersion coefficients. A flat plasmon is not what is
seen.

## Landau damping: the one with a positive prognosis

Landau damping is the wave-particle resonance at v = ω/k, with the rate set by
∂f/∂v there. It requires a **velocity distribution**, which is why single-fluid
theories — and MHD — miss it entirely.

RealQM looks fluid-like but is not a single fluid. N electrons are N
non-overlapping domains, each carrying its own density and, in the
time-dependent form, its own current and drift velocity. **That is structurally
a multi-beam plasma**: a set of cold streams with distributed velocities.

And multi-beam models *do* reproduce Landau damping — it emerges as **phase
mixing** among the beams (Dawson; the Van Kampen modes are the exact analogue).
So RealQM has the machinery in principle, it would arise for the right reason
rather than as an inserted damping term, and it would be **reversible** — which
matches the physics, since the plasma-echo experiments show Landau damping is not
true dissipation.

The fault line reappears in where the velocity spread comes from:

- **thermal plasma** — spread from thermal motion, which the framework can carry
  (Brownian dynamics already exists in the solver). Should work.
- **degenerate plasma** — the spread *is* the Fermi sea, a consequence of Pauli.
  No Fermi sea here, so no spread and no damping where it should exist.

## The diagnosis, in one line

Three independent observables, one root cause:

| observable | governed by | RealQM |
|---|---|---|
| plasma frequency ω_p | charge + inertia | **exact** |
| Thomas–Fermi screening length | compressibility | zero — no screening |
| plasmon dispersion | compressibility | zero — flat |
| Landau damping, thermal | velocity spread from motion | should work — multi-beam phase mixing |
| Landau damping, degenerate | velocity spread from Pauli | absent |

**RealQM reproduces what follows from charge, inertia and thermal motion, and
misses what follows from Fermi statistics** — whether that appears as
compressibility or as a Fermi sea. That is a
sharper statement than "it lacks Fermi pressure", it is derivable on paper, and
it says exactly where to look: not in the electrostatics, which is sound, but in
what happens when a uniform density is compressed.

## The thermal extension, for the Debye limit

Both limits follow from k² = 4π dn/dμ. The degenerate case is above; the
classical case has dn/dμ = n/k_BT, giving the Debye result k²_D = 4πn/k_BT.
A finite-temperature occupation in the same functional should interpolate
between them — which is the point of interest, because standard plasma physics
*switches models* at the degeneracy boundary while this would be one functional
throughout.

## Status

Settled analytically and confirmed in the solver source, with no computation
beyond arithmetic — a useful contrast with the nuclear ladder, where three days
of machine time were spent on a harness.

The result is in two parts. **The form is right:** partitioning into
non-overlapping cells gives a kinetic energy density C·n^(5/3), the Thomas–Fermi
form, and with C = C_TF the screening length is reproduced exactly at every
density. **The coefficient is zero:** for a uniform gas with Neumann interfaces
the minimising density is constant and C = 0, so the model has no degeneracy
pressure and no quantum screening.

So the plasma track's first question is not about plasmas. It is whether RealQM
should have a mechanism supplying Fermi pressure in the absence of an attractor,
and if so what it is. Candidates worth examining: whether the free-boundary
condition at a *moving* interface differs from the static Neumann condition used
here; and whether the constraint that each electron occupy a *connected* domain
of prescribed charge does work that the energy integral alone does not capture.

Until that is resolved, a RealQM plasma screens thermally (Debye) and not
quantum-mechanically, which is itself a testable claim.
