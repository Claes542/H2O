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

## What that makes the screening length: a measurement of C

Taking the naive hard-wall cubic cell of side a, the ground state is
E = 3π²/2a², so C_box = 3π²/2 = 14.804. That is 5.16× the Thomas–Fermi value and
gives a screening length **2.27× too long**, at every density (the ratio is
√(C/C_TF), independent of n).

The two boundary conditions bracket the answer:

| cell boundary condition | C | λ vs Thomas–Fermi |
|---|---|---|
| Neumann / free (constant ground state) | 0 | ∞ — no screening at all |
| **Thomas–Fermi** | **2.871** | **1.00** |
| Dirichlet / hard wall | 14.804 | 2.27× too long |

**This is the useful outcome.** RealQM's true localisation constant lies strictly
between the free and hard-wall extremes, and screening measures it directly
against a known answer. The interface condition is exactly the question recorded
as decisive in the atom calibration (Neumann free plane against Dirichlet node);
here it becomes a *number* with an experimental-strength target, rather than a
qualitative choice.

## The thermal extension, for the Debye limit

Both limits follow from k² = 4π dn/dμ. The degenerate case is above; the
classical case has dn/dμ = n/k_BT, giving the Debye result k²_D = 4πn/k_BT.
A finite-temperature occupation in the same functional should interpolate
between them — which is the point of interest, because standard plasma physics
*switches models* at the degeneracy boundary while this would be one functional
throughout.

## Status and next step

Analytic, checkable, and it required no computation beyond arithmetic. The next
step is not a simulation: it is to derive C for RealQM's actual free-boundary
cell — neither hard-wall nor free, but the interface where neighbouring densities
meet with continuity — and see whether it lands on 2.871. If it does, RealQM
reproduces Thomas–Fermi screening from first principles. If it does not, the
discrepancy is a single number and it says something specific about the
interface condition.
