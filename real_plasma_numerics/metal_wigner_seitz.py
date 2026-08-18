"""
A metal in RealQM: the Wigner-Seitz cell, and what the framework leaves out.

The uniform electron gas is an idealisation and a poor test. A metal is real, and
it is also where RealQM's prescription turns out to coincide with an established
method: one electron domain per ion, filling the cell, with the normal derivative
vanishing at the cell face by symmetry -- which is precisely the Wigner-Seitz
boundary condition used for alkali metals since 1933.

So the comparison is sharp. Standard WS theory computes the band bottom E_0(r_s)
from exactly this problem, and then ADDS the mean kinetic energy of filling the
band, (3/5)E_F, which comes from the exclusion principle. RealQM has the first
term and not the second. This script computes both and reads off the consequences
that can be measured: the equilibrium radius and the bulk modulus.

Model: sodium, one valence electron per ion, Ashcroft empty-core pseudopotential

    V(r) = 0        r < r_c
    V(r) = -1/r     r >= r_c        (r_c = 1.67 a0 for Na)

in a Wigner-Seitz sphere of radius r_s, with the electron's own Hartree field and
the standard -0.9/r_s electrostatic (Madelung) term of the neutral sphere.
Boundary condition psi'(r_s) = 0.

Experimental sodium: r_s = 3.93 a0, bulk modulus 6.3 GPa, cohesive energy 1.11 eV.

STATUS (2026-08-18): SET UP, BASELINE NOT YET VALID. Do not quote the numbers.

    equilibrium r_s   RealQM 2.50 (hit search bound)  standard 2.74   exp 3.93
    bulk modulus      RealQM 790 GPa                  standard 315    exp 6.3

The STANDARD column is 40% off in radius and 50x too stiff, so the model is
wrong before RealQM enters and the comparison cannot discriminate. Two
omissions account for it: exchange and correlation are absent entirely
(exchange alone is -0.458/r_s Ha, a large binding term), and the electrostatic
term -0.9/r_s assumes a UNIFORM electron while the band bottom is computed from
a non-uniform state, so the two double-count.

One qualitative signal, recorded but not relied on: the RealQM curve has no
interior minimum in the searched range -- it wants to collapse -- which is what
omitting a repulsive term growing as r_s^-2 should do. Produced by a setup that
does not reproduce sodium, so it establishes nothing on its own.

TO FIX: add exchange -0.458/r_s and a correlation parameterisation (Wigner or
Perdew-Zunger), and make the electrostatics consistent with the computed density
instead of a uniform one. If the standard column then lands near 3.93 a0 and
6.3 GPa, the RealQM column becomes meaningful.

usage: python3 metal_wigner_seitz.py
"""
import math

HA_EV = 27.211386245988
HA_GPA = 29421.0158                  # 1 Ha/a0^3 in GPa
RC_NA = 1.67                         # Ashcroft empty core for Na, in Bohr


def band_bottom(rs, rc=RC_NA, N=1500):
    """Lowest s-state in the WS sphere with psi'(rs) = 0, u = r*psi, u(0) = 0.

    Includes the Hartree field of the electron's own uniform compensating
    charge? No: the electron IS the charge, and self-interaction is excluded in
    RealQM. What is included is the ion potential and, separately below, the
    electrostatic energy of the neutral cell.
    """
    h = rs / N
    r = [(i + 1) * h for i in range(N)]

    def shoot(E):
        u_prev, u = 0.0, 1e-12
        nodes = 0
        for i in range(1, N):
            V = 0.0 if r[i] < rc else -1.0 / r[i]
            u_next = 2.0 * u - u_prev + h * h * 2.0 * (V - E) * u
            if u_next * u < 0.0:
                nodes += 1
            u_prev, u = u, u_next
        up = (u - u_prev) / h
        return nodes, up - u / rs           # psi'(rs)=0  <=>  u'(rs) = u(rs)/rs

    lo, hi = -2.0, 5.0 / (rs * rs) + 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        nodes, resid = shoot(mid)
        if nodes > 0:
            hi = mid
        else:
            _, r_lo = shoot(lo)
            if r_lo * resid < 0.0:
                hi = mid
            else:
                lo = mid
    return 0.5 * (lo + hi)


def E_fermi(rs):
    """Fermi energy of one electron per WS sphere of radius rs (atomic units)."""
    return (9.0 * math.pi / 4.0) ** (2.0 / 3.0) / (2.0 * rs * rs)


_CACHE = {}


def energies(rs):
    """Energy per electron, with and without the band-filling term."""
    key = round(rs, 6)
    if key in _CACHE:
        return _CACHE[key]
    E0 = band_bottom(rs)
    electrostatic = -0.9 / rs                     # neutral-sphere Madelung term
    E_realqm = E0 + electrostatic                 # what the framework gives
    E_standard = E_realqm + 0.6 * E_fermi(rs)     # + mean kinetic energy of the filled band
    _CACHE[key] = (E_realqm, E_standard)
    return E_realqm, E_standard


def bulk_modulus(f, rs, drs=0.02):
    """B = v d2E/dv2 at v = (4pi/3) rs^3, by finite differences in rs."""
    def E(x):
        return f(x)
    v = (4.0 * math.pi / 3.0) * rs ** 3
    dv_drs = 4.0 * math.pi * rs ** 2
    e0, ep, em = E(rs), E(rs + drs), E(rs - drs)
    dE_drs = (ep - em) / (2 * drs)
    d2E_drs2 = (ep - 2 * e0 + em) / drs ** 2
    # dE/dv = (dE/drs)/(dv/drs);  d2E/dv2 = [d2E/drs2 - (dE/dv)(d2v/drs2)] / (dv/drs)^2
    dE_dv = dE_drs / dv_drs
    d2v_drs2 = 8.0 * math.pi * rs
    d2E_dv2 = (d2E_drs2 - dE_dv * d2v_drs2) / dv_drs ** 2
    return v * d2E_dv2 * HA_GPA


def minimise(f, lo=2.5, hi=7.0):
    for _ in range(40):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if f(m1) < f(m2):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def main():
    print("Sodium in the Wigner-Seitz cell (Ashcroft empty core, r_c = 1.67 a0)")
    print("experiment: r_s = 3.93 a0,  B = 6.3 GPa\n")
    print(f"{'r_s':>6} {'E_0+es (RealQM)':>17} {'+ (3/5)E_F (std)':>18} {'(3/5)E_F':>11}   (Ha)")
    for rs in (3.0, 3.5, 3.93, 4.5, 5.0, 6.0):
        a, b = energies(rs)
        print(f"{rs:>6.2f} {a:>17.4f} {b:>18.4f} {0.6*E_fermi(rs):>11.4f}")

    rq = minimise(lambda x: energies(x)[0])
    st = minimise(lambda x: energies(x)[1])
    print(f"\nequilibrium r_s:   RealQM {rq:.2f} a0     standard {st:.2f} a0     experiment 3.93 a0")
    B_rq = bulk_modulus(lambda x: energies(x)[0], rq)
    B_st = bulk_modulus(lambda x: energies(x)[1], st)
    print(f"bulk modulus:      RealQM {B_rq:.1f} GPa   standard {B_st:.1f} GPa   experiment 6.3 GPa")
    print("\nThe band-filling term (3/5)E_F is the one RealQM lacks. It is repulsive and")
    print("grows as r_s shrinks, so omitting it should pull the equilibrium radius in and")
    print("change the stiffness -- which is exactly what a measurement can check.")


if __name__ == '__main__':
    main()
