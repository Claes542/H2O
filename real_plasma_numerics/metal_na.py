"""
Sodium in a Wigner-Seitz cell: standard theory against RealQM, done properly.

RealQM's prescription for a metal coincides with the Wigner-Seitz construction --
one electron domain per ion, filling the cell, normal derivative vanishing at the
cell face by symmetry. The frameworks then differ in the kinetic energy, and only
there:

  standard    (3/5)E_F, the mean kinetic energy of filling the band, plus an
              exchange term -0.458/r_s and a correlation term, both consequences
              of the exclusion principle;

  RealQM      the kinetic energy of the actual cell state, <T> from the solved
              band-bottom wavefunction, with no exchange or correlation term:
              non-overlap is supposed to do that work.

Electrostatics is common to both: -0.9/r_s for a point ion in a neutral sphere,
plus 1.5 r_c^2/r_s^3 from the Ashcroft empty core (r_c = 1.67 a0 for Na), which
is what pushes the equilibrium radius out to a realistic value.

Sodium: r_s = 3.93 a0, bulk modulus 6.3 GPa, cohesive energy 1.11 eV.

usage: python3 metal_na.py
"""
import math

HA_EV = 27.211386245988
HA_GPA = 29421.0158
RC = 1.67


def cell_state(rs, rc=RC, N=3000):
    """Band bottom in the WS sphere with psi'(rs)=0. Returns (E0, T, V)."""
    h = rs / N
    r = [(i + 1) * h for i in range(N)]

    def shoot(E):
        u = [0.0] * N
        u_prev, u_cur = 0.0, 1e-10
        u[0] = u_cur
        nodes = 0
        for i in range(1, N):
            V = 0.0 if r[i] < rc else -1.0 / r[i]
            u_next = 2.0 * u_cur - u_prev + h * h * 2.0 * (V - E) * u_cur
            if u_next * u_cur < 0.0:
                nodes += 1
            u_prev, u_cur = u_cur, u_next
            u[i] = u_cur
        return nodes, (u_cur - u_prev) / h - u_cur / rs, u

    lo, hi = -1.5, 5.0 / (rs * rs)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        nodes, resid, _ = shoot(mid)
        if nodes > 0:
            hi = mid
        else:
            _, r_lo, _ = shoot(lo)
            if r_lo * resid < 0.0:
                hi = mid
            else:
                lo = mid
    E0 = 0.5 * (lo + hi)
    _, _, u = shoot(E0)

    nrm = sum(x * x for x in u) * h
    if nrm <= 0:
        return E0, float('nan'), float('nan')
    V_exp = sum((0.0 if r[i] < rc else -1.0 / r[i]) * u[i] * u[i] for i in range(N)) * h / nrm
    return E0, E0 - V_exp, V_exp          # T = E0 - <V>


def E_fermi(rs):
    return (9.0 * math.pi / 4.0) ** (2.0 / 3.0) / (2.0 * rs * rs)


def electrostatic(rs, rc=RC):
    """Point ion in a neutral sphere, plus the empty-core repulsion."""
    return -0.9 / rs + 1.5 * rc * rc / rs ** 3


def exchange(rs):
    return -0.458 / rs


def correlation(rs):
    return -0.44 / (rs + 7.8)           # Wigner


_C = {}


def energies(rs):
    key = round(rs, 5)
    if key in _C:
        return _C[key]
    _, T, _ = cell_state(rs)
    es = electrostatic(rs)
    E_std = 0.6 * E_fermi(rs) + es + exchange(rs) + correlation(rs)
    E_rqm = T + es                       # cell kinetic energy, no exchange, no correlation
    _C[key] = (E_rqm, E_std, T)
    return _C[key]


def minimise(f, lo=1.5, hi=12.0):
    for _ in range(60):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if f(m1) < f(m2):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def bulk_modulus(f, rs, d=0.03):
    v = (4 * math.pi / 3) * rs ** 3
    dv = 4 * math.pi * rs ** 2
    e0, ep, em = f(rs), f(rs + d), f(rs - d)
    dE = (ep - em) / (2 * d)
    d2E = (ep - 2 * e0 + em) / d ** 2
    return v * (d2E - (dE / dv) * (8 * math.pi * rs)) / dv ** 2 * HA_GPA


def main():
    print("Sodium, Wigner-Seitz cell, Ashcroft empty core r_c = 1.67 a0")
    print("experiment: r_s = 3.93 a0, B = 6.3 GPa\n")
    print(f"{'r_s':>6} {'T_cell':>9} {'(3/5)E_F':>10} {'E_es':>9} {'E_x':>9} "
          f"{'E_RealQM':>10} {'E_std':>9}   (Ha)")
    for rs in (3.0, 3.5, 3.93, 4.5, 5.0, 6.0):
        a, b, T = energies(rs)
        print(f"{rs:>6.2f} {T:>9.4f} {0.6*E_fermi(rs):>10.4f} {electrostatic(rs):>9.4f} "
              f"{exchange(rs):>9.4f} {a:>10.4f} {b:>9.4f}")

    r_std = minimise(lambda x: energies(x)[1])
    r_rqm = minimise(lambda x: energies(x)[0])
    print(f"\n{'':22}{'RealQM':>12}{'standard':>12}{'experiment':>13}")
    print(f"{'equilibrium r_s (a0)':22}{r_rqm:>12.2f}{r_std:>12.2f}{3.93:>13.2f}")
    B_r = bulk_modulus(lambda x: energies(x)[0], r_rqm)
    B_s = bulk_modulus(lambda x: energies(x)[1], r_std)
    print(f"{'bulk modulus (GPa)':22}{B_r:>12.1f}{B_s:>12.1f}{6.3:>13.1f}")
    print("\nIf the standard column reproduces sodium, the RealQM column is a real test:")
    print("it keeps the same electrostatics and differs only in the kinetic energy --")
    print("the cell state's own <T> instead of the band-filling term, and no exchange.")


if __name__ == '__main__':
    main()
