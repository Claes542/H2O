"""
The alkali series in a Wigner-Seitz cell: RealQM against standard theory.

metal_na.py established the comparison on sodium. One metal invites the question
whether it was cherry-picked, and the series answers it: five alkalis, each
bringing one parameter (its Ashcroft empty-core radius) and two measured numbers
(equilibrium radius and bulk modulus), across which r_s varies by 70% and B by a
factor of six.

Both columns solve the same Wigner-Seitz cell problem -- one electron domain per
ion, filling the cell, psi'(r_s) = 0 at the face, which is RealQM's prescription
and the standard construction alike. They differ only in the kinetic energy:

  standard   (3/5)E_F from filling the band, plus exchange and correlation
  RealQM     <T> of the cell state itself, and neither exchange nor correlation

Electrostatics is common: -0.9/r_s for a point ion in a neutral sphere, plus
1.5 r_c^2/r_s^3 from the empty core.

usage: python3 metal_alkalis.py
"""
import math

HA_GPA = 29421.0158

#                  r_c (a0)   r_s exp    B exp (GPa)
ALKALIS = {
    'Li': (1.06, 3.25, 11.6),
    'Na': (1.67, 3.93, 6.3),
    'K':  (2.14, 4.86, 3.1),
    'Rb': (2.31, 5.20, 2.5),
    'Cs': (2.50, 5.62, 2.0),
}


def cell_state(rs, rc, N=3000):
    """Band bottom in the WS sphere with psi'(rs)=0; returns (E0, <T>)."""
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
        return E0, float('nan')
    V_exp = sum((0.0 if r[i] < rc else -1.0 / r[i]) * u[i] * u[i]
                for i in range(N)) * h / nrm
    return E0, E0 - V_exp


def E_fermi(rs):
    return (9.0 * math.pi / 4.0) ** (2.0 / 3.0) / (2.0 * rs * rs)


def make_energy(rc):
    cache = {}

    def energies(rs):
        key = round(rs, 5)
        if key in cache:
            return cache[key]
        _, T = cell_state(rs, rc)
        es = -0.9 / rs + 1.5 * rc * rc / rs ** 3
        std = 0.6 * E_fermi(rs) + es - 0.458 / rs - 0.44 / (rs + 7.8)
        rqm = T + es                 # strict non-overlap: cell state's own <T>
        sea = es                     # shared territory: T = 0 for a uniform sea
        cache[key] = (rqm, std, sea)
        return cache[key]
    return energies


def minimise(f, lo=1.5, hi=14.0):
    for _ in range(70):
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
    print("Alkali metals in the Wigner-Seitz cell (Ashcroft empty core)\n")
    print(f"{'':4}{'r_c':>6}{'':3}{'r_s split':>10}{'r_s sea':>10}{'r_s std':>9}{'r_s exp':>9}"
          f"{'':3}{'B split':>9}{'B sea':>9}{'B std':>8}{'B exp':>8}")
    rows = []
    for el, (rc, rs_exp, B_exp) in ALKALIS.items():
        f = make_energy(rc)
        r_r = minimise(lambda x: f(x)[0])
        r_s = minimise(lambda x: f(x)[1])
        r_sea = minimise(lambda x: f(x)[2])
        B_r = bulk_modulus(lambda x: f(x)[0], r_r)
        B_s = bulk_modulus(lambda x: f(x)[1], r_s)
        B_sea = bulk_modulus(lambda x: f(x)[2], r_sea)
        rows.append((el, r_r, r_s, rs_exp, B_r, B_s, B_exp, r_sea, B_sea))
        print(f"{el:4}{rc:>6.2f}{'':3}{r_r:>10.2f}{r_sea:>10.2f}{r_s:>9.2f}{rs_exp:>9.2f}"
              f"{'':3}{B_r:>9.1f}{B_sea:>9.1f}{B_s:>8.1f}{B_exp:>8.1f}")

    def mape(i, j):
        return 100 * sum(abs(r[i] - r[j]) / r[j] for r in rows) / len(rows)

    print(f"\nmean abs error r_s:  split-domain {mape(1,3):.1f}%   shared-sea {mape(7,3):.1f}%"
          f"   standard {mape(2,3):.1f}%")
    print(f"mean abs error B:    split-domain {mape(4,6):.1f}%   shared-sea {mape(8,6):.1f}%"
          f"   standard {mape(5,6):.1f}%")
    print("\n'split' = strict non-overlap, each electron its own Wigner-Seitz cell.")
    print("'sea'   = the relaxed constraint: valence electrons share one territory, so a")
    print("          uniform density has grad psi = 0 and the kinetic term vanishes exactly.")
    print("\nAcross the series r_s varies by 70% and B by a factor of six, so tracking")
    print("the trend is a stronger statement than matching any single metal.")


if __name__ == '__main__':
    main()
