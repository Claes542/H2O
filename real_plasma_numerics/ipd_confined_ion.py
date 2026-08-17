"""
Ionisation potential depression from RealQM's own partition.

Continuum-lowering theory assigns each ion a sphere of radius R0 holding enough
electrons to neutralise it -- the "ion sphere" -- and everyone acknowledges that
assignment as an approximation. In RealQM it is not an approximation: the
non-overlapping domains ARE the partition, and R0 follows from the density,
(4 pi / 3) R0^3 n_ion = 1.

So the depression is a confined-atom problem. For a hydrogenic ion of charge Z,

    -1/2 psi'' - (Z/r) psi = E psi   on 0 < r < R0

with the boundary condition at R0 set by what happens where two domains meet,
and

    Delta E(n) = E_confined(R0(n)) - E_free,     E_free = -Z^2/2 .

The whole point is that the answer depends on that boundary condition, which is
the same choice that makes the localisation constant vanish for a uniform gas
(see realqm_screening_note.md):

    Dirichlet  psi(R0) = 0     hard sphere; energy rises fast
    Neumann    psi'(R0) = 0    RealQM's free plane; rises slowly

RealQM therefore predicts LESS depression than a hard-sphere model at the same
density -- and the two standard theories differ in exactly that direction, with
Ecker-Kroell giving more depression than Stewart-Pyatt and the X-ray FEL
measurements on aluminium not having settled it. Here the interface condition is
the prediction, and there is data to check it against.

Atomic units throughout (hbar = m = e = 1, 4 pi eps0 = 1); energies in Hartree,
converted to eV for the comparison.

STATUS (2026-08-17): PARTIAL. Validated in the dilute limit, NOT usable dense.

  n_ion=1e21: Neumann 3.61 eV against Stewart-Pyatt 3.57 -- a real check, since
  nothing here is tuned. But the Neumann branch peaks near R0 ~ 3 a0 and then
  DECREASES (8.59 -> 8.12 -> 5.41 eV at 1e23, 1.8e23, 1e24), and a depression
  falling with density is not physics. Converged in N (stable from 2000 to
  20000 points), so it is the model and not the arithmetic.

  Cause: at small R0 the -Z/r singularity dominates the smeared background, so
  the single electron is pressed onto the nucleus and re-binds. Real pressure
  ionisation comes from crowding by the OTHER electrons, which a uniform
  background does not represent. Dirichlet hides this -- the imposed node
  forbids the collapse -- which is why it stays monotone and then overshoots
  both standard theories (83 eV against SP 62, EK 46 at 1e24).

  So no claim is made here about which theory RealQM favours. What the dense
  regime needs is the neighbouring electrons represented explicitly, as their
  own non-overlapping domains, rather than smeared into a background -- which is
  RealQM's actual content and is what makes it worth doing at all.

usage: python3 ipd_confined_ion.py [Z]
"""
import math
import sys

HA_EV = 27.211386245988
A0_CM = 0.529177210903e-8            # Bohr radius in cm


def solve_confined(Z, R, bc, N=4000, l=0, background=True):
    """Lowest eigenvalue of -1/2 u'' + V(r) u = E u on (0, R], u(0)=0.

    V(r) = -Z/r + (Z/2R)(3 - r^2/R^2) for r < R when `background` is on: the
    nucleus PLUS the uniform neutralising cloud of Z electrons filling the cell.
    That second term is the dominant one in every ion-sphere theory and is what
    lowers the continuum; omitting it gives a NEGATIVE depression, since
    truncating the domain then merely discards the tail's kinetic energy.
    Note V(R) = -Z/R + Z/R = 0, so the cell is neutral at its face and the
    continuum threshold sits at zero, as it should.

    u = r * psi.  The condition at R is on psi, so in terms of u:
        Dirichlet  psi(R)=0        ->  u(R) = 0
        Neumann    psi'(R)=0       ->  u'(R) = u(R)/R
    Finite differences plus bisection on the number of nodes; robust enough for
    a first pass and needs no external solver.
    """
    h = R / N
    r = [ (i + 1) * h for i in range(N) ]          # r_1 .. r_N, r_N = R

    def count_nodes(E):
        """Shoot outward; return (nodes, boundary residual)."""
        u_prev, u = 0.0, 1e-12                      # u(0)=0
        nodes = 0
        for i in range(1, N):
            V = -Z / r[i] + (l * (l + 1)) / (2 * r[i] ** 2)
            if background:
                V += (Z / (2.0 * R)) * (3.0 - (r[i] / R) ** 2)
            k = 2.0 * (V - E)
            u_next = 2.0 * u - u_prev + h * h * k * u
            if u_next * u < 0.0:
                nodes += 1
            u_prev, u = u, u_next
        if bc == 'dirichlet':
            resid = u                               # want u(R) = 0
        else:                                       # psi'(R) = 0  <=>  u'(R) = u(R)/R
            up = (u - u_prev) / h
            resid = up - u / R
        return nodes, resid

    # bracket the ground state: E below the well bottom, up to a large positive
    lo, hi = -0.6 * Z * Z - 50.0 / (R * R), 50.0 / (R * R) + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        nodes, resid = count_nodes(mid)
        # ground state has zero nodes and the residual changes sign through it
        if nodes > 0:
            hi = mid
        else:
            n_lo, r_lo = count_nodes(lo)
            if r_lo * resid < 0.0:
                hi = mid
            else:
                lo = mid
    return 0.5 * (lo + hi)


def R0_from_density(n_ion_cm3):
    """Wigner-Seitz radius in Bohr from ion number density in cm^-3."""
    n_au = n_ion_cm3 * A0_CM ** 3                  # ions per Bohr^3
    return (3.0 / (4.0 * math.pi * n_au)) ** (1.0 / 3.0)


def stewart_pyatt(Z, n_e_cm3, T_eV):
    """Stewart-Pyatt depression in eV (the standard interpolation)."""
    n_e_au = n_e_cm3 * A0_CM ** 3
    T_au = T_eV / HA_EV
    lam_D = math.sqrt(T_au / (4.0 * math.pi * n_e_au))         # Debye length, Bohr
    R0 = (3.0 / (4.0 * math.pi * n_e_au / max(Z, 1))) ** (1.0 / 3.0)
    s = R0 / lam_D
    dE = (3.0 * (Z + 1) / (2.0 * R0)) * (((1.0 + s ** 3) ** (2.0 / 3.0) - s ** 2)
                                         / (1.0 + s ** 2) ** 0 if False else 1.0)
    # standard SP form:
    dE = (3.0 * (Z + 1) / (2.0 * R0)) * ((1.0 + (lam_D / R0) ** 3) ** (2.0 / 3.0) - (lam_D / R0) ** 2)
    return dE * HA_EV


def ecker_kroell(Z, n_e_cm3):
    """Ecker-Kroell depression in eV, ion-sphere (strong-coupling) branch."""
    n_e_au = n_e_cm3 * A0_CM ** 3
    R0 = (3.0 / (4.0 * math.pi * n_e_au / max(Z, 1))) ** (1.0 / 3.0)
    return (Z + 1) / R0 * HA_EV                     # C_EK ~ 1 convention


def main():
    Z = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    E_free = -0.5 * Z * Z
    print(f"Hydrogenic ion, Z = {Z}.  Free ionisation energy = {-E_free*HA_EV:.2f} eV\n")
    print(f"{'n_ion (cm^-3)':>14} {'R0 (a0)':>9} "
          f"{'IPD Neumann':>13} {'IPD Dirichlet':>14} {'SP':>9} {'EK':>9}   (eV)")

    for n_ion in (1e21, 1e22, 1e23, 6e22, 1.8e23, 1e24):
        R0 = R0_from_density(n_ion)
        E_n = solve_confined(Z, R0, 'neumann')
        E_d = solve_confined(Z, R0, 'dirichlet')
        ipd_n = (E_n - E_free) * HA_EV
        ipd_d = (E_d - E_free) * HA_EV
        sp = stewart_pyatt(Z, n_ion * Z, 10.0)
        ek = ecker_kroell(Z, n_ion * Z)
        print(f"{n_ion:>14.2e} {R0:>9.3f} {ipd_n:>13.2f} {ipd_d:>14.2f} {sp:>9.2f} {ek:>9.2f}")

    print("\nThe discriminating fact: RealQM's free (Neumann) plane confines the bound")
    print("electron far more weakly than a hard sphere, so it predicts LESS depression.")
    print("Ecker-Kroell gives more depression than Stewart-Pyatt, and the X-ray FEL")
    print("measurements on aluminium have not settled which is right -- so the interface")
    print("condition becomes a prediction with data to check it against, which is exactly")
    print("what the uniform-gas screening analysis could not supply (there C = 0, so")
    print("there was no target at all).")


if __name__ == '__main__':
    main()
