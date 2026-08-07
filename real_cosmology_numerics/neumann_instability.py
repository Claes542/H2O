import numpy as np
from scipy.special import erf
# "Boundary force" on a Neumann+cont p-e interface. Two unit clouds, +1 and -1, Gaussian
# widths s_p, s_e, centres separated by d. Cross-Coulomb (the only d-dependent term; RealQM has
# no self-energy, and localisation energy is d-independent for fixed sizes):
#   U(d) = -erf( d / (sqrt(2)*S) ) / d ,   S = sqrt(s_p^2 + s_e^2)   (attraction, Hartree-like units)
# STABLE free boundary needs an interior minimum: dU/dd = 0 with U''>0.
def U(d,S):  return -erf(d/(np.sqrt(2)*S))/np.where(d==0,1e-12,d)
def U0(S):   return -np.sqrt(2/np.pi)/S    # contact value d->0
print("Opposite charges: is there an interior equilibrium of the boundary (dU/dd=0)?")
print(f"{'s_p=s_e':>8} {'U(contact)':>11} {'U(d=2)':>9} {'min at':>10}")
for s in [0.05, 0.2, 0.5, 1.0, 2.0]:
    S=np.sqrt(2)*s
    d=np.linspace(1e-4,15,60000); u=U(d,S)
    imin=np.argmin(u)
    where = 'd->0 (contact)' if imin<3 else f'd={d[imin]:.2f}'
    print(f"{s:8.2f} {U0(S):11.3f} {U(2.0,S):9.3f} {where:>14}")
print("\n-> for opposite charges U(d) is monotone: the minimum is always at CONTACT (d->0).")
print("   No interior equilibrium => NO restoring force => the Neumann free boundary is unstable")
print("   at ALL sizes; it runs to contact. The only thing that ever stops it is either")
print("   (a) a POINT proton (contact = the electron centring on a zero-size charge = the stable")
print("       Kato-cusp atom, no overlap conflict), or (b) the hard non-overlap wall for a FINITE")
print("       proton -- i.e. the equilibrium is pinned AT the constraint, not at a smooth minimum.")
print("\nLike charges (sign flip) give U(d)=+erf/d, monotone DECREASING: force is repulsive =")
print("restoring => the free boundary IS stable. Sign of the cross-interface force decides it.")
