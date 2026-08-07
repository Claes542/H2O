import numpy as np
from scipy.optimize import minimize_scalar

# Minimal independent model (NOT the production RealQM solver):
# electron 1s trial psi ~ exp(-r/a), proton = uniformly charged sphere radius Rp (charge +1).
# Answers: how does H binding soften as the proton's charge radius Rp grows toward a0?
# All in atomic units (Hartree, Bohr). Point-proton H: E=-0.5 Ha, a=1 (a0).

def Vexp(a, Rp, n=200000):
    # <V> for normalized 1s over a uniformly charged sphere potential
    # |psi|^2 = exp(-2r/a)/(pi a^3);  integrate 4 pi r^2 |psi|^2 V(r) dr
    rmax = max(30*a, 5*Rp)
    r = np.linspace(1e-9, rmax, n)
    dens = np.exp(-2*r/a)/(np.pi*a**3)
    V = np.where(r > Rp, -1.0/r, -(3.0/(2*Rp) - r**2/(2*Rp**3)))
    return np.trapz(4*np.pi*r**2*dens*V, r)

def Ebind(Rp):
    # minimize E(a) = 1/(2 a^2) + <V>(a,Rp) over a
    f = lambda a: 0.5/a**2 + Vexp(a, Rp)
    res = minimize_scalar(f, bounds=(0.2, 60), method='bounded')
    return res.fun, res.x  # E, optimal electron size a

print(f"{'Rp/a0':>8} {'E_bind(Ha)':>12} {'E/E_H(%)':>10} {'a_e/a0':>8} {'a_e/Rp':>8}")
for Rp in [1e-5, 0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    E, a = Ebind(Rp)
    print(f"{Rp:8.5g} {E:12.4f} {100*E/-0.5:10.1f} {a:8.3f} {a/Rp:8.2f}")

# point-proton reference
E0,a0 = Ebind(1e-6)
print(f"\npoint proton: E={E0:.4f} Ha  a={a0:.3f} a0   (exact -0.5, 1.0)")
print("real proton: Rp ~ 1 fm = 1.9e-5 a0  ->  finite-size effect utterly negligible")
