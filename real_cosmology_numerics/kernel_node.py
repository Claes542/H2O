import numpy as np
from scipy.optimize import minimize_scalar
# Spherically symmetric electron, R(r), forced R(0)=0 (density zero at the kernel).
# Flexible hollow trial: R(r) = (1 - exp(-r/c)) * exp(-r/a).  c = hole radius, a = cloud size.
# Energy in Hartree.  <T>=1/2 * 4pi ∫ (R')^2 r^2 dr ,  <V>=-4pi ∫ R^2 r dr ,  norm=4pi ∫ R^2 r^2 dr.
r = np.linspace(1e-7, 60, 400000)
def E(a, c):
    R = (1-np.exp(-r/c))*np.exp(-r/a)
    dR = np.gradient(R, r)
    norm = np.trapz(R**2 * r**2, r)
    T = 0.5*np.trapz(dR**2 * r**2, r)/norm
    V = -np.trapz(R**2 * r, r)/norm
    return T+V
def Emin_a(c):
    res = minimize_scalar(lambda a: E(a,c), bounds=(0.3,6), method='bounded')
    return res.fun, res.x

print("Fixed hole radius c (genuine hollow electron, R(0)=0):")
print(f"{'c(a0)':>7} {'E(Ha)':>9} {'E(eV)':>8} {'a(a0)':>7}")
for c in [2.0, 1.0, 0.5, 0.2, 0.1, 0.03, 0.01, 0.003]:
    e,a = Emin_a(c); print(f"{c:7.3g} {e:9.4f} {e*27.211:8.2f} {a:7.3f}")
print("\n-> as the hole c shrinks, E -> -0.5 Ha (=-13.6 eV): a node at the KERNEL POINT costs nothing.")
print("   the single-parameter r*exp(-r/a) hollow (c=a, fully hollow) gives:")
res = minimize_scalar(lambda a: E(a,a), bounds=(0.3,3), method='bounded')
print(f"   E = {res.fun:.4f} Ha = {res.fun*27.211:.2f} eV  (a={res.x:.3f}) -- a genuinely hollow shell")
