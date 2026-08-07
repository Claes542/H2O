import numpy as np
from scipy.linalg import eigh_tridiagonal
# u=r*psi, H=-1/2 u'' -(1/r)u, Dirichlet u=0 at r=Rp and r=rmax. Rp->0 => standard H (-0.5).
def Eg(Rp, rmax=60.0, N=3000):
    r = np.linspace(Rp if Rp>0 else 0.0, rmax, N+2)[1:-1]
    h = r[1]-r[0]
    d  = 1.0/h**2 - 1.0/r
    e  = -0.5/h**2*np.ones(N-1)
    return eigh_tridiagonal(d, e, select='i', select_range=(0,0))[0][0]
print(f"{'Rp/a0':>7} {'E(Ha)':>9} {'E(eV)':>8} {'%H':>6}")
for Rp in [0.0,1e-3,0.01,0.05,0.1,0.3,0.5,1.0,2.0]:
    E=Eg(Rp); print(f"{Rp:7.3g} {E:9.4f} {E*27.211:8.2f} {100*E/-0.5:6.1f}")
