import numpy as np
# 3D gradient-flow (imaginary time) relaxation. Electron core (-1) INSIDE a proton shell (+1),
# Neumann+cont free boundary (no imposed nodes), non-overlap penalty. RealQM energy (no self-energy).
# Question: does it settle to a stable bound core-shell, or collapse/finger (unstable)?
N=48; L=16.0; h=L/N
x=np.linspace(0,L,N,endpoint=False); X,Y,Z=np.meshgrid(x,x,x,indexing='ij')
R=np.sqrt((X-L/2)**2+(Y-L/2)**2+(Z-L/2)**2)
k=2*np.pi*np.fft.fftfreq(N,d=h); KX,KY,KZ=np.meshgrid(k,k,k,indexing='ij')
K2=KX**2+KY**2+KZ**2; K2[0,0,0]=1.0
def poisson(rho):
    pk=4*np.pi*np.fft.fftn(rho)/K2; pk[0,0,0]=0; return np.real(np.fft.ifftn(pk))
def lap(f):
    fk=np.fft.fftn(f); return np.real(np.fft.ifftn(-K2*fk))
def norm(psi):
    return psi/np.sqrt(np.sum(psi**2)*h**3)
def run(mp=1.0, lam=6.0, steps=400, dt=0.004, perturb=False):
    pe=norm(np.exp(-R**2/(2*1.5**2)))                      # electron core (inside)
    shell=np.exp(-(R-3.0)**2/(2*1.0**2))                   # proton shell at r~3
    if perturb: shell*=(1+0.3*(X-L/2)/ (R+1e-6))           # dipole (l=1) kick to test asymmetry
    pp=norm(shell)
    E=[]
    for s in range(steps):
        rho_e=pe**2; rho_p=pp**2
        phi_p=poisson(rho_p); phi_e=poisson(rho_e)
        # energies
        Te=0.5*np.sum((np.gradient(pe,h)[0]**2+np.gradient(pe,h)[1]**2+np.gradient(pe,h)[2]**2))*h**3
        Tp=(0.5/mp)*np.sum((np.gradient(pp,h)[0]**2+np.gradient(pp,h)[1]**2+np.gradient(pp,h)[2]**2))*h**3
        Ees=-np.sum(rho_e*phi_p)*h**3; Eov=lam*np.sum(rho_e*rho_p)*h**3
        E.append(Te+Tp+Ees+Eov)
        # gradient-flow steps
        pe=pe-dt*(-lap(pe)-2*pe*phi_p+2*lam*pe*rho_p); pe=norm(pe)
        pp=pp-dt*(-(1/mp)*lap(pp)-2*pp*phi_e+2*lam*pp*rho_e); pp=norm(pp)
    # final radii (mean r of each density)
    re=np.sum(R*pe**2)*h**3; rp=np.sum(R*pp**2)*h**3
    ov=np.sum(pe**2*pp**2)*h**3
    return E, re, rp, ov
for mp,tag in [(1.0,"equal-mass (big proton)"),(1836.0,"heavy proton (control)")]:
    E,re,rp,ov=run(mp=mp)
    print(f"{tag:26s}: E {E[0]:+.3f} -> {E[-1]:+.3f} | <r>_e={re:.2f} <r>_p={rp:.2f} overlap={ov:.3f}")
    conv = "CONVERGED (stable)" if abs(E[-1]-E[-2])<1e-4 and E[-1]<E[0] else "not settled"
    print(f"{'':26s}  {conv}; electron {'INSIDE' if re<rp else 'OUTSIDE'} proton shell")
