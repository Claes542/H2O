import numpy as np
# 3D periodic box. Proton (+1) central slab, electron surrounds it (net neutral).
# Ripple the p-e interface with amplitude A, mode n (wavelength L/n) in x; measure dE = E(A)-E(0).
# dE<0 => ripple grows => Neumann free boundary UNSTABLE ; dE>0 => stable.
# RealQM energy: von Weizsaecker kinetic (surface tension) + cross-Coulomb (NO self-energy).
N=64; L=20.0; h=L/N; w=1.0; A=0.6
x=np.linspace(0,L,N,endpoint=False)
X,Y,Z=np.meshgrid(x,x,x,indexing='ij')
k=2*np.pi*np.fft.fftfreq(N,d=h); KX,KY,KZ=np.meshgrid(k,k,k,indexing='ij')
K2=KX**2+KY**2+KZ**2; K2[0,0,0]=1.0
def poisson(rho):
    pk=4*np.pi*np.fft.fftn(rho)/K2; pk[0,0,0]=0
    return np.real(np.fft.ifftn(pk))
def dens(A,n):
    hw = L/4 + A*np.cos(2*np.pi*n*X/L)          # rippled half-width of central slab
    tp = (1-np.tanh((np.abs(Z-L/2)-hw)/w))/2    # proton in central slab
    te = 1.0-tp                                  # electron outside (continuous, Neumann-like)
    tp/= tp.sum()*h**3; te/= te.sum()*h**3
    return tp,te
def grad2(f):
    gx,gy,gz=np.gradient(f,h); return gx**2+gy**2+gz**2
def energy(A,n,like,mp=1.0):
    rp,re=dens(A,n)
    T = (0.5/mp)*np.sum(grad2(np.sqrt(rp+1e-30)))*h**3 + 0.5*np.sum(grad2(np.sqrt(re+1e-30)))*h**3
    phip=poisson(rp)
    Ees = (+1 if like else -1)*np.sum(re*phip)*h**3   # opposite: attractive(-), like: repulsive(+)
    return T+Ees, T, Ees
E0,_,_=energy(0,1,False)
print(f"Flat interface reference E0 = {E0:.4f}\n")
print(f"{'mode n':>6} {'lambda':>7} | {'OPPOSITE dE':>12} {'(dT':>7} {'dEes)':>8} | {'LIKE dE':>9}")
for n in [1,2,3,4,6,8]:
    Eo,To,eo=energy(A,n,False); dTo=To-energy(0,n,False)[1]
    El,_,_=energy(A,n,True)
    dEo=Eo-energy(0,n,False)[0]; dEl=El-energy(0,n,True)[0]
    dEes=eo-energy(0,n,False)[2]
    print(f"{n:>6} {L/n:7.1f} | {dEo:12.4f} {dTo:7.3f} {dEes:8.3f} | {dEl:9.4f}")
print("\ndE<0 => that ripple lowers energy => grows => interface UNSTABLE.")
