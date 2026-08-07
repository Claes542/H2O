import numpy as np
N=64; L=20.0; h=L/N
x=np.linspace(0,L,N,endpoint=False); X,Y,Z=np.meshgrid(x,x,x,indexing='ij')
k=2*np.pi*np.fft.fftfreq(N,d=h); KX,KY,KZ=np.meshgrid(k,k,k,indexing='ij')
K2=KX**2+KY**2+KZ**2; K2[0,0,0]=1.0
def poisson(rho):
    pk=4*np.pi*np.fft.fftn(rho)/K2; pk[0,0,0]=0; return np.real(np.fft.ifftn(pk))
def blob(cx,cy,cz,s):
    r2=(X-cx)**2+(Y-cy)**2+(Z-cz)**2; g=np.exp(-r2/(2*s**2)); return g/(g.sum()*h**3)
# (1) TRANSLATION/MERGER test: +blob fixed at centre, -blob approaching. E_es(d) and total.
print("Translation (merger) mode: proton fixed, electron centre at distance d")
print(f"{'d':>5} {'E_es(attr)':>11} {'kin(fixed)':>10} {'E_tot':>9}")
s=2.0; rp=blob(L/2,L/2,L/2,s); phip=poisson(rp)
def grad2(f):
    gx,gy,gz=np.gradient(f,h); return gx**2+gy**2+gz**2
Tp=0.5*np.sum(grad2(np.sqrt(rp+1e-30)))*h**3
for d in [8,6,4,2,1,0.5,0.01]:
    re=blob(L/2+d,L/2,L/2,s)
    Ees=-np.sum(re*phip)*h**3
    Te=0.5*np.sum(grad2(np.sqrt(re+1e-30)))*h**3
    print(f"{d:5.2f} {Ees:11.4f} {Tp+Te:10.4f} {Tp+Te+Ees:9.4f}")
print("\n-> confirms electrostatics ALIVE and attractive; but with fixed sizes kinetic is constant,")
print("   so E_tot just tracks E_es: min at contact (d->0) = they bind into an adjacent pair.")
# (2) ripple electrostatic response with full precision
print("\nInterface ripple, full-precision electrostatic change dEes and kinetic dT:")
w=1.0; A=0.6
def dens(A,n):
    hw=L/4+A*np.cos(2*np.pi*n*X/L); tp=(1-np.tanh((np.abs(Z-L/2)-hw)/w))/2; te=1-tp
    return tp/(tp.sum()*h**3), te/(te.sum()*h**3)
def parts(A,n):
    rp,re=dens(A,n); T=0.5*np.sum(grad2(np.sqrt(rp+1e-30))+grad2(np.sqrt(re+1e-30)))*h**3
    return T, -np.sum(re*poisson(rp))*h**3
T0,e0=parts(0,1)
for n in [1,2,4]:
    T,e=parts(A,n); print(f"  n={n}: dT={T-T0:+.5f}  dEes={e-e0:+.5f}  net={ (T-T0)+(e-e0):+.5f}")
