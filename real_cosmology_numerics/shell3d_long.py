import numpy as np
N=48; L=16.0; h=L/N
x=np.linspace(0,L,N,endpoint=False); X,Y,Z=np.meshgrid(x,x,x,indexing='ij')
R=np.sqrt((X-L/2)**2+(Y-L/2)**2+(Z-L/2)**2)
k=2*np.pi*np.fft.fftfreq(N,d=h); KX,KY,KZ=np.meshgrid(k,k,k,indexing='ij')
K2=KX**2+KY**2+KZ**2; K2[0,0,0]=1.0
def poisson(rho):
    pk=4*np.pi*np.fft.fftn(rho)/K2; pk[0,0,0]=0; return np.real(np.fft.ifftn(pk))
def lap(f): return np.real(np.fft.ifftn(-K2*np.fft.fftn(f)))
def norm(psi): return psi/np.sqrt(np.sum(psi**2)*h**3)
def g2(f): gx,gy,gz=np.gradient(f,h); return gx**2+gy**2+gz**2
def run(mp=1.0,lam=6.0,steps=2500,dt=0.004):
    pe=norm(np.exp(-R**2/(2*1.5**2))); pp=norm(np.exp(-(R-3.0)**2/(2*1.0**2)))
    for s in range(steps):
        rho_e=pe**2; rho_p=pp**2; phi_p=poisson(rho_p); phi_e=poisson(rho_e)
        if s%500==0 or s==steps-1:
            E=0.5*np.sum(g2(pe))*h**3+(0.5/mp)*np.sum(g2(pp))*h**3-np.sum(rho_e*phi_p)*h**3+lam*np.sum(rho_e*rho_p)*h**3
            re=np.sum(R*rho_e)*h**3; rp=np.sum(R*rho_p)*h**3
            print(f"   step {s:5d}: E={E:+.4f}  <r>_e={re:.2f} <r>_p={rp:.2f} overlap={np.sum(rho_e*rho_p)*h**3:.4f}")
        pe=norm(pe-dt*(-lap(pe)-2*pe*phi_p+2*lam*pe*rho_p))
        pp=norm(pp-dt*(-(1/mp)*lap(pp)-2*pp*phi_e+2*lam*pp*rho_e))
print("EQUAL-MASS big proton (electron core inside proton shell, Neumann):")
run(mp=1.0)
print("HEAVY proton control:")
run(mp=1836.0)
