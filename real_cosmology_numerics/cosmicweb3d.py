import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
# 3D +/- mass toy, STABLE semi-Lagrangian. rho=+/- mass; like attract, unlike repel via a=-sign(rho)grad(psi),
# lap(psi)=rho. Velocity capped to < 0.5 cell/step; mass pinned; mild smoothing for stability. Stop at
# intermediate time (filamentary/segregated stage, before point-collapse).
N=96; L=1.0; h=L/N
k=2*np.pi*np.fft.fftfreq(N,d=h); KX,KY,KZ=np.meshgrid(k,k,k,indexing='ij')
K2=KX**2+KY**2+KZ**2; K2[0,0,0]=1.0
def poisson(rho): pk=np.fft.fftn(rho)/(-K2); pk[0,0,0]=0; return np.real(np.fft.ifftn(pk))
def grad(f):
    fk=np.fft.fftn(f)
    return (np.real(np.fft.ifftn(1j*KX*fk)),np.real(np.fft.ifftn(1j*KY*fk)),np.real(np.fft.ifftn(1j*KZ*fk)))
rng=np.random.default_rng(7)
rho=np.real(np.fft.ifftn(np.fft.fftn(rng.standard_normal((N,N,N)))*np.exp(-K2*(0.045*L)**2)))
rho-=rho.mean(); rho/=np.abs(rho).std(); M0=rho.sum()
I,J,Kk=np.meshgrid(np.arange(N),np.arange(N),np.arange(N),indexing='ij')
STEPS=350
for s in range(STEPS):
    psi=poisson(rho); gx,gy,gz=grad(psi); sr=np.sign(rho)
    vx,vy,vz=-sr*gx,-sr*gy,-sr*gz
    vmax=np.sqrt(vx*vx+vy*vy+vz*vz).max()+1e-12
    dt=0.4*h/vmax                              # <0.4 cell/step -> stable, accurate
    cx=(I-vx*dt/h); cy=(J-vy*dt/h); cz=(Kk-vz*dt/h)   # backtrace (cell units)
    rho=map_coordinates(rho,[cx,cy,cz],order=1,mode='wrap')
    if s%18==0: rho=gaussian_filter(rho,0.5,mode="wrap")   # mild regularization
    rho-=(rho.sum()-M0)/rho.size
pos=np.clip(rho,0,None).sum(axis=2); neg=np.clip(-rho,0,None).sum(axis=2)
def norm(a): a=a-a.min(); return a/(a.max()+1e-12)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
img=np.zeros((N,N,3)); img[...,0]=norm(pos); img[...,2]=norm(neg); img[...,1]=0.12*norm(pos)
plt.figure(figsize=(6,6),dpi=140); plt.imshow(np.transpose(img,(1,0,2)),origin='lower'); plt.axis('off')
plt.title('3D two-valued-Poisson cosmology (column projection)\nred = positive-mass web, blue = negative-mass voids',fontsize=9)
plt.tight_layout(); plt.savefig('cosmic_web_3d.png',bbox_inches='tight')
am=np.abs(rho).ravel(); thr=np.quantile(am,0.90)
print(f"grid {N}^3, steps {STEPS}; rho range [{rho.min():.2f},{rho.max():.2f}]; mass drift {rho.sum()-M0:.1e}")
print(f"|mass| in top-10%% cells: {am[am>=thr].sum()/am.sum():.2f} (uniform 0.10 -> higher = clumped)")
print("saved cosmic_web_3d.png")
