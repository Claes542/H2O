"""
Offline port of the contrast model of cosmology_contrast_3d_gpu.html.

  d_t rho + div m                      = 0
  d_t m   + div(m u) + grad p + rho grad phi = -H m        (Hubble drag)
  d_t e   + div(e u) + p div u         = -3 H gamma e      (adiabatic cooling)
  lap phi = G (rho - rhobar)                                <-- CONTRAST source
  u = m/rho,  p = gamma e,  e = rho T

Matter is POSITIVE everywhere; only the gravitational source rho-rhobar changes sign.
Periodic box, FFT Poisson (exact, and automatically kills the zero mode -- which IS
the statement that the mean does not gravitate).  Upwind donor-cell transport.

Purpose: decide whether the observed dissolution of the web in coasting mode is
physical (growth freeze) or numerical (diffusion), by running the SAME seed with
expansion on and off.
"""
import numpy as np, json, sys

N        = int(sys.argv[1]) if len(sys.argv) > 1 else 96
MODE     = sys.argv[2] if len(sys.argv) > 2 else 'coasting'   # 'coasting' | 'none'
TEND     = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
SEED     = 12345

RHOBAR   = 1.0
GAMMA    = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2   # gamma_ad = 1 + gamma
G_GRAV   = 18.0
T0       = 0.01
SEED_AMP = 0.35
CORRCELL = 12.0 * N / 200.0   # seed correlation length, scaled from the N=200 default
H0       = 0.30           # coasting Hubble at t=0 (a = 1 + H0 t)
CFL      = 0.35
h        = 1.0 / N

rng = np.random.default_rng(SEED)

# ---------- seed: positive matter, zero-mean contrast ----------
def smoothed_field(n, corr):
    """Gaussian random field, correlation length `corr` cells, unit variance, zero mean."""
    w = rng.standard_normal((n, n, n))
    k = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    filt = np.exp(-0.5 * k2 * (corr / (2*np.pi))**2 * (2*np.pi/n)**2 * n**2 / n**2)
    filt = np.exp(-0.5 * k2 * (corr**2) / (n**2) * (np.pi**2))
    f = np.fft.ifftn(np.fft.fftn(w) * filt).real
    f -= f.mean()
    f /= f.std()
    return f

d   = smoothed_field(N, CORRCELL)
rho = np.maximum(RHOBAR * (1.0 + SEED_AMP * d), 0.02 * RHOBAR)
rho *= RHOBAR / rho.mean()                     # exact mean
e   = rho * T0
mx  = np.zeros_like(rho); my = np.zeros_like(rho); mz = np.zeros_like(rho)

# ---------- FFT Poisson on the contrast ----------
k    = np.fft.fftfreq(N) * N
KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
# discrete 7-point Laplacian eigenvalues
lap_eig = (2*np.cos(2*np.pi*KX/N) + 2*np.cos(2*np.pi*KY/N) + 2*np.cos(2*np.pi*KZ/N) - 6.0) / h**2
lap_eig[0, 0, 0] = 1.0                          # zero mode: source has zero mean by construction

def poisson(src):
    S = np.fft.fftn(src)
    S[0, 0, 0] = 0.0                            # the mean does not gravitate
    return np.fft.ifftn(S / lap_eig).real

def grad(f):
    return ((np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2*h),
            (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2*h),
            (np.roll(f, -1, 2) - np.roll(f, 1, 2)) / (2*h))

def upwind_div(q, ux, uy, uz):
    """div(q u) by donor-cell upwinding (diffusion-free in the sense of no centred smoothing)."""
    out = np.zeros_like(q)
    for ax, u in ((0, ux), (1, uy), (2, uz)):
        uf = 0.5 * (u + np.roll(u, -1, ax))                    # face velocity i+1/2
        qf = np.where(uf > 0, q, np.roll(q, -1, ax))           # donor cell
        F  = uf * qf                                           # flux at i+1/2
        out += (F - np.roll(F, 1, ax)) / h
    return out

# ---------- run ----------
t, step, a, H = 0.0, 0, 1.0, (H0 if MODE == 'coasting' else 0.0)
snapshots = []

def diagnostics(rho, t):
    m = rho.mean()
    deep  = 100.0 * (rho < 0.2 * m).mean()
    under = 100.0 * (rho < m).mean()
    drift = 100.0 * (m - RHOBAR) / RHOBAR
    # smooth (2 box passes) for morphology, as the browser does
    rp = rho.copy()
    for _ in range(2):
        rp = (rp + np.roll(rp,1,0)+np.roll(rp,-1,0)+np.roll(rp,1,1)
              + np.roll(rp,-1,1)+np.roll(rp,1,2)+np.roll(rp,-1,2)) / 7.0
    mean = rp.mean()
    hxx = (np.roll(rp,-1,0)-2*rp+np.roll(rp,1,0))
    hyy = (np.roll(rp,-1,1)-2*rp+np.roll(rp,1,1))
    hzz = (np.roll(rp,-1,2)-2*rp+np.roll(rp,1,2))
    hrms = np.sqrt(np.mean(hxx**2 + hyy**2 + hzz**2))
    # median |curvature| over OVERDENSE cells: robust to the few extreme peaks that
    # dominate an RMS once peak/mean is large (the RMS threshold mislabels ridges
    # as 'diffuse' exactly in the clumped regime we care about).
    over0 = rp > rp.mean()
    hmed = np.median(np.abs(np.concatenate([hxx[over0], hyy[over0], hzz[over0]])))
    eps  = 0.15 * hmed
    def d2(f, a1, a2):
        return (np.roll(np.roll(f,-1,a1),-1,a2) - np.roll(np.roll(f,-1,a1),1,a2)
                - np.roll(np.roll(f,1,a1),-1,a2) + np.roll(np.roll(f,1,a1),1,a2)) / 4.0
    hxy, hxz, hyz = d2(rp,0,1), d2(rp,0,2), d2(rp,1,2)
    over = rp > mean
    Hm = np.empty(rp.shape + (3,3))
    Hm[...,0,0]=hxx; Hm[...,1,1]=hyy; Hm[...,2,2]=hzz
    Hm[...,0,1]=Hm[...,1,0]=hxy; Hm[...,0,2]=Hm[...,2,0]=hxz; Hm[...,1,2]=Hm[...,2,1]=hyz
    ev = np.linalg.eigvalsh(Hm[over])
    nneg = (ev < -eps).sum(axis=1)
    nov = max(1, over.sum())
    node = 100.0*(nneg==3).sum()/nov; fil = 100.0*(nneg==2).sum()/nov
    wall = 100.0*(nneg==1).sum()/nov; diff = 100.0*(nneg==0).sum()/nov
    flat = np.sort(rho.ravel())[::-1]
    top5 = max(1, int(0.05*flat.size))
    return dict(t=round(t,3), deep=round(deep,1), under=round(under,1), drift=round(drift,3),
                peak_over_mean=round(float(rho.max()/m),2),
                top5_mass=round(100.0*float(flat[:top5].sum()/flat.sum()),1),
                node=round(node,1), fil=round(fil,1), wall=round(wall,1), diffuse=round(diff,1),
                rho_min=round(float(rho.min()),4))

snapshots.append(diagnostics(rho, t))
next_report = TEND / 6.0

while t < TEND and step < 200000:
    ux, uy, uz = mx/rho, my/rho, mz/rho
    cs = np.sqrt((1.0+GAMMA) * GAMMA * e / rho)
    vmax = float(np.max(np.sqrt(ux**2+uy**2+uz**2) + cs))
    dt = CFL * h / max(vmax, 1e-6)
    dt = min(dt, 0.01)

    phi = poisson(G_GRAV * (rho - rho.mean()) / a**2)
    gx, gy, gz = grad(phi)
    p = GAMMA * e
    px, py, pz = grad(p)

    rho_n = rho - dt * upwind_div(rho, ux, uy, uz) / a
    mx_n  = mx - dt * (upwind_div(mx, ux, uy, uz)/a + px/a + rho*gx/a + H*mx)
    my_n  = my - dt * (upwind_div(my, ux, uy, uz)/a + py/a + rho*gy/a + H*my)
    mz_n  = mz - dt * (upwind_div(mz, ux, uy, uz)/a + pz/a + rho*gz/a + H*mz)
    divu  = (grad(ux)[0] + grad(uy)[1] + grad(uz)[2]) / a
    e_n   = e - dt * (upwind_div(e, ux, uy, uz)/a + GAMMA*e*divu + 3.0*H*GAMMA*e)

    rho = np.maximum(rho_n, 1e-4)
    e   = np.maximum(e_n, 1e-8)
    mx, my, mz = mx_n, my_n, mz_n

    t += dt; step += 1
    if MODE == 'coasting':
        a = 1.0 + H0 * t
        H = H0 / a
    if not np.isfinite(rho).all():
        print("NaN at t=", t); break
    if t >= next_report:
        snapshots.append(diagnostics(rho, t))
        next_report += TEND / 6.0

snapshots.append(diagnostics(rho, t))
print(json.dumps(dict(N=N, mode=MODE, steps=step, t_end=round(t,3), snapshots=snapshots), indent=1))
