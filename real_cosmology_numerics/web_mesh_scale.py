"""
What sets the mesh size of the web?

Candidates: (a) the Jeans length lam_J = 2 pi sqrt(gamma_s T0) / sqrt(G rhobar),
(b) the seed correlation length, (c) neither -- coarsening washes both out.

Measure, don't assert: run the contrast model to the web stage and read the mesh
off the spherically averaged power spectrum of delta = rho/rhobar - 1, as
lam_peak = 2 pi / k_peak (k in box^-1). Vary G (which moves lam_J as G^-1/2 at
fixed T0) and vary the seed correlation length independently.

If (a): lam_peak tracks lam_J and ignores the seed.
If (b): lam_peak tracks the seed and ignores G.

Usage: python3 web_mesh_scale.py [N] [t_measure]
"""
import numpy as np, sys, json

N     = int(sys.argv[1]) if len(sys.argv) > 1 else 96
TMEAS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
SEED  = 12345
RHOBAR, T0, SEED_AMP, H0, CFL = 1.0, 0.01, 0.35, 0.30, 0.35
h = 1.0 / N

k1 = np.fft.fftfreq(N) * N
KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing='ij')
k2 = KX**2 + KY**2 + KZ**2
kmag = np.sqrt(k2)
lap_eig = (2*np.cos(2*np.pi*KX/N) + 2*np.cos(2*np.pi*KY/N) + 2*np.cos(2*np.pi*KZ/N) - 6.0) / h**2
lap_eig[0,0,0] = 1.0

def poisson(src):
    S = np.fft.fftn(src); S[0,0,0] = 0.0
    return np.fft.ifftn(S / lap_eig).real

def grad(q):
    return ((np.roll(q,-1,0)-np.roll(q,1,0))/(2*h),
            (np.roll(q,-1,1)-np.roll(q,1,1))/(2*h),
            (np.roll(q,-1,2)-np.roll(q,1,2))/(2*h))

def upwind_div(q, ux, uy, uz):
    out = np.zeros_like(q)
    for ax, u in ((0,ux),(1,uy),(2,uz)):
        uf = 0.5*(u + np.roll(u,-1,ax))
        F  = uf * np.where(uf > 0, q, np.roll(q,-1,ax))
        out += (F - np.roll(F,1,ax))/h
    return out

def mesh_scale(rho):
    """Peak of the spherically averaged P(k) of the density contrast -> mesh length."""
    d = rho/rho.mean() - 1.0
    P = np.abs(np.fft.fftn(d))**2
    kb = kmag.astype(int).ravel()
    Ps = np.bincount(kb, weights=P.ravel())
    Nk = np.bincount(kb)
    good = Nk > 0
    Pk = Ps[good]/Nk[good]
    kk = np.arange(len(Ps))[good]
    m = (kk >= 1) & (kk <= N//2)
    kk, Pk = kk[m], Pk[m]
    # power per logarithmic interval, k^3 P(k): its peak is the dominant scale
    w = kk**3 * Pk
    kpk = kk[np.argmax(w)]
    return float(1.0/kpk), float(kpk)      # wavelength in box units

def run(G, corrcell, tmeas=TMEAS):
    rng = np.random.default_rng(SEED)
    w = rng.standard_normal((N,N,N))
    f = np.fft.ifftn(np.fft.fftn(w)*np.exp(-0.5*k2*corrcell**2/N**2*np.pi**2)).real
    f -= f.mean(); f /= f.std()
    rho = np.maximum(RHOBAR*(1.0+SEED_AMP*f), 0.02*RHOBAR); rho *= RHOBAR/rho.mean()
    e = rho*T0
    mx = np.zeros_like(rho); my = np.zeros_like(rho); mz = np.zeros_like(rho)
    GAMMA = 0.4
    lamJ = 2*np.pi*np.sqrt(GAMMA*T0)/np.sqrt(G*RHOBAR)
    t, a, H, step = 0.0, 1.0, H0, 0
    while t < tmeas and step < 200000:
        ux, uy, uz = mx/rho, my/rho, mz/rho
        cs = np.sqrt((1+GAMMA)*GAMMA*e/rho)
        dt = min(CFL*h/max(float(np.max(np.sqrt(ux**2+uy**2+uz**2)+cs)),1e-6), 0.01)
        phi = poisson(G*(rho-rho.mean())/a**2)
        gx, gy, gz = grad(phi); px, py, pz = grad(GAMMA*e)
        rho_n = rho - dt*upwind_div(rho,ux,uy,uz)/a
        mx_n = mx - dt*(upwind_div(mx,ux,uy,uz)/a + px/a + rho*gx/a + H*mx)
        my_n = my - dt*(upwind_div(my,ux,uy,uz)/a + py/a + rho*gy/a + H*my)
        mz_n = mz - dt*(upwind_div(mz,ux,uy,uz)/a + pz/a + rho*gz/a + H*mz)
        divu = (grad(ux)[0]+grad(uy)[1]+grad(uz)[2])/a
        e_n = e - dt*(upwind_div(e,ux,uy,uz)/a + GAMMA*e*divu + 3.0*H*GAMMA*e)
        rho, e = np.maximum(rho_n,1e-4), np.maximum(e_n,1e-8)
        mx, my, mz = mx_n, my_n, mz_n
        t += dt; step += 1; a = 1.0+H0*t; H = H0/a
    lam, kpk = mesh_scale(rho)
    return dict(G=G, corr_cells=corrcell, lam_J=round(lamJ,4), lam_J_cells=round(lamJ*N,1),
                lam_mesh=round(lam,4), lam_mesh_cells=round(lam*N,1),
                ratio_mesh_over_J=round(lam/lamJ,2), t=round(t,2),
                peak_over_mean=round(float(rho.max()/rho.mean()),1))

cases = []
# (a) vary G at fixed seed: lam_J moves as G^-1/2
for G in (9.0, 18.0, 36.0, 72.0):
    cases.append(run(G, 12.0*N/200.0))
    print(json.dumps(cases[-1]), flush=True)
# (b) vary seed correlation at fixed G
for corr in (5.0, 10.0, 20.0):
    cases.append(run(18.0, corr))
    print(json.dumps(cases[-1]), flush=True)

print()
print(f"{'G':>6} {'seed(cells)':>12} {'lamJ(cells)':>12} {'mesh(cells)':>12} {'mesh/lamJ':>10} {'peak/mean':>10}")
for c in cases:
    print(f"{c['G']:>6} {c['corr_cells']:>12.1f} {c['lam_J_cells']:>12.1f} "
          f"{c['lam_mesh_cells']:>12.1f} {c['ratio_mesh_over_J']:>10} {c['peak_over_mean']:>10}")
