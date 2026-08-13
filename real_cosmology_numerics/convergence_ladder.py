"""
Convergence ladder for the contrast model: is the arrest physical or the mesh?

The paper's criterion is that a small-viscosity solution is what SURVIVES as the
viscosity is decreased, and here the viscosity is the discretisation -- so the
criterion is tested by refining N. All runs use the SAME seed field (generated at
the finest N and coarsened by block-averaging, so the physical seed is identical
rather than merely statistically similar) and are compared at the SAME time.

Watch: peak/mean at fixed t. If it keeps climbing with N, the bounded peak/mean
reported at gamma_ad = 1.4 is the grid, not the Jeans-mass arrest. Also reports
lam_J in CELLS at the densest point, against the Truelove >= 4 rule of thumb.

Usage: python3 convergence_ladder.py [t_measure] [N1,N2,...]
"""
import numpy as np, sys, json

TMEAS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
NLIST = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['64','96','128','192'])]
SEEDTYPE = sys.argv[3] if len(sys.argv) > 3 else 'gauss'   # 'gauss' (one scale) | 'power' (P ~ k)
NFINE = max(NLIST)
SEED  = 12345
RHOBAR, GAMMA, G_GRAV, T0 = 1.0, 0.4, 18.0, 0.01
SEED_AMP, H0, CFL = 0.35, 0.30, 0.35
CORR_BOX = 12.0 / 200.0            # seed correlation length in BOX units, N-independent

def seed_field(NF):
    """'gauss': one correlation length (a preferred scale -- the web then inherits it).
       'power': the SCALE-FREE seed the paper actually argues for, P_phi ~ k^-3 giving
       P_rho ~ k, i.e. |delta_k| ~ k^(1/2). No seed scale exists, so the only length
       left in the problem is the Jeans length."""
    rng = np.random.default_rng(SEED)
    w = rng.standard_normal((NF, NF, NF))
    k = np.fft.fftfreq(NF) * NF
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    kmag = np.sqrt(k2); kmag[0,0,0] = 1.0
    if SEEDTYPE == 'power':
        amp = kmag**0.5                                   # P(k) = |delta_k|^2 ~ k
        amp *= np.exp(-0.5*(kmag/(0.35*NF))**2)           # de-alias at the grid scale only
        amp[0,0,0] = 0.0
        f = np.fft.ifftn(np.fft.fftn(w)*amp).real
    else:
        corr_cells = CORR_BOX * NF
        f = np.fft.ifftn(np.fft.fftn(w)*np.exp(-0.5*k2*corr_cells**2/NF**2*np.pi**2)).real
    f -= f.mean(); f /= f.std()
    return f

def coarsen(f, N):
    """Restrict the fine seed to N^3 by Fourier truncation: keeps the identical
    physical field on every grid (block-averaging would need N | NFINE)."""
    NF = f.shape[0]
    if NF == N: return f.copy()
    F = np.fft.fftn(f)
    hN = N // 2
    idx = list(range(0, hN)) + list(range(NF - hN, NF))
    return np.fft.ifftn(F[np.ix_(idx, idx, idx)]).real

FINE = seed_field(NFINE)

def run(N, tmeas):
    h = 1.0/N
    k = np.fft.fftfreq(N)*N
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    lap = (2*np.cos(2*np.pi*KX/N)+2*np.cos(2*np.pi*KY/N)+2*np.cos(2*np.pi*KZ/N)-6.0)/h**2
    lap[0,0,0] = 1.0
    def poisson(src):
        S = np.fft.fftn(src); S[0,0,0] = 0.0
        return np.fft.ifftn(S/lap).real
    def grad(q):
        return ((np.roll(q,-1,0)-np.roll(q,1,0))/(2*h),
                (np.roll(q,-1,1)-np.roll(q,1,1))/(2*h),
                (np.roll(q,-1,2)-np.roll(q,1,2))/(2*h))
    def updiv(q, ux, uy, uz):
        out = np.zeros_like(q)
        for ax, u in ((0,ux),(1,uy),(2,uz)):
            uf = 0.5*(u+np.roll(u,-1,ax))
            F  = uf*np.where(uf > 0, q, np.roll(q,-1,ax))
            out += (F-np.roll(F,1,ax))/h
        return out

    f = coarsen(FINE, N); f = (f - f.mean())/f.std()
    rho = np.maximum(RHOBAR*(1.0+SEED_AMP*f), 0.02*RHOBAR); rho *= RHOBAR/rho.mean()
    e = rho*T0
    mx = np.zeros_like(rho); my = np.zeros_like(rho); mz = np.zeros_like(rho)
    t, a, H, step = 0.0, 1.0, H0, 0
    while t < tmeas and step < 400000:
        ux, uy, uz = mx/rho, my/rho, mz/rho
        cs = np.sqrt((1+GAMMA)*GAMMA*e/rho)
        dt = min(CFL*h/max(float(np.max(np.sqrt(ux**2+uy**2+uz**2)+cs)),1e-6), 0.01)
        dt = min(dt, tmeas-t)
        phi = poisson(G_GRAV*(rho-rho.mean())/a**2)
        gx, gy, gz = grad(phi); px, py, pz = grad(GAMMA*e)
        rho_n = rho - dt*updiv(rho,ux,uy,uz)/a
        mx_n = mx - dt*(updiv(mx,ux,uy,uz)/a + px/a + rho*gx/a + H*mx)
        my_n = my - dt*(updiv(my,ux,uy,uz)/a + py/a + rho*gy/a + H*my)
        mz_n = mz - dt*(updiv(mz,ux,uy,uz)/a + pz/a + rho*gz/a + H*mz)
        divu = (grad(ux)[0]+grad(uy)[1]+grad(uz)[2])/a
        e_n = e - dt*(updiv(e,ux,uy,uz)/a + GAMMA*e*divu + 3.0*H*GAMMA*e)
        rho, e = np.maximum(rho_n,1e-4), np.maximum(e_n,1e-8)
        mx, my, mz = mx_n, my_n, mz_n
        t += dt; step += 1; a = 1.0+H0*t; H = H0/a

    m = rho.mean()
    imax = np.unravel_index(np.argmax(rho), rho.shape)
    cs_pk = np.sqrt((1+GAMMA)*GAMMA*e[imax]/rho[imax])
    lamJ_pk_cells = float(2*np.pi*cs_pk/np.sqrt(G_GRAV*rho[imax]) / h)
    flat = np.sort(rho.ravel())[::-1]; top5 = max(1, int(0.05*flat.size))
    return dict(N=N, t=round(t,3), steps=step,
                peak_over_mean=round(float(rho.max()/m),1),
                deep=round(100.0*float((rho < 0.2*m).mean()),1),
                top5=round(100.0*float(flat[:top5].sum()/flat.sum()),1),
                drift=round(100.0*float((m-RHOBAR)/RHOBAR),4),
                lamJ_at_peak_cells=round(lamJ_pk_cells,2))

rows = []
for N in NLIST:
    r = run(N, TMEAS); rows.append(r); print(json.dumps(r), flush=True)

print()
print(f"t = {TMEAS}   (same seed field at every N, block-averaged from {NFINE}^3)")
print(f"{'N':>5} {'peak/mean':>10} {'deep%':>7} {'top5%':>7} {'lamJ@peak(cells)':>18} {'drift%':>8}")
for r in rows:
    print(f"{r['N']:>5} {r['peak_over_mean']:>10} {r['deep']:>7} {r['top5']:>7} "
          f"{r['lamJ_at_peak_cells']:>18} {r['drift']:>8}")
print()
print("If peak/mean keeps climbing with N, the arrest is not converged.")
print("Truelove: lamJ at the peak should be >~ 4 cells for the collapse to be resolved.")
