"""
Does the SEED carry the observed P(k)?

If structure is supplied by the seed rather than grown by gravity (the horn taken
when the coasting arena is kept), then the shape of the matter power spectrum is
the seed's responsibility, not gravity's. That is a testable statement, and this
is the test.

Observed, for reference: the matter power spectrum rises as P(k) ~ k^n_s with
n_s ~ 0.96 on large scales, turns over at k_eq, and falls as ~k^-3 ln^2 k on
small ones. The large-scale slope is the cleanest target: n = +1, essentially.

Measured here for each seed, before and after evolution:
  - the log-log slope of P(k) on large scales (low k)
  - the slope on small scales (high k)
  - where k^3 P(k) peaks, i.e. which scale carries the structure

Usage: python3 power_spectrum.py [N] [t_measure]
"""
import numpy as np, sys, json

N     = int(sys.argv[1]) if len(sys.argv) > 1 else 128
TMEAS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
GFAC  = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0   # 0 = gravity off (control)

# identical to convergence_ladder.py
SEED = 12345
RHOBAR, GAMMA, G_GRAV, T0 = 1.0, 0.4, 18.0, 0.01
SEED_AMP, H0, CFL = 0.35, 0.30, 0.35
CORR_BOX = 12.0/200.0


def seed_field(NF, kind):
    rng = np.random.default_rng(SEED)
    w = rng.standard_normal((NF, NF, NF))
    k = np.fft.fftfreq(NF)*NF
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    kmag = np.sqrt(k2); kmag[0, 0, 0] = 1.0
    if kind == 'power':
        amp = kmag**0.5                              # P(k) = |delta_k|^2 ~ k
        amp *= np.exp(-0.5*(kmag/(0.35*NF))**2)      # de-alias at the grid scale only
        amp[0, 0, 0] = 0.0
        f = np.fft.ifftn(np.fft.fftn(w)*amp).real
    else:                                            # 'gauss': one correlation length
        corr = CORR_BOX*NF
        f = np.fft.ifftn(np.fft.fftn(w)*np.exp(-0.5*k2*corr**2/NF**2*np.pi**2)).real
    f -= f.mean(); f /= f.std()
    return f


def pk(delta):
    """Spherically averaged P(k) in integer k-shells, k in units of the fundamental mode."""
    n = delta.shape[0]
    D = np.fft.fftn(delta)/n**3
    k = np.fft.fftfreq(n)*n
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kbin = np.rint(kmag).astype(int)
    power = (np.abs(D)**2).ravel()
    kb = kbin.ravel()
    nk = np.bincount(kb, minlength=n)
    ps = np.bincount(kb, weights=power, minlength=n)
    good = nk > 0
    ks = np.arange(n)[good]
    ps = ps[good]/nk[good]
    return ks[1:], ps[1:]                            # drop k=0


def slope(ks, ps, klo, khi):
    m = (ks >= klo) & (ks <= khi) & (ps > 0)
    if m.sum() < 3:
        return float('nan')
    return float(np.polyfit(np.log(ks[m]), np.log(ps[m]), 1)[0])


def evolve(rho0, tmeas):
    n = rho0.shape[0]; h = 1.0/n
    k = np.fft.fftfreq(n)*n
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    lap = (2*np.cos(2*np.pi*KX/n)+2*np.cos(2*np.pi*KY/n)+2*np.cos(2*np.pi*KZ/n)-6.0)/h**2
    lap[0, 0, 0] = 1.0

    def poisson(src):
        S = np.fft.fftn(src); S[0, 0, 0] = 0.0
        return np.fft.ifftn(S/lap).real

    def grad(q):
        return ((np.roll(q, -1, 0)-np.roll(q, 1, 0))/(2*h),
                (np.roll(q, -1, 1)-np.roll(q, 1, 1))/(2*h),
                (np.roll(q, -1, 2)-np.roll(q, 1, 2))/(2*h))

    def updiv(q, ux, uy, uz):
        out = np.zeros_like(q)
        for ax, u in ((0, ux), (1, uy), (2, uz)):
            uf = 0.5*(u+np.roll(u, -1, ax))
            F = uf*np.where(uf > 0, q, np.roll(q, -1, ax))
            out += (F-np.roll(F, 1, ax))/h
        return out

    rho = rho0.copy(); e = rho*T0
    mx = np.zeros_like(rho); my = np.zeros_like(rho); mz = np.zeros_like(rho)
    t, a, H, step = 0.0, 1.0, H0, 0
    while t < tmeas and step < 400000:
        ux, uy, uz = mx/rho, my/rho, mz/rho
        cs = np.sqrt((1+GAMMA)*GAMMA*e/rho)
        dt = min(CFL*h/max(float(np.max(np.sqrt(ux**2+uy**2+uz**2)+cs)), 1e-6), 0.01)
        dt = min(dt, tmeas-t)
        phi = poisson(GFAC*G_GRAV*(rho-rho.mean())/a**2)
        gx, gy, gz = grad(phi); px, py, pz = grad(GAMMA*e)
        rho_n = rho - dt*updiv(rho, ux, uy, uz)/a
        mx_n = mx - dt*(updiv(mx, ux, uy, uz)/a + px/a + rho*gx/a + H*mx)
        my_n = my - dt*(updiv(my, ux, uy, uz)/a + py/a + rho*gy/a + H*my)
        mz_n = mz - dt*(updiv(mz, ux, uy, uz)/a + pz/a + rho*gz/a + H*mz)
        divu = (grad(ux)[0]+grad(uy)[1]+grad(uz)[2])/a
        e_n = e - dt*(updiv(e, ux, uy, uz)/a + GAMMA*e*divu + 3.0*H*GAMMA*e)
        rho, e = np.maximum(rho_n, 1e-4), np.maximum(e_n, 1e-8)
        mx, my, mz = mx_n, my_n, mz_n
        t += dt; step += 1; a = 1.0+H0*t; H = H0/a
    return rho, t, step


# large-scale band: well above the fundamental, well below the seed/grid scales.
KLO, KHI = 2, 8
KVLO, KVHI = 1, 4
KHI_LO, KHI_HI = 16, N//3

print(f"N = {N},  t = {TMEAS},  G x {GFAC},  large-scale band k = {KLO}-{KHI}, "
      f"small-scale band k = {KHI_LO}-{KHI_HI}")
print("observed target: P(k) ~ k^+1 on large scales (n_s ~ 0.96); turnover; then falling\n")

rows = []
for kind in ('gauss', 'power'):
    f = seed_field(N, kind)
    rho0 = np.maximum(RHOBAR*(1.0+SEED_AMP*f), 0.02*RHOBAR)
    rho0 *= RHOBAR/rho0.mean()
    d0 = rho0/rho0.mean() - 1.0
    ks, p0 = pk(d0)
    rho1, t, step = evolve(rho0, TMEAS)
    d1 = rho1/rho1.mean() - 1.0
    _, p1 = pk(d1)
    d2 = ks**3*p1
    kpeak = int(ks[np.argmax(d2)])
    r = dict(seed=kind, steps=step, G=GFAC,
             n_vlarge_seed=round(slope(ks, p0, KVLO, KVHI), 2),
             n_vlarge_evolved=round(slope(ks, p1, KVLO, KVHI), 2),
             n_large_seed=round(slope(ks, p0, KLO, KHI), 2),
             n_large_evolved=round(slope(ks, p1, KLO, KHI), 2),
             n_small_seed=round(slope(ks, p0, KHI_LO, KHI_HI), 2),
             n_small_evolved=round(slope(ks, p1, KHI_LO, KHI_HI), 2),
             k_peak_of_k3P=kpeak,
             peak_over_mean=round(float(rho1.max()/rho1.mean()), 1))
    rows.append(r)
    print(json.dumps(r), flush=True)

print()
print(f"{'seed':>7} {'n(k=1-4) seed':>14} {'evolved':>9} {'n(k=2-8) seed':>14} {'evolved':>9} "
      f"{'k peak':>7} {'peak/mean':>10}")
for r in rows:
    print(f"{r['seed']:>7} {r['n_vlarge_seed']:>14} {r['n_vlarge_evolved']:>9} "
          f"{r['n_large_seed']:>14} {r['n_large_evolved']:>9} "
          f"{r['k_peak_of_k3P']:>7} {r['peak_over_mean']:>10}")
print()
print("Read against the G x 0 control before attributing anything to gravity: the")
print("large-scale slope degrades there too, so the reprocessing is pressure")
print("smoothing (high-k power decays, leaving relatively more low-k), which gravity")
print("partly opposes rather than causes. The consequence for the paper is the same")
print("either way -- the late-time slope is NOT the seed slope, so a seed cannot")
print("simply be handed the observed P(k) and expected to deliver it.")
