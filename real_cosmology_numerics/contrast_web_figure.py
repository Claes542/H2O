"""
Figure for the contrast model: a PERSISTENT web at gamma_ad = 1.4 > 4/3, coasting.

Runs the same equations as contrast_offline.py (and as the browser solver
cosmology_contrast_3d_gpu.html) and writes three line-of-sight projections at
increasing time, with the measured volume statistics printed under each panel.

Line-of-sight projection, not a thin slice: a slice cuts most filaments and a
single plane misrepresents connectivity. Projection is also what a survey sees.

Usage:  python3 contrast_web_figure.py [N] [t_end]
"""
import numpy as np, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N     = int(sys.argv[1]) if len(sys.argv) > 1 else 128
TEND  = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
SEED  = 12345

RHOBAR, GAMMA, G_GRAV, T0 = 1.0, 0.4, 18.0, 0.01     # GAMMA=0.4 -> gamma_ad=1.4 > 4/3
SEED_AMP  = 0.35
CORRCELL  = 12.0 * N / 200.0
H0, CFL   = 0.30, 0.35
h         = 1.0 / N
SHOW_AT   = [float(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0.5','0.9','1.4'])]

rng = np.random.default_rng(SEED)
w  = rng.standard_normal((N, N, N))
k  = np.fft.fftfreq(N) * N
KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
k2 = KX**2 + KY**2 + KZ**2
f  = np.fft.ifftn(np.fft.fftn(w) * np.exp(-0.5 * k2 * CORRCELL**2 / N**2 * np.pi**2)).real
f -= f.mean(); f /= f.std()

rho = np.maximum(RHOBAR * (1.0 + SEED_AMP * f), 0.02 * RHOBAR)
rho *= RHOBAR / rho.mean()
e   = rho * T0
mx  = np.zeros_like(rho); my = np.zeros_like(rho); mz = np.zeros_like(rho)

lap_eig = (2*np.cos(2*np.pi*KX/N) + 2*np.cos(2*np.pi*KY/N) + 2*np.cos(2*np.pi*KZ/N) - 6.0) / h**2
lap_eig[0, 0, 0] = 1.0

def poisson(src):
    S = np.fft.fftn(src); S[0, 0, 0] = 0.0        # the mean does not gravitate
    return np.fft.ifftn(S / lap_eig).real

def grad(q):
    return ((np.roll(q,-1,0)-np.roll(q,1,0))/(2*h),
            (np.roll(q,-1,1)-np.roll(q,1,1))/(2*h),
            (np.roll(q,-1,2)-np.roll(q,1,2))/(2*h))

def upwind_div(q, ux, uy, uz):
    out = np.zeros_like(q)
    for ax, u in ((0,ux), (1,uy), (2,uz)):
        uf = 0.5*(u + np.roll(u,-1,ax))
        qf = np.where(uf > 0, q, np.roll(q,-1,ax))
        F  = uf*qf
        out += (F - np.roll(F,1,ax))/h
    return out

def stats(rho):
    m = rho.mean()
    deep  = 100.0*(rho < 0.2*m).mean()
    over  = 100.0*(rho > m).mean()
    flat  = np.sort(rho.ravel())[::-1]
    top5  = max(1, int(0.05*flat.size))
    return deep, over, float(rho.max()/m), 100.0*float(flat[:top5].sum()/flat.sum()), 100.0*(m-RHOBAR)/RHOBAR

panels, t, a, H, step = [], 0.0, 1.0, H0, 0
while t < TEND and step < 200000:
    ux, uy, uz = mx/rho, my/rho, mz/rho
    cs   = np.sqrt((1.0+GAMMA)*GAMMA*e/rho)
    dt   = min(CFL*h/max(float(np.max(np.sqrt(ux**2+uy**2+uz**2)+cs)), 1e-6), 0.01)
    phi  = poisson(G_GRAV*(rho - rho.mean())/a**2)
    gx, gy, gz = grad(phi)
    px, py, pz = grad(GAMMA*e)
    rho_n = rho - dt*upwind_div(rho, ux, uy, uz)/a
    mx_n  = mx - dt*(upwind_div(mx,ux,uy,uz)/a + px/a + rho*gx/a + H*mx)
    my_n  = my - dt*(upwind_div(my,ux,uy,uz)/a + py/a + rho*gy/a + H*my)
    mz_n  = mz - dt*(upwind_div(mz,ux,uy,uz)/a + pz/a + rho*gz/a + H*mz)
    divu  = (grad(ux)[0] + grad(uy)[1] + grad(uz)[2])/a
    e_n   = e - dt*(upwind_div(e,ux,uy,uz)/a + GAMMA*e*divu + 3.0*H*GAMMA*e)
    rho, e = np.maximum(rho_n, 1e-4), np.maximum(e_n, 1e-8)
    mx, my, mz = mx_n, my_n, mz_n
    t += dt; step += 1
    a = 1.0 + H0*t; H = H0/a
    if SHOW_AT and t >= SHOW_AT[0]:
        panels.append((t, rho.sum(axis=2).copy(), stats(rho)))
        print(f"t={t:5.2f}  deep={stats(rho)[0]:5.1f}%  peak/mean={stats(rho)[2]:8.1f}  "
              f"top5={stats(rho)[3]:5.1f}%  drift={stats(rho)[4]:+.3f}%", flush=True)
        SHOW_AT.pop(0)

np.savez_compressed('cosmic_web_contrast_panels.npz',
                    times=np.array([p[0] for p in panels]),
                    projs=np.array([p[1] for p in panels]),
                    stats=np.array([p[2] for p in panels]), N=N, gamma_ad=1+GAMMA)

fig, axes = plt.subplots(1, len(panels), figsize=(4.6*len(panels), 5.4), facecolor='black')
for ax, (tt, proj, (deep, over, pk, top5, drift)) in zip(np.atleast_1d(axes), panels):
    img = np.log10(proj / proj.mean())
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.8)
    ax.imshow(img.T, origin='lower', cmap='magma', vmin=lo, vmax=hi, interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"$t = {tt:.1f}$", color='white', fontsize=13, pad=8)
    ax.set_xlabel(f"deep voids {deep:.0f}%   peak/mean {pk:.0f}$\\times$   top 5% hold {top5:.0f}% of mass",
                  color='#cccccc', fontsize=9, labelpad=7)
fig.suptitle(f"Contrast model $\\nabla^2\\varphi_g = 4\\pi G(\\rho-\\bar\\rho)$, coasting, "
             f"$\\gamma_{{ad}} = {1+GAMMA:.1f} > 4/3$   ($N = {N}^3$, line-of-sight projection). "
             f"The web forms, then coarsens: it is a transient here too.",
             color='white', fontsize=11.5, y=0.985)
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig('cosmic_web_contrast.png', dpi=140, facecolor='black')
print("wrote cosmic_web_contrast.png")
