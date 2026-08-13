"""Re-plot cosmic_web_contrast.png from saved panels — no solver re-run."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = np.load('cosmic_web_contrast_panels.npz')
times, projs, st, N, gad = d['times'], d['projs'], d['stats'], int(d['N']), float(d['gamma_ad'])
fig, axes = plt.subplots(1, len(times), figsize=(4.6*len(times), 5.4), facecolor='black')
for ax, tt, proj, s in zip(np.atleast_1d(axes), times, projs, st):
    img = np.log10(proj / proj.mean())
    ax.imshow(img.T, origin='lower', cmap='magma',
              vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.8), interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"$t = {tt:.1f}$", color='white', fontsize=13, pad=8)
    ax.set_xlabel(f"deep voids {s[0]:.0f}%   peak/mean {s[2]:.0f}$\\times$   top 5% hold {s[3]:.0f}% of mass",
                  color='#cccccc', fontsize=9, labelpad=8)
fig.suptitle(f"Contrast model $\\nabla^2\\varphi_g = 4\\pi G(\\rho-\\bar\\rho)$, coasting, "
             f"$\\gamma_{{ad}} = {gad:.1f} > 4/3$   ($N = {N}^3$, line-of-sight projection). "
             f"The web forms, then coarsens: it is a transient here too.",
             color='white', fontsize=11.5, y=0.985)
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig('cosmic_web_contrast.png', dpi=140, facecolor='black')
print('replotted')
