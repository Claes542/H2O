#!/usr/bin/env python3
"""
Is C = 0 the reason U is infinite?

Two electrons in a cell can sit two ways:

    SPLIT     two domains, one unit each, meeting at an interface
              -> each is more tightly confined, but they are separated, so they repel less

    TOGETHER  one domain, occupancy two
              -> each is less confined, but they overlap completely, so they repel more
              -> that overlap repulsion IS the Hubbard U

The trade is confinement against repulsion, and RealQM has NO confinement cost: a free
interface carries no localisation energy (C = 0). So splitting is free -- you take the
reduced repulsion and pay nothing -- and single occupancy wins at every density. That
would make C = 0 and U = infinity the same statement seen from two sides, and would mean
the paper's two proposed repairs (Robin beta for the equation of state, finite U for
conduction) are ONE repair.

This tests it. The cell is a jellium sphere of radius R with a uniform +2 background,
neutral overall so the free-space Poisson problem is well posed. Both configurations
carry a surface energy (beta/2) Int psi^2 dS on every boundary of every domain -- which
is the natural boundary condition of the Robin interface, so writing the energy down is
enough. The split configuration therefore pays for one EXTRA surface, the equatorial
disc, and that is the cost of keeping the electrons apart.

    E_split(beta) - E_together(beta) = 0   defines beta*, the crossover.

Below beta* the electrons split (single occupancy, U effectively infinite); above it they
double up (finite U). The question is where beta* falls relative to the beta*a = 1.146
that the degenerate electron gas requires. With 2 electrons in a sphere of radius R the
cell size is a = n^(-1/3) = R (2pi/3)^(1/3) = 1.3104 R.

Energies exclude electron self-interaction, as RealQM does: E_es = U_12 + U_1bg + U_2bg
+ U_bgbg, each from its own Poisson solve. The background term is identical in both
configurations and cancels in the comparison, but is carried anyway.

Everything is assembled as a quadratic form and minimised, so the matrices are symmetric
by construction and the Robin condition emerges from the surface term rather than being
imposed by hand.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.fft as sfft
import sys

R    = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0   # cell radius, a0
NBOX = int(__import__('os').environ.get('NBOX', 48))        # grid points across the box
LBOX = 3.0 * R                                             # box side (Poisson needs room)
H    = LBOX / NBOX
A_CELL = 1.3104 * R                                        # n^(-1/3) for 2 electrons in the sphere

x1 = (np.arange(NBOX) + 0.5) * H - LBOX / 2
X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
RAD = np.sqrt(X**2 + Y**2 + Z**2)
inside = RAD < R                                           # the cell
upper  = Z > 0                                             # the two halves
NCELL = inside.sum()

# uniform +2 background on the cell
n_bg = np.zeros_like(RAD)
n_bg[inside] = 2.0 / (inside.sum() * H**3)


# FREE-SPACE Poisson by zero-padded FFT (Hockney): the potential of an ISOLATED charge
# distribution, not one in a box. This is essential here. A Dirichlet box forces phi = 0
# at the wall, where a unit charge still has phi ~ 1/D -- the first attempt put the wall at
# 4.5 a0 and the sphere self-energy came out 0.103 against the exact 0.200. Enlarging the
# box does not save it (1% would need walls at 250 a0), and the individual electron and
# background potentials are each non-neutral, so their truncation errors do not cancel
# between the two configurations being compared.
_NP = 2 * NBOX
_g = np.fft.fftfreq(_NP, d=1.0 / _NP) * H          # signed offsets, wrapped
_GX, _GY, _GZ = np.meshgrid(_g, _g, _g, indexing='ij')
_GR = np.sqrt(_GX**2 + _GY**2 + _GZ**2)
_KER = np.zeros_like(_GR)
np.divide(1.0, _GR, out=_KER, where=_GR > 0)
_KER[0, 0, 0] = 2.38 / H                            # cell-averaged 1/r for a cubic cell
_KHAT = np.fft.rfftn(_KER)


def potential(n):
    """phi of the isolated distribution n, treated as POSITIVE charge. Exact 1/r kernel."""
    pad = np.zeros((_NP, _NP, _NP))
    pad[:NBOX, :NBOX, :NBOX] = n
    out = np.fft.irfftn(np.fft.rfftn(pad) * _KHAT, s=(_NP, _NP, _NP))
    return out[:NBOX, :NBOX, :NBOX] * H**3


print(f"cell R = {R} a0   box {LBOX:.1f} ({NBOX}^3, h = {H:.3f})   "
      f"cell size a = n^(-1/3) = {A_CELL:.3f} a0", flush=True)
print(f"domain: {NCELL} cells in the sphere", flush=True)

# validation: a uniform sphere of unit charge has phi(0) = 3/(2R) and self-energy 3/(5R)
_ntest = np.zeros_like(RAD); _ntest[inside] = 1.0 / (inside.sum() * H**3)
_pt = potential(_ntest)
_self = 0.5 * float((_ntest * _pt).sum()) * H**3
print(f"Poisson check: sphere self-energy {_self:.5f} vs exact {3/(5*R):.5f} "
      f"({100*abs(_self-3/(5*R))/(3/(5*R)):.2f}% error)", flush=True)

PHI_BG = potential(n_bg)
U_BGBG = 0.5 * float((n_bg * PHI_BG).sum()) * H**3


def kinetic_and_surface(mask, beta):
    """
    Symmetric A with u^T A u = Int |grad u|^2 dV over the masked domain, plus the surface
    term beta * Int u^2 dS on EVERY face of the domain that borders a non-domain cell.
    A face to a cell outside the domain is simply omitted from the stiffness, which is the
    natural (zero-flux) condition; the surface term then supplies the Robin condition.
    """
    idx = -np.ones(mask.shape, dtype=np.int64)
    idx[mask] = np.arange(mask.sum())
    rows, cols, vals = [], [], []
    diag = np.zeros(mask.sum())
    nface = np.zeros(mask.sum())                    # exposed faces, for the surface term
    for ax in range(3):
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[ax] = slice(0, mask.shape[ax] - 1); sl_b[ax] = slice(1, mask.shape[ax])
        ia = idx[tuple(sl_a)].ravel(); ib = idx[tuple(sl_b)].ravel()
        both = (ia >= 0) & (ib >= 0)
        rows.append(ia[both]); cols.append(ib[both]); vals.append(-H * np.ones(both.sum()))
        rows.append(ib[both]); cols.append(ia[both]); vals.append(-H * np.ones(both.sum()))
        np.add.at(diag, ia[both], H); np.add.at(diag, ib[both], H)
        # exposed faces: in the domain but the neighbour is not
        ea = (ia >= 0) & (ib < 0); eb = (ib >= 0) & (ia < 0)
        np.add.at(nface, ia[ea], 1.0); np.add.at(nface, ib[eb], 1.0)
        # and the box faces themselves
        sl = [slice(None)] * 3
        sl[ax] = 0
        e0 = idx[tuple(sl)].ravel(); np.add.at(nface, e0[e0 >= 0], 1.0)
        sl[ax] = mask.shape[ax] - 1
        e1 = idx[tuple(sl)].ravel(); np.add.at(nface, e1[e1 >= 0], 1.0)
    diag = diag + beta * nface * H**2               # (beta/2) Int u^2 dS, doubled below
    rows.append(np.arange(mask.sum())); cols.append(np.arange(mask.sum())); vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(mask.sum(),) * 2)
    return A, nface


def solve_config(masks, beta, label, verbose=False):
    """
    masks: list of domain masks, one per ELECTRON. Two identical masks = both electrons on
    the same domain (occupancy two). Two half masks = one each.
    Returns the total energy, self-interaction excluded.
    """
    ops = [kinetic_and_surface(m, beta) for m in masks]
    dens = []
    for m in masks:
        d = np.zeros_like(RAD); d[m] = 1.0 / (m.sum() * H**3)
        dens.append(d)

    E = None
    for it in range(40):
        newd, Ts, Ss = [], [], []
        for e, m in enumerate(masks):
            A, nface = ops[e]
            other = np.zeros_like(RAD)
            for f in range(len(masks)):
                if f != e:
                    other += dens[f]
            V = potential(other) - PHI_BG          # repelled by the other electron, drawn to bg
            Hm = 0.5 * A + sp.diags(V[m] * H**3)
            M = sp.diags(np.full(m.sum(), H**3))
            try:
                w, v = spl.eigsh(Hm.tocsc(), k=1, M=M.tocsc(), sigma=-2.5,
                                 which='LM', maxiter=6000)
            except Exception:
                w, v = spl.eigsh(Hm.tocsc(), k=1, M=M.tocsc(), which='SA', maxiter=20000)
            u = np.abs(v[:, 0])
            # A failed eigensolve must be LOUD. At NBOX=28 this returned NaN and the
            # crossover was reported anyway (beta* = 0.187) with only a RuntimeWarning to
            # show for it -- a silently poisoned number is worse than no number.
            nrm = float(u @ (M @ u))
            if not np.isfinite(nrm) or nrm <= 0 or not np.all(np.isfinite(u)):
                raise RuntimeError(
                    f"eigensolve failed: label={label} electron={e} beta={beta} "
                    f"norm={nrm} -- refusing to report a crossover from this")
            u /= np.sqrt(nrm)
            d = np.zeros_like(RAD); d[m] = u**2
            newd.append(d)
            Ts.append(0.5 * float(u @ (kinetic_and_surface(m, 0.0)[0] @ u)))
            Ss.append(0.5 * beta * float((u**2 * nface).sum()) * H**2)
        shift = max(float(np.abs(newd[i] - dens[i]).max()) for i in range(len(masks)))
        dens = [0.4 * newd[i] + 0.6 * dens[i] for i in range(len(masks))]
        for i in range(len(masks)):
            dens[i] /= float(dens[i].sum()) * H**3
        if shift < 1e-9:
            break

    T = sum(Ts); S = sum(Ss)
    phis = [potential(d) for d in dens]
    U12 = 0.0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            U12 += float((dens[i] * phis[j]).sum()) * H**3
    Uebg = -sum(float((d * PHI_BG).sum()) * H**3 for d in dens)
    E = T + S + U12 + Uebg + U_BGBG
    if verbose:
        print(f"    {label:9s} T={T:+.5f} S={S:+.5f} U12={U12:+.5f} Uebg={Uebg:+.5f} "
              f"E={E:+.5f}  (it={it+1}, d={shift:.1e})", flush=True)
    return E


print("\n  beta   beta*a      E_split    E_together      split-together", flush=True)
print('-' * 66, flush=True)
rows = []
for beta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0]:
    Es = solve_config([inside & upper, inside & ~upper], beta, 'split')
    Et = solve_config([inside, inside], beta, 'together')
    rows.append((beta, Es, Et))
    d = Es - Et
    tag = '  split wins' if d < 0 else '  TOGETHER wins'
    print(f"{beta:6.2f} {beta*A_CELL:7.3f}  {Es:+11.5f}  {Et:+11.5f}  {d:+11.5f}{tag}",
          flush=True)

cross = None
for i in range(1, len(rows)):
    d0, d1 = rows[i-1][1] - rows[i-1][2], rows[i][1] - rows[i][2]
    if d0 < 0 <= d1:
        b0, b1 = rows[i-1][0], rows[i][0]
        cross = b0 + (b1 - b0) * (-d0) / (d1 - d0)
        break
print()
if cross:
    print(f"CROSSOVER  beta* = {cross:.3f} a0^-1,  beta* a = {cross*A_CELL:.3f}")
    print(f"           the electron gas requires beta a = 1.146")
    print(f"           ratio = {cross*A_CELL/1.146:.2f}")
else:
    print("no crossover in the scanned range")
