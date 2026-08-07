# Changes to `real_cosmology_article` since the Universe (MDPI) submission

**Submitted to Universe:** 2026-08-06 (the PDF the editor/referees have).
**Purpose of this log:** track every change made *after* submission so they can be folded cleanly into the
first revision (R1) uploaded through the MDPI portal, rather than swapped in mid-triage.

Policy (agreed): bank improvements here; do **not** email the editor a replacement file mid-review. Apply them
in R1.

---

## 2026-08-07

- **§10 Open problems, item 1 rewritten** — "The nuclear scale" → **"The proton–electron size difference,"**
  split into three questions of unequal status:
  - *(i) the ratio* — electron ∼α⁻² larger than the proton, read as geometric (a₀/R_p ≈ 1/α²); **addressed**.
  - *(ii) the absolute scale* — what fixes R_p itself (= m_p/m_e = nuclear energy scale): one open number, not
    derived.
  - *(iii) the sign asymmetry* — why + is compact and − diffuse *at all*: pure Coulomb+Laplacian is
    charge-conjugation symmetric, so the asymmetry **cannot** come from the symmetric ingredients and needs a
    symmetry-breaking element; candidate = a **non-Gaussian (skewed) seed** whose super-/sub-level sets are not
    mirror images (compact pits vs diffuse swells) — a geometric route from the *statistics* of the seed.
  - Mass-ratio item now cross-references 1(ii).
  - Page count 10 → 11.

- **§9 new subsection "Why the proton must be a point: free-boundary stability"** — the size asymmetry is what
  makes the atom *well-posed and stable*, not just complex. A RealQM free boundary is stable iff the cross-interface
  force is repulsive: electron–electron (like) → stable; electron–proton (opposite, finite) → attractive → a
  Neumann-continuous free boundary has no interior energy minimum and runs away to collapse (unstable), while a
  Dirichlet-0 node is stable but weaker. Nature's resolution = point proton ⇒ no finite interface, electron cusps
  (Kato) at the point, instability never arises. Backed by minimal spherically symmetric computations
  (`real_cosmology_numerics/`): finite-proton scan shows both BCs → −13.6 eV as R_p→0 and diverge only for large
  R_p (at R_p=1 a₀: Neumann −10.2 eV/75%, Dirichlet −4.9 eV/36%); forcing the electron density to zero at the
  kernel *point* leaves −13.6 eV unchanged (measure-zero; the electron closes the hole and returns to the cusp),
  only a *finite* hollow shell costs energy (≈ −10.6 eV fully hollow). Numerics scripts committed:
  `proton_size_scan.py` (Neumann/softening, variational 1s), `dirichlet_exclusion.py` (radial eigensolver),
  `kernel_node.py` (spherically symmetric hollow trial). These are minimal independent models, NOT the production
  WebGPU solver `molecule_nucleus.js` (which couldn't be run headless here), but robustly capture the BC-sensitivity
  and the R_p→0 convergence. Page count still 11.
  - **Refinement (same day):** the subsection was sharpened after noting Dirichlet-0 is *unphysical* (imposes a
    node with no physical cause) even though stable. New computation `neumann_instability.py`: the cross-interface
    energy U(d) for opposite charges is *monotone*, minimum always at contact → the Neumann free boundary has **no
    interior equilibrium for ANY finite proton size** (not just large) → unstable at all finite sizes. So the
    conclusion is stronger: a finite p–e interface has **no acceptable BC** — physical (Neumann) is unstable,
    stable (Dirichlet) is unphysical — and only the *point* proton (no interface, Kato cusp) works. Size asymmetry
    reframed from "convenience" to "necessity."
  - **Added paragraph "The neutron and the alpha"** (end of §9 subsection): the single-pair free-boundary
    instability *is* the free-neutron instability — a neutron = one compact electron in one proton, proton and
    compact electron comparable in size = the ill-posed regime → model predicts free neutron unstable (β-decay
    ~15 min), the instability = the electron expanding back out. Nuclei escape via *collective confinement*: in the
    α (⁴He = 4p+2e, 2 compact electrons caging 4 protons) each electron is held by several protons → restoring
    force a lone pair lacks; same variational Coulomb binds it (He-4/deuteron ratio 13.1 vs 12.7), no strong force.
    Free neutron unstable, bound neutron stable = one mechanism. Ties §9 free-boundary stability to the RealNucleus
    section. Also settled: instability criterion is R_p/a0 ≪ 1 ⟺ m_p/m_e ≫ 1; 1836 is overkill (could be ~10–100
    and still well-posed; ~1/positronium regime is ill-posed) — the model explains the ratio must be large, not its
    value.
  - **Correction (same day): the earlier "no interior equilibrium for ANY finite proton" was too strong.** The
    U(d) monotone-to-contact result is only the *driver*; "contact" is benign (the stable cusped atom) when the
    proton nests as a point inside the electron, and pathological (forbidden overlap → collapse) only when the
    clouds are comparable. Correct **instability criterion: R_p ≳ a_e ⟺ m_p ≲ m_e** (comparable sizes / order-unity
    mass ratio). For R_p ≪ a_e (m_p ≫ m_e) Neumann is *stable* (proton_size_scan confirms 99%+ binding at small
    R_p). Lands right: real H (ratio 1836) deep stable; free neutron (nuclear compact-e ~ proton size, ratio ~1) at
    threshold ⇒ unstable; alpha rescued by collective confinement. Paper §9 paragraph 2 rewritten with the
    criterion as a displayed equation. Order-of-magnitude threshold; sharp value needs the free-boundary solver.
  - **MAJOR RETRACTION + REWRITE: the free-boundary-stability framing was WRONG.** Independent 3D CPU checks
    (`real_cosmology_numerics/interface3d.py`, `verify3d.py`, `shell3d.py`) showed: (a) the Neumann +/- interface
    is STABLE to ripples (kinetic surface tension dominates; electrostatic response negligible) — no Mullins–Sekerka
    fingering; (b) "energy minimised at contact" is a stable BOUND MINIMUM (two magnets sticking), NOT a runaway — I
    had inverted it; (c) a big single proton doesn't hold a caged electron — the electron EXPANDS OUT (energetics),
    which IS β-decay, not a boundary instability. Conclusion: **stability is ENERGETICS (which config is the ground
    state), not free-boundary stability; the Neumann +/- boundary is always stable — a non-issue.** §9 subsection
    retitled "…free-boundary stability" → **"Why the proton must be compact: an energetic argument"** and fully
    rewritten: electron has TWO branches (expanded/atomic ~eV vs compact/nuclear ~MeV); a POINT proton has no
    interior so holds the electron only AROUND it (atom, −13.6 eV, BC-independent as R_p→0) and CANNOT hold one
    inside; holding an electron INSIDE (compact) needs a CAGE of ≥2 protons; the two branches CROSS at the neutron
    (1p+1e, R_p~a_e), where the compact branch is not the ground state → electron expands → β-decay (m_n≈m_p+m_e,
    0.78 MeV = on the ridge); stable nuclei begin at the deuteron (2p+1e), α (4p+2e) deeply bound (ratio 13.1 vs
    12.7). ALL free-boundary-instability claims RETRACTED (fingering, no-interior-equilibrium, "Neumann unstable /
    no acceptable BC", the R_p≳a_e criterion eq). Now 11 pp.
  - Harness `big_proton_test.html` added (molecule_nucleus.js: electron-inside-proton-shell, Neumann
    [USER_DIRICHLET_EP=false] vs Dirichlet ['both'] × big [m=1] vs heavy [m=1836] proton) — WebGPU/browser, the
    definitive check for the author to run before finalizing §9. Numerics added to `real_cosmology_numerics/`:
    interface3d.py, verify3d.py, shell3d.py, shell3d_long.py.

## 2026-08-07 (cont.) — two-level BC/energetics + nuclear-scale ceiling

- **§9: two-level statement added** — free-boundary stability is the *enabling* condition (surface tension +
  non-overlap hold any interface at a minimum; universal, so it can't discriminate), energetics is the *deciding*
  condition (selects which boundary-stable config is the ground state: atom / decaying neutron / bound nucleus).
  Resolves the flip-flop: the boundary is always stable; the physics is ground-state selection.
- **§9 numerics updated to the REAL solver run** (`big_proton_test.html`, molecule_nucleus.js, 4 configs): all bind
  (my crude "electron expands/unbinds" was an artifact); the *trend* supports energetics — 1p → most expanded
  electron (atom, e size 1.5 a0), 2p+heavy → most compressed + deepest (−28.6 Ha, e size 0.47), i.e. compression
  toward the nuclear branch with proton number+compactness. Grid h~0.08 a0 can't resolve fm, so the compact branch
  / neutron instability are below resolution — those rest on RealNucleus + the energetic argument.
- **§4: nuclear-scale CEILING added, relativity-free** — RealNucleus needs exactly ONE dimensionful input (the
  nuclear scale R_p, or one binding energy); everything dimensionless (ratios, ladder) is parameter-free. That one
  number is BOUNDED: RealQM = fixed persistent charge clouds, valid only while binding < electron energy content
  E_e (~0.5 MeV, MEASURED, not relativistic). e²/R_p ≲ E_e ⇒ R_p ≳ e²/E_e (= r_e). **Stated α-free** (author: the
  a0/R_p ≲ 1/α² form just re-expresses it as a ratio to a0 and imports the atomic coupling — not needed). Crossing
  it (pair creation) is relativistic & outside RealQM; the boundary itself is a plain energy comparison. Nature
  SATURATES it: deuteron e²/R_p ≈ 0.7 MeV > E_e ≈ 0.5 MeV → nuclear/β regime is where the static-cloud picture is
  most stretched. Ties to §5 (inertia = energy).
