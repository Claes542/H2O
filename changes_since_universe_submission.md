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
