# Draft note to the editor — RealNucleus, corrected binding ratio

**Status: DRAFT for the author to review, edit and send. Not sent.**

Applies if the version under consideration at *Physics Essays* carries the claim
"the alpha emerges at 103% of its experimental binding (alpha/deuteron ratio 13.1
versus 12.7)". Check the submitted PDF before sending — if the submitted version
predates that claim, this note is unnecessary.

---

Dear Editor,

I am writing to correct a quantitative claim in my submitted manuscript
*[title]*, reference *[number]*, before it goes further in review.

The paper reports nuclear binding from a Coulomb-only model and quotes, in the
abstract and conclusion, an alpha/deuteron binding ratio of 13.1 against the
experimental 12.7, with the alpha at 103% of its experimental binding. Those
figures were obtained at a single mesh resolution. On re-examination with a
resolution ladder they do not hold.

Computing the deuteron on four meshes (N = 140, 170, 200, 230 across the same
domain) gives energies of −5.21, −3.52, −2.70 and −2.74 in model units: the
value plateaus near −2.7, with the two finest meshes agreeing to 1.6%. The
figure of −2.12 used in the manuscript to fix the energy scale is therefore not
converged; the converged value is some 27% deeper. The alpha over the same range
settles near −29.

Because the ladder is normalised on the deuteron, this propagates:

- the alpha/deuteron ratio is **approximately 11**, not 13.1, against an
  experimental 12.7;
- the alpha stands at **approximately 87%** of its experimental binding, not
  103%;
- every energy in the ladder table is rescaled by a factor of about 0.79.

The discretisation error is *differential*, which is why it was not caught by
inspection: it flatters the deuteron — loosely bound and spatially extended —
considerably more than the tightly bound alpha, so the ratio of the two carries
the full difference.

I should also record a second limitation, which the revised manuscript now
states explicitly. The Be-8 and O-16 entries are one arrangement selected from a
scan over shell configurations. The spread across arrangements is much larger
than any mesh effect — for Be-8, the same constituents give −68, −58, −30, and,
as two near-touching alphas, an unbound +47 — so those rows are selections
rather than predictions until a rule for choosing the configuration is stated in
advance.

The central claim of the paper is unaffected in kind: nuclear binding of the
right order arises from Coulomb geometry with no strong force and no fitted
parameter. But the agreement is approximately 13% low rather than 3% high, and
the manuscript overstated it. A revised version carrying the corrected figures,
a new section documenting the resolution study, and the caveat on configuration
selection is available, and I would be glad to submit it in place of the current
one.

I apologise for the correction and hope it reaches you before it has cost the
referees time.

Yours sincerely,

Claes Johnson
Professor emeritus of Applied Mathematics
KTH Royal Institute of Technology

---

## Supporting numbers, if the editor or a referee asks

Deuteron, published configuration, energies in model units:

| N | 140 | 170 | 200 | 230 |
|---|---|---|---|---|
| E | −5.21 | −3.52 | −2.70 | −2.74 |

He-4 over the same range settles near −29 (−34.2 at N=140, −29.8 at N=200).
Ratio by mesh: about 6.6 at N=140, 11.05 at N=200, and flat thereafter.

Method: each configuration is taken unmodified from its own page and driven to
the solver's convergence test on a sequence of meshes; the reported value is the
energy at the stop, with the trace retained so that runs which stopped while
still descending are flagged. Scripts are in `real_nucleus_numerics/`
(`ladder_protocol.py`), and the resolution study is reproducible from them.

One honest caveat on the correction itself: no run has settled below about
10⁻³ relative drift, because the solver's stopping test fires while the energy is
still slowly descending, and a longer time budget does not help — only a tighter
threshold does. The ratio of about 11 could therefore move by a few percent. It
will not move back to 13.1.
