"""
Light-nucleus binding ladder, run to a protocol instead of to a single number.

The first attempt (run_ladder.py) took each page's own grid and its own auto-stop
and divided everything by the deuteron. That produced ratios scattering over
+-50%, and four separate reasons why:

  1. the deuteron is the worst possible reference -- barely bound, huge, right at
     the edge of stability, so its energy is maximally sensitive to everything,
     and every ratio divided by it;
  2. the auto-stop is ABSOLUTE, |dE| < 0.01, which a shallow descent trips on a
     plateau while still drifting (18 s for the deuteron against 175 s for He-4);
  3. the grid was never shown to be converged -- the deuteron gives -5.08 at
     N=100 and -2.695 at N=200, so N=200 may be a point on a curve, not a limit;
  4. the shell radii are hand-set, so the energy is a configuration CHOICE and
     not a variational result.

This runs the protocol that answers all four:

  Phase A  minimise over an overall size factor lambda, which scales the seed
           radii and both zone boundaries together. The nucleus then chooses its
           own size, and R(A) is an output -- which is the open question (what
           sets saturation), not a knob.
  Phase B  grid ladder at lambda*, Richardson-extrapolated to h -> 0, quoted
           with the extrapolation residual as an error bar.
  Phase C  ratios against HE-4, not the deuteron. The deuteron then appears as a
           prediction, which is a far stronger test than using it to calibrate.

The auto-stop is kept but tightened by a factor of twenty (|dE| < 5e-4 against
the pages' 0.01), because it is also how a result gets reported at all -- with it
disabled the run simply never posts back. A wall-clock budget is the fallback,
and convergence is judged afterwards from the E-trace by the relative drift over
its last quarter, so a run that stopped early is still flagged as unsettled
rather than used silently.

usage:
  python3 ladder_protocol.py scan   [--grid 160] [--budget 300] [case ...]
  python3 ladder_protocol.py ladder [--grids 120,160,200,240] [--budget 600] [case ...]
  python3 ladder_protocol.py report
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path('/Users/cgjoh/Development/H2O')
HERE = Path(__file__).resolve().parent
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
STATE = HERE / 'ladder_state.json'

# case -> (page, measured binding energy / MeV).  AME2020.
# He-4 is the reference: deep, tightly bound, geometrically clean.
CASES = {
    'he4':      ('nucleus_he4_2plus2.html',      28.2957),
    'deuteron': ('nucleus_deuterium_2shell.html', 2.2246),
    'he3':      ('nucleus_he3.html',              7.7180),
    'li6':      ('nucleus_li6_matched.html',     31.9946),
    'be8':      ('nucleus_be8_2alpha_cloud.html',56.4996),   # the dumbbell, not one shell
}
REFERENCE = 'he4'
# The solver stops on |dE| < THRESH between samples. Disabling it entirely (0)
# does not work: the run then never reports, so the tight threshold is both the
# convergence fix and the mechanism by which a result comes back at all.
THRESH = 5e-4
LAMBDAS = [0.75, 0.85, 1.00, 1.15, 1.30]

DRIVER = """<!DOCTYPE html><html><body style="margin:0;background:#000">
<pre id="out" style="color:#0f0;font:12px monospace">starting</pre>
<script>
window.__err = [];
window.onerror = function (m, src, l) { window.__err.push(m + ' @' + (src||'').split('/').pop() + ':' + l); };
</script>
<script>
%(config)s
window.CONVERGENCE_THRESHOLD = %(thresh)s;   // the solver's own stop, tightened; the budget is a fallback
window.__label = %(label)r;
window.__trace = [];
window.__done = false;
window.onSweepDone = function (E) { window.__done = true; report(Number(E)); };
function gpuState() {
  try { return 'ready=' + gpuReady + ' err=' + (gpuError || 'none') + ' init=' + (typeof initProgress !== 'undefined' ? initProgress : '?'); }
  catch (ex) { return 'binding: ' + ex.message; }
}
function report(E) {
  var r = JSON.stringify({case: window.__label, E: E, trace: window.__trace, err: window.__err,
                          gpu: gpuState(), prevE: (window._prevE === undefined ? 'undef' : String(window._prevE))});
  document.getElementById('out').textContent = 'RESULT ' + r;
  fetch('/__result?payload=' + encodeURIComponent(r));
}
setInterval(function () {
  if (window._prevE === undefined || !isFinite(window._prevE)) return;
  var t = window.__trace, v = Number(Number(window._prevE).toFixed(4));
  if (!t.length || t[t.length - 1] !== v) t.push(v);
  var e = document.getElementById('out');
  if (e && !window.__done) e.textContent = 'PROGRESS n=' + t.length + ' E=' + v;
}, 250);
window.__deadline = setTimeout(function () {         // budget reached: report whatever there is
  if (window.__done) return;
  window.__done = true;
  report(window.__trace.length ? window.__trace[window.__trace.length - 1] : null);
}, %(budget_ms)d);
</script>
<script src="/molecule_nucleus.js"></script>
<script src="/real_nucleus_numerics/p5.min.js"></script>
</body></html>
"""


def build(label, page, grid, lam, budget_s):
    """Extract the page's config, scale every radius by lam, and rescale the grid."""
    html = (ROOT / page).read_text()
    body = html[html.index('<script>') + 8:html.rindex('</script>')]
    cfg = body[:body.index('var s = document.createElement')]
    cfg = re.sub(r'window\.onSweepDone\s*=\s*function[\s\S]*?\};', '', cfg)

    if lam != 1.0:
        # the whole radius family together: seed radii and both zone boundaries
        def scale_assign(m):
            return f'{m.group(1)}{float(m.group(2)) * lam:.4f}'
        cfg = re.sub(r'(\brIn\s*=\s*)([0-9.]+)', scale_assign, cfg)
        cfg = re.sub(r'(\brOut\s*=\s*)([0-9.]+)', scale_assign, cfg)
        cfg = re.sub(r'(window\.USER_SHELL_RADIUS\s*=\s*)([0-9.]+)', scale_assign, cfg)
        cfg = re.sub(r'(window\.USER_SHELL_RADIUS2\s*=\s*)([0-9.]+)', scale_assign, cfg)
    if grid:
        cfg = re.sub(r'var gridN = \d+', f'var gridN = {grid}', cfg, count=1)
        cfg = re.sub(r'N2 = \d+', f'N2 = {grid // 2}', cfg, count=1)

    out = HERE / f'_drv_{label}_{grid}_{int(lam*100)}.html'
    out.write_text(DRIVER % dict(config=cfg, label=label, budget_ms=int(budget_s * 1000),
                                 thresh=repr(THRESH)))
    return out


class _Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith('/__result'):
            q = parse_qs(urlparse(self.path).query)
            self.server.result = json.loads(q['payload'][0])
            self.send_response(204); self.end_headers()
            return
        return super().do_GET()


def serve():
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), partial(_Handler, directory=str(ROOT)))
    httpd.result = None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def run(httpd, driver, budget_s):
    httpd.result = None
    url = f'http://127.0.0.1:{httpd.server_address[1]}/real_nucleus_numerics/{driver.name}'
    profile = tempfile.mkdtemp(prefix='ladder-')
    chrome = subprocess.Popen(
        [CHROME, '--new-window', f'--user-data-dir={profile}', '--no-first-run',
         '--no-default-browser-check', '--window-position=3000,3000',
         '--window-size=500,400',
         # without this an offscreen window counts as occluded, Chrome stops
         # requestAnimationFrame, p5's draw() never runs and nothing is computed
         '--disable-features=CalculateNativeWinOcclusion',
         '--autoplay-policy=no-user-gesture-required', url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        while httpd.result is None and time.time() - t0 < budget_s + 45:
            time.sleep(2)
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=15)
        except subprocess.TimeoutExpired:
            chrome.kill()
        shutil.rmtree(profile, ignore_errors=True)
    return httpd.result


def settled(trace, tol=2e-3):
    """Judge convergence from the trace: relative drift over its last quarter."""
    if len(trace) < 12:
        return False, float('nan')
    tail = trace[-max(4, len(trace) // 4):]
    span = (max(tail) - min(tail)) / max(abs(trace[-1]), 1e-9)
    return span < tol, span


def richardson(points):
    """E(h) = E0 + C h^2 with h = 1/N; least squares on x = 1/N^2 -> intercept."""
    if len(points) < 2:
        return (points[0][1] if points else float('nan')), float('nan')
    xs = [1.0 / (n * n) for n, _ in points]
    ys = [e for _, e in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    e0 = my - slope * mx
    resid = max(abs(y - (e0 + slope * x)) for x, y in zip(xs, ys))
    return e0, resid


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(st):
    STATE.write_text(json.dumps(st, indent=2))


def phase_scan(args, grid, budget, cases):
    """Phase A: minimise E over the overall size factor lambda."""
    st = load(); st.setdefault('scan', {})
    httpd = serve()
    for label in cases:
        page, _ = CASES[label]
        best = None
        print(f'--- scan {label} (grid {grid})', flush=True)
        for lam in LAMBDAS:
            drv = build(label, page, grid, lam, budget)
            r = run(httpd, drv, budget)
            if r is None or r.get('E') is None:
                if r is None:
                    print(f'    lambda={lam:.2f}  NO POST-BACK at all', flush=True)
                else:
                    print(f'    lambda={lam:.2f}  posted but E=None | gpu: {r.get("gpu")} | '
                          f'prevE={r.get("prevE")} | trace={len(r.get("trace") or [])} | err={(r.get("err") or [])[:2]}', flush=True)
                continue
            ok, span = settled(r['trace'])
            print(f'    lambda={lam:.2f}  E={r["E"]:>10.4f}  {"settled" if ok else "UNSETTLED"} (drift {span:.1e})', flush=True)
            if r['E'] is not None and (best is None or r['E'] < best[1]):
                best = (lam, r['E'])
        if best:
            st['scan'][label] = {'lambda': best[0], 'E': best[1], 'grid': grid}
            print(f'    -> lambda* = {best[0]:.2f}', flush=True)
        save(st)


def phase_ladder(args, grids, budget, cases):
    """Phase B: grid ladder at lambda*, extrapolated to h -> 0."""
    st = load(); st.setdefault('ladder', {})
    httpd = serve()
    for label in cases:
        page, _ = CASES[label]
        lam = st.get('scan', {}).get(label, {}).get('lambda', 1.0)
        pts = []
        print(f'--- ladder {label} (lambda {lam:.2f})', flush=True)
        for N in grids:
            drv = build(label, page, N, lam, budget)
            r = run(httpd, drv, budget)
            if r is None or r.get('E') is None:
                why = (r or {}).get('err') or ['no post-back within the budget']
                print(f'    N={N}  NO RESULT: {why[:2]}', flush=True); continue
            ok, span = settled(r['trace'])
            print(f'    N={N:>4}  E={r["E"]:>10.4f}  {"settled" if ok else "UNSETTLED"} (drift {span:.1e})', flush=True)
            if ok:
                pts.append((N, r['E']))
        if pts:
            e0, resid = richardson(pts)
            st['ladder'][label] = {'lambda': lam, 'points': pts, 'E0': e0, 'resid': resid}
            print(f'    -> h->0: E = {e0:.4f} +- {resid:.4f}', flush=True)
        save(st)


def phase_report():
    st = load()
    lad = st.get('ladder', {})
    if REFERENCE not in lad:
        print(f'no extrapolated value for the reference ({REFERENCE}); run the ladder first')
        return
    Eref = lad[REFERENCE]['E0']
    print(f'\nreference: {REFERENCE}, E = {Eref:.4f} (h->0)\n')
    print(f'{"case":>9} {"E (h->0)":>12} {"+-":>8} {"E/E_ref":>9} {"exp":>8} {"dev":>8}  lambda*')
    for label in CASES:
        if label not in lad:
            continue
        d = lad[label]
        ratio = d['E0'] / Eref
        exp = CASES[label][1] / CASES[REFERENCE][1]
        dev = 100 * (ratio - exp) / exp
        print(f'{label:>9} {d["E0"]:>12.4f} {d["resid"]:>8.4f} {ratio:>9.3f} {exp:>8.3f} {dev:>7.1f}%  {d["lambda"]:.2f}')
    print('\nlambda* is the size the nucleus chose, not one imposed: the trend across A')
    print('is the size law, and is an output of this protocol rather than an input.')


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); return
    phase = args.pop(0)
    grid, grids, budget = 160, [120, 160, 200, 240], 300
    if '--grid' in args:
        i = args.index('--grid'); grid = int(args[i + 1]); del args[i:i + 2]
    if '--grids' in args:
        i = args.index('--grids'); grids = [int(x) for x in args[i + 1].split(',')]; del args[i:i + 2]
    if '--thresh' in args:
        i = args.index('--thresh'); globals()['THRESH'] = float(args[i + 1]); del args[i:i + 2]
    if '--budget' in args:
        i = args.index('--budget'); budget = int(args[i + 1]); del args[i:i + 2]
    cases = args or list(CASES)

    if not (HERE / 'p5.min.js').exists():
        subprocess.run(['curl', '-sSo', str(HERE / 'p5.min.js'),
                        'https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js'], check=True)

    if phase == 'scan':
        phase_scan(args, grid, budget, cases)
    elif phase == 'ladder':
        phase_ladder(args, grids, budget, cases)
    elif phase == 'report':
        phase_report()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
