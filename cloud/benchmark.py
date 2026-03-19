"""Benchmark: measure ms/step on A100 for hairpin, compare with M4 estimate."""
import modal, math, json, time

app = modal.App("realqm-benchmark")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("cupy-cuda12x", "numpy")
    .add_local_file("cloud/solver.py", "/root/solver.py")
)


def build_hairpin_config():
    gridN = 300; scr = 80; h = scr / gridN; N2 = gridN // 2
    A = 1.8897; atoms = []

    def addAtom(au_x, au_y, Z, el):
        atoms.append({
            'i': round(N2 + au_x / h), 'j': round(N2 + au_y / h), 'k': N2,
            'Z': Z, 'el': el,
            'rc': {'H': 0, 'C': 0.3, 'N': 0.3, 'O': 0.4}.get(el, 0),
            'x': au_x, 'y': au_y,
        })

    nRes = 12; resAdv = 3.4 * A; zig = 0.5 * A
    bendAngle = 90 * math.pi / 180
    arcLen = (nRes - 1) * resAdv; arcRadius = arcLen / bendAngle

    bb = []
    for r in range(nRes):
        t = (r / (nRes - 1) - 0.5) * bendAngle
        cx = arcRadius * math.sin(t); cy = arcRadius * (1 - math.cos(t))
        tx = math.cos(t); ty = math.sin(t)
        nx = -ty; ny = tx
        zigSign = 1 if (r % 2 == 0) else -1
        bb.append({
            'N': [cx - 0.35*resAdv*tx + zigSign*zig*0.6*nx, cy - 0.35*resAdv*ty + zigSign*zig*0.6*ny],
            'Ca': [cx, cy - zigSign*zig*0.4*ny],
            'C': [cx + 0.35*resAdv*tx + zigSign*zig*0.3*nx, cy + 0.35*resAdv*ty + zigSign*zig*0.3*ny],
        })

    bNH = 1.01*A; bCH = 1.09*A; bCO = 1.24*A
    for r in range(nRes):
        res = bb[r]; Np, Cap, Cp = res['N'], res['Ca'], res['C']
        nca_x, nca_y = Cap[0]-Np[0], Cap[1]-Np[1]
        nca_l = math.sqrt(nca_x**2 + nca_y**2)
        nca_px, nca_py = -nca_y/nca_l, nca_x/nca_l
        cac_x, cac_y = Cp[0]-Cap[0], Cp[1]-Cap[1]
        cac_l = math.sqrt(cac_x**2 + cac_y**2)
        cac_px, cac_py = -cac_y/cac_l, cac_x/cac_l
        addAtom(Np[0], Np[1], 3, 'N')
        nhSide = -1 if r < 6 else 1
        addAtom(Np[0]+nhSide*nca_px*bNH, Np[1]+nhSide*nca_py*bNH, 1, 'H')
        if r == 0: addAtom(Np[0]-nhSide*nca_px*bNH, Np[1]-nhSide*nca_py*bNH, 1, 'H')
        addAtom(Cap[0], Cap[1], 4, 'C')
        addAtom(Cap[0]+cac_px*bCH, Cap[1]+cac_py*bCH, 1, 'H')
        addAtom(Cap[0]-cac_px*bCH, Cap[1]-cac_py*bCH, 1, 'H')
        addAtom(Cp[0], Cp[1], 4, 'C')
        oSide = ((1 if r%2==0 else -1) if r<6 else (-1 if r%2==0 else 1))
        addAtom(Cp[0]+oSide*cac_px*bCO, Cp[1]+oSide*cac_py*bCO, 2, 'O')

    lastC = bb[nRes-1]['C']; lastCa = bb[nRes-1]['Ca']
    lc_x = lastC[0]-lastCa[0]; lc_y = lastC[1]-lastCa[1]
    lc_l = math.sqrt(lc_x**2+lc_y**2)
    oh_x = lastC[0]+lc_x/lc_l*1.34*A; oh_y = lastC[1]+lc_y/lc_l*1.34*A
    addAtom(oh_x, oh_y, 2, 'O')
    addAtom(oh_x+lc_x/lc_l*0.97*A, oh_y+lc_y/lc_l*0.97*A, 1, 'H')
    nProtein = len(atoms)

    ohBond = 0.96*A; halfAngle = 52.25*math.pi/180
    waterSpacing = 2.8*A; minDist = 2.5*A; shellDist = 8*A; gridLimit = scr/2-3
    nWater = 0; wx = -gridLimit
    while wx <= gridLimit:
        wy = -gridLimit
        while wy <= gridLimit:
            tooClose = False; nearProtein = False
            for p in range(nProtein):
                dx = wx-atoms[p]['x']; dy = wy-atoms[p]['y']; d2 = dx*dx+dy*dy
                if d2 < minDist*minDist: tooClose = True; break
                if d2 < shellDist*shellDist: nearProtein = True
            if not tooClose and nearProtein:
                angle = (wx*7.3+wy*11.1)%(2*math.pi)
                addAtom(wx, wy, 2, 'O')
                addAtom(wx+ohBond*math.cos(angle+halfAngle), wy+ohBond*math.sin(angle+halfAngle), 1, 'H')
                addAtom(wx+ohBond*math.cos(angle-halfAngle), wy+ohBond*math.sin(angle-halfAngle), 1, 'H')
                nWater += 1
            wy += waterSpacing
        wx += waterSpacing
    for a in atoms: a.pop('x', None); a.pop('y', None)
    return {'gridN': gridN, 'screen': scr, 'atoms': atoms, 'dynamics': False}


@app.function(gpu="A100", image=image, timeout=300)
def benchmark_a100(config: dict) -> dict:
    import sys; sys.path.insert(0, '/root')
    import cupy as cp
    from solver import RealQMSolver

    solver = RealQMSolver(config)
    solver.initialize()

    # Warmup
    for _ in range(100):
        solver._poisson_step()
        solver.U, solver.U2 = solver.U2, solver.U

    # Benchmark: 1000 pure ITP steps
    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    STEPS = 1000
    for step in range(1, STEPS + 1):
        solver._poisson_step()
        import numpy as np
        _update_kernel = None  # use solver's run method instead

    # Actually just use run() for clean timing
    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    results = solver.run(total_steps=2000, report_interval=500, norm_interval=20)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - t0

    ms_per_step = elapsed / 2000 * 1000

    return {
        'nElec': solver.nElec,
        'gridN': solver.NN,
        'ms_per_step': float(ms_per_step),
        'steps_per_sec': float(1000 / ms_per_step),
        'snapshots': results,
    }


@app.local_entrypoint()
def main():
    config = build_hairpin_config()
    nElec = sum(1 for a in config['atoms'] if a['Z'] > 0)
    print(f"Benchmarking: {nElec} electrons, {config['gridN']}^3 grid on A100...")

    result = benchmark_a100.remote(config)

    ms = result['ms_per_step']
    # M4 Air estimate: memory-bound at ~100 GB/s
    # Per step: ~nElec operations on 301^3 grid, each ~108MB read+write
    # Rough: 2 * 27M * 4 * (2 + nElec*0.01) / 100e9 * 1000
    m4_est = nElec * 27e6 * 8 / 100e9 * 1000 * 3  # rough estimate with overhead

    print(f"\n{'='*50}")
    print(f"BENCHMARK: {nElec} electrons, {config['gridN']}^3 grid")
    print(f"{'='*50}")
    print(f"A100:     {ms:.1f} ms/step  ({result['steps_per_sec']:.0f} steps/sec)")
    print(f"M4 est:   {m4_est:.0f} ms/step  ({1000/m4_est:.1f} steps/sec)")
    print(f"Speedup:  {m4_est/ms:.0f}x")
    print(f"\nEnergy convergence:")
    for s in result['snapshots']:
        print(f"  step {s['step']}: E={s['E']:.4f}")
