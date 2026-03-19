"""
Submit the bent solvated hairpin job to Modal cloud GPU.

Usage:
  modal run cloud/run_hairpin.py
"""

import modal
import math
import json

app = modal.App("realqm-hairpin")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("cupy-cuda12x", "numpy")
    .add_local_file("cloud/solver.py", "/root/solver.py")
)


def build_hairpin_config():
    """Build the bent solvated hairpin atom config (matches hairpin_bent_solvated.html)."""
    gridN = 300
    scr = 80
    h = scr / gridN
    N2 = gridN // 2
    A = 1.8897  # angstrom to au

    atoms = []

    def addAtom(au_x, au_y, Z, el):
        atoms.append({
            'i': round(N2 + au_x / h),
            'j': round(N2 + au_y / h),
            'k': N2,
            'Z': Z, 'el': el,
            'rc': {'H': 0, 'C': 0.3, 'N': 0.3, 'O': 0.4}.get(el, 0),
            'x': au_x, 'y': au_y,
        })

    nRes = 12
    resAdv = 3.4 * A
    zig = 0.5 * A

    # U-shape geometry with tanh-blended turn (matches local HTML)
    openingAngle = 30 * math.pi / 180  # angle between strands at top
    hingeIdx = 5.5
    turnWidth = 2.5

    angle1 = math.pi / 2 + openingAngle / 2
    angle2 = math.pi / 2 - openingAngle / 2
    startAngle = -(math.pi - angle1)
    endAngle = angle2

    angles = []
    for r in range(nRes):
        blend = 0.5 * (1 + math.tanh((r - hingeIdx) / turnWidth))
        angles.append(startAngle * (1 - blend) + endAngle * blend)

    positions = [{'x': 0, 'y': 0}]
    for r in range(1, nRes):
        avgAngle = (angles[r - 1] + angles[r]) / 2
        positions.append({
            'x': positions[r-1]['x'] + resAdv * math.cos(avgAngle),
            'y': positions[r-1]['y'] + resAdv * math.sin(avgAngle),
        })

    sumX = sum(p['x'] for p in positions)
    sumY = sum(p['y'] for p in positions)
    cenX, cenY = sumX / nRes, sumY / nRes

    bb = []
    for r in range(nRes):
        cx = positions[r]['x'] - cenX
        cy = positions[r]['y'] - cenY
        tlx = math.cos(angles[r])
        tly = math.sin(angles[r])
        nx = -tly
        ny = tlx
        zigSign = 1 if (r % 2 == 0) else -1
        yShift = 15

        bb.append({
            'N': [cx - 0.35 * resAdv * tlx + zigSign * zig * 0.6 * nx,
                  cy - 0.35 * resAdv * tly + zigSign * zig * 0.6 * ny + yShift],
            'Ca': [cx, cy - zigSign * zig * 0.4 * ny + yShift],
            'C': [cx + 0.35 * resAdv * tlx + zigSign * zig * 0.3 * nx,
                  cy + 0.35 * resAdv * tly + zigSign * zig * 0.3 * ny + yShift],
        })

    bNH = 1.01 * A
    bCH = 1.09 * A
    bCO = 1.24 * A

    for r in range(nRes):
        res = bb[r]
        Np, Cap, Cp = res['N'], res['Ca'], res['C']
        nca_x, nca_y = Cap[0] - Np[0], Cap[1] - Np[1]
        nca_l = math.sqrt(nca_x**2 + nca_y**2)
        nca_px, nca_py = -nca_y / nca_l, nca_x / nca_l
        cac_x, cac_y = Cp[0] - Cap[0], Cp[1] - Cap[1]
        cac_l = math.sqrt(cac_x**2 + cac_y**2)
        cac_px, cac_py = -cac_y / cac_l, cac_x / cac_l

        addAtom(Np[0], Np[1], 3, 'N')
        nhSide = -1 if r < 6 else 1
        addAtom(Np[0] + nhSide * nca_px * bNH, Np[1] + nhSide * nca_py * bNH, 1, 'H')
        if r == 0:
            addAtom(Np[0] - nhSide * nca_px * bNH, Np[1] - nhSide * nca_py * bNH, 1, 'H')
        addAtom(Cap[0], Cap[1], 4, 'C')
        addAtom(Cap[0] + cac_px * bCH, Cap[1] + cac_py * bCH, 1, 'H')
        addAtom(Cap[0] - cac_px * bCH, Cap[1] - cac_py * bCH, 1, 'H')
        addAtom(Cp[0], Cp[1], 4, 'C')
        oSide = ((1 if r % 2 == 0 else -1) if r < 6 else (-1 if r % 2 == 0 else 1))
        addAtom(Cp[0] + oSide * cac_px * bCO, Cp[1] + oSide * cac_py * bCO, 2, 'O')

    # C-terminal OH
    lastC = bb[nRes - 1]['C']
    lastCa = bb[nRes - 1]['Ca']
    lc_x = lastC[0] - lastCa[0]
    lc_y = lastC[1] - lastCa[1]
    lc_l = math.sqrt(lc_x**2 + lc_y**2)
    oh_x = lastC[0] + lc_x / lc_l * 1.34 * A
    oh_y = lastC[1] + lc_y / lc_l * 1.34 * A
    addAtom(oh_x, oh_y, 2, 'O')
    addAtom(oh_x + lc_x / lc_l * 0.97 * A, oh_y + lc_y / lc_l * 0.97 * A, 1, 'H')

    nProtein = len(atoms)

    # Solvation shell
    ohBond = 0.96 * A
    halfAngle = 52.25 * math.pi / 180
    waterSpacing = 2.8 * A
    minDist = 2.5 * A
    shellDist = 8 * A
    gridLimit = scr / 2 - 3

    nWater = 0
    wx = -gridLimit
    while wx <= gridLimit:
        wy = -gridLimit
        while wy <= gridLimit:
            tooClose = False
            nearProtein = False
            for p in range(nProtein):
                dx = wx - atoms[p]['x']
                dy = wy - atoms[p]['y']
                d2 = dx*dx + dy*dy
                if d2 < minDist * minDist:
                    tooClose = True
                    break
                if d2 < shellDist * shellDist:
                    nearProtein = True
            if not tooClose and nearProtein:
                angle = (wx * 7.3 + wy * 11.1) % (2 * math.pi)
                addAtom(wx, wy, 2, 'O')
                addAtom(wx + ohBond * math.cos(angle + halfAngle),
                        wy + ohBond * math.sin(angle + halfAngle), 1, 'H')
                addAtom(wx + ohBond * math.cos(angle - halfAngle),
                        wy + ohBond * math.sin(angle - halfAngle), 1, 'H')
                nWater += 1
            wy += waterSpacing
        wx += waterSpacing

    # Clean up x/y fields (not needed by solver)
    for a in atoms:
        a.pop('x', None)
        a.pop('y', None)

    print(f"Built hairpin: {nProtein} protein + {nWater} waters = {len(atoms)} atoms")

    return {
        'gridN': gridN,
        'screen': scr,
        'atoms': atoms,
        'steps': 10000,
        'report_interval': 1000,
        'dynamics': True,
        'forceScale': 500.0,
        'fold_atoms': [0, 45, 83],  # N-term, mid Ca, C-term
    }


@app.function(gpu="A100", image=image, timeout=1800)
def run_hairpin(config: dict) -> dict:
    import sys
    sys.path.insert(0, '/root')
    from solver import RealQMSolver

    solver = RealQMSolver(config)
    if 'fold_atoms' in config:
        solver.set_fold_atoms(config['fold_atoms'])

    print(f"Running: {solver.nElec} electrons, {solver.NN}^3 grid")
    results = solver.run(
        total_steps=config.get('steps', 10000),
        report_interval=config.get('report_interval', 500),
    )

    return {
        'snapshots': results,
        'final_E': float(solver.E),
        'final_E_bind': float(solver.E - solver.E_atoms_sum),
        'final_nucPos': [[float(x) for x in row] for row in solver.atom_pos],
    }


@app.local_entrypoint()
def main():
    config = build_hairpin_config()
    print(f"\nSubmitting to A100...")
    result = run_hairpin.remote(config)

    print(f"\n{'='*60}")
    print(f"RESULTS: Bent Solvated Hairpin")
    print(f"{'='*60}")
    print(f"Final E = {result['final_E']:.6f} Ha")
    print(f"E_bind  = {result['final_E_bind']:.6f} Ha")
    print(f"\nTimeline:")
    for s in result['snapshots']:
        fold = f"  fold={s['fold_angle']:.1f}°" if 'fold_angle' in s else ""
        print(f"  step {s['step']:6d}: E={s['E']:.6f}  "
              f"E_bind={s['E_bind']:.6f}  "
              f"({s['ms_per_step']:.1f} ms/step){fold}")

    # Save results
    with open('cloud/hairpin_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to cloud/hairpin_results.json")
