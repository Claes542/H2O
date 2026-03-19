"""
Modal.com deployment for RealQM solver.

Usage:
  1. Install: pip install modal
  2. Auth:    modal token new
  3. Deploy:  modal deploy cloud/app.py
  4. Run:     curl -X POST https://<your-app>.modal.run/solve -d @job.json

Or run locally for testing:
  modal run cloud/app.py
"""

import modal

app = modal.App("realqm-solver")

# GPU image with CuPy
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("cupy-cuda12x", "numpy", "fastapi[standard]")
)


@app.function(gpu="A100", image=image, timeout=600)
def solve(config: dict) -> dict:
    """Run a quantum simulation job on GPU. Returns energy snapshots."""
    from solver import RealQMSolver

    solver = RealQMSolver(config)

    if 'fold_atoms' in config:
        solver.set_fold_atoms(config['fold_atoms'])

    total_steps = config.get('steps', 10000)
    report_interval = config.get('report_interval', 500)

    print(f"Starting: {solver.nElec} electrons, {solver.NN}^3 grid, "
          f"{total_steps} steps, screen={solver.screen} au")

    results = solver.run(
        total_steps=total_steps,
        report_interval=report_interval,
    )

    return {
        'config': {
            'gridN': solver.NN,
            'screen': solver.screen,
            'nElec': solver.nElec,
            'nAtoms': solver.nAtoms,
        },
        'snapshots': results,
        'final_nucPos': solver.atom_pos.tolist(),
        'final_E': solver.E,
        'final_E_bind': solver.E - solver.E_atoms_sum,
    }


@app.function(gpu="A100", image=image, timeout=600)
def sweep(config: dict) -> dict:
    """Run a bond-length sweep. Returns energy vs bond length."""
    from solver import RealQMSolver
    import numpy as np

    bond_lengths = config['bond_lengths']
    base_atoms = config['atoms']
    sweep_results = []

    for i, r in enumerate(bond_lengths):
        # Rebuild atoms with new bond length
        atoms = config['make_atoms'](r) if callable(config.get('make_atoms')) else base_atoms
        cfg = {**config, 'atoms': atoms}
        solver = RealQMSolver(cfg)

        steps = config.get('steps', 8000)
        print(f"\n=== Sweep [{i+1}/{len(bond_lengths)}] R={r:.2f} au ===")
        results = solver.run(total_steps=steps, report_interval=steps)

        sweep_results.append({
            'bond': r,
            'E': solver.E,
            'E_T': solver.E_T,
            'E_eK': solver.E_eK,
            'E_ee': solver.E_ee,
            'E_KK': solver.E_KK,
        })

    return {'sweep': sweep_results}


# --- Web endpoint for browser-based job submission ---

@app.function(gpu="A100", image=image, timeout=600)
@modal.web_endpoint(method="POST")
def solve_web(job: dict) -> dict:
    """HTTP endpoint: POST job config, get results back."""
    return solve.local(job)


# --- CLI test ---

@app.local_entrypoint()
def main():
    """Test with a small H2O molecule."""
    import json, math

    NN = 100
    scr = 10
    h = scr / NN
    N2 = NN // 2
    angle = 104.5 * math.pi / 180
    rOH = 1.81  # au

    Oi, Oj = N2, N2
    H1i = round(Oi + rOH * math.cos(math.pi/2 + angle/2) / h)
    H1j = round(Oj - rOH * math.sin(math.pi/2 + angle/2) / h)
    H2i = round(Oi + rOH * math.cos(math.pi/2 - angle/2) / h)
    H2j = round(Oj - rOH * math.sin(math.pi/2 - angle/2) / h)

    config = {
        'gridN': NN,
        'screen': scr,
        'steps': 5000,
        'report_interval': 1000,
        'atoms': [
            {'i': H1i, 'j': H1j, 'Z': 1, 'el': 'H', 'rc': 0},
            {'i': Oi, 'j': Oj, 'Z': 2, 'el': 'O', 'rc': 0.4},
            {'i': H2i, 'j': H2j, 'Z': 1, 'el': 'H', 'rc': 0},
        ],
    }

    print("Submitting H2O job to cloud GPU...")
    result = solve.remote(config)
    print(f"\nFinal E = {result['final_E']:.6f} Ha")
    print(f"E_bind = {result['final_E_bind']:.6f} Ha")
    print(f"\nSnapshots:")
    for s in result['snapshots']:
        print(f"  step {s['step']}: E={s['E']:.6f} ({s['ms_per_step']:.1f} ms/step)")
