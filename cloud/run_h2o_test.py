"""Quick validation: H2O on 100^3 grid, 3 electrons. Should converge to E ~ -17 Ha."""
import modal, math, json

app = modal.App("realqm-h2o-test")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("cupy-cuda12x", "numpy")
    .add_local_file("cloud/solver.py", "/root/solver.py")
)

@app.function(gpu="A100", image=image, timeout=300)
def run_h2o(config: dict) -> dict:
    import sys; sys.path.insert(0, '/root')
    from solver import RealQMSolver
    solver = RealQMSolver(config)
    results = solver.run(
        total_steps=config.get('steps', 5000),
        report_interval=config.get('report_interval', 500),
    )
    return {'snapshots': results, 'final_E': float(solver.E)}

@app.local_entrypoint()
def main():
    NN = 100; scr = 10; h = scr / NN; N2 = NN // 2
    angle = 104.5 * math.pi / 180; rOH = 1.81
    Oi, Oj = N2, N2
    H1i = round(Oi + rOH * math.cos(math.pi/2 + angle/2) / h)
    H1j = round(Oj - rOH * math.sin(math.pi/2 + angle/2) / h)
    H2i = round(Oi + rOH * math.cos(math.pi/2 - angle/2) / h)
    H2j = round(Oj - rOH * math.sin(math.pi/2 - angle/2) / h)

    config = {
        'gridN': NN, 'screen': scr, 'steps': 20000, 'report_interval': 2000,
        'atoms': [
            {'i': H1i, 'j': H1j, 'Z': 1, 'el': 'H', 'rc': 0},
            {'i': Oi, 'j': Oj, 'Z': 2, 'el': 'O', 'rc': 0.4},
            {'i': H2i, 'j': H2j, 'Z': 1, 'el': 'H', 'rc': 0},
        ],
    }
    print("Running H2O validation on A100...")
    result = run_h2o.remote(config)
    print(f"\nH2O Results (expected E ~ -17 Ha):")
    for s in result['snapshots']:
        print(f"  step {s['step']:5d}: E={s['E']:.6f}  T={s['E_T']:.4f}  V_eK={s['E_eK']:.4f}  V_ee={s['E_ee']:.4f}  V_KK={s['E_KK']:.4f}")
