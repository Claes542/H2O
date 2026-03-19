"""Debug H2O: check norms, U max, domain sizes."""
import modal, math
app = modal.App("realqm-h2o-debug")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("cupy-cuda12x", "numpy")
    .add_local_file("cloud/solver.py", "/root/solver.py")
)

@app.function(gpu="A100", image=image, timeout=300)
def run_debug(config: dict) -> dict:
    import sys; sys.path.insert(0, '/root')
    import cupy as cp
    import numpy as np
    from solver import RealQMSolver
    solver = RealQMSolver(config)
    solver.initialize()

    # Check initial state
    U_cpu = solver.U.get()
    label_cpu = solver.label.get()
    K_cpu = solver.K_buf.get()
    P_cpu = solver.P.get()

    info = {}
    info['U_max'] = float(U_cpu.max())
    info['U_min'] = float(U_cpu.min())
    info['K_max'] = float(K_cpu.max())
    info['K_min'] = float(K_cpu.min())
    info['P_max'] = float(P_cpu.max())
    info['P_min'] = float(P_cpu.min())

    for n in range(solver.nAtoms):
        if solver.Z_eff[n] > 0:
            mask = (label_cpu == n)
            domain_size = int(mask.sum())
            u_domain = U_cpu[mask]
            norm = float((u_domain**2).sum() * solver.h3)
            info[f'atom{n}_Z'] = float(solver.Z_eff[n])
            info[f'atom{n}_domain_size'] = domain_size
            info[f'atom{n}_norm'] = norm
            info[f'atom{n}_U_max'] = float(u_domain.max())

    # Run 1000 steps and check
    results = solver.run(total_steps=1000, report_interval=200)

    U_cpu = solver.U.get()
    P_cpu = solver.P.get()
    info['after_U_max'] = float(U_cpu.max())
    info['after_P_max'] = float(P_cpu.max())
    info['after_P_min'] = float(P_cpu.min())

    for n in range(solver.nAtoms):
        if solver.Z_eff[n] > 0:
            mask = (label_cpu == n)
            u_domain = U_cpu[mask]
            norm = float((u_domain**2).sum() * solver.h3)
            info[f'atom{n}_after_norm'] = norm

    return {'info': info, 'snapshots': results}

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
        'gridN': NN, 'screen': scr, 'steps': 1000, 'report_interval': 200,
        'atoms': [
            {'i': H1i, 'j': H1j, 'Z': 1, 'el': 'H', 'rc': 0},
            {'i': Oi, 'j': Oj, 'Z': 2, 'el': 'O', 'rc': 0.4},
            {'i': H2i, 'j': H2j, 'Z': 1, 'el': 'H', 'rc': 0},
        ],
    }
    result = run_debug.remote(config)
    print("\n=== Debug Info ===")
    for k, v in sorted(result['info'].items()):
        print(f"  {k}: {v}")
    print("\n=== Energy ===")
    for s in result['snapshots']:
        print(f"  step {s['step']}: E={s['E']:.6f}  T={s['E_T']:.4f}  V_eK={s['E_eK']:.4f}  V_ee={s['E_ee']:.4f}")
