"""Is the He number converged, and is the beta-sensitivity robust to the mesh?"""
import importlib.util, sys

print("He: exact -2.9037 Ha, Z_eff variational -2.8477", flush=True)
print(f"{'N':>5} {'h':>7} {'E(b=0)':>10} {'E(b=1.146)':>12} {'shift':>8}", flush=True)
for N in (64, 96, 128):
    sys.argv = ['x', str(N), '14']
    spec = importlib.util.spec_from_file_location(f"he{N}", "robin_interface_he.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # __name__ != '__main__', so main() does not run
    E0, _ = m.solve(0.0, steps=6000)
    E1, _ = m.solve(1.146, steps=6000)
    print(f"{N:>5} {m.h:>7.3f} {E0:>10.4f} {E1:>12.4f} {E1-E0:>8.4f}", flush=True)
