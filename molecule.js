// Molecule Quantum Simulation — WebGPU Compute Shaders
// Up to 10 atoms placed interactively, 3D geometry

const NN = window.USER_NN || 200;
const S = NN + 1;
const S2 = S * S;
const S3 = S * S * S;
const N2 = Math.round(NN / 2);
const MAX_ATOMS = 100;
const _uz = window.USER_Z || [2, 3, 1, 0, 0];
while (_uz.length < MAX_ATOMS) _uz.push(0);
const NELEC = _uz.filter(z => z > 0).length || 3;
const NRED = NELEC + 3;  // N norms + T + V_eK + V_ee
const r_cut = window.USER_RC || [0.5, 0.3, 0.1, 0, 0];
while (r_cut.length < MAX_ATOMS) r_cut.push(0);
let R_out = 2.0;   // au, outer w cutoff
let Z = [..._uz];
let Ne = [..._uz];
const Z_orig = [..._uz];
const Ne_orig = [..._uz];

// Atom positions from interactive placement or defaults
const _atoms = window.USER_ATOMS || [
  { i: 80, j: 120, Z: 2, el: 'O' },
  { i: 100, j: 77, Z: 3, el: 'N' },
  { i: 120, j: 120, Z: 1, el: 'H' },
  { i: 100, j: 100, Z: 0, el: '' },
  { i: 100, j: 100, Z: 0, el: '' }
];
while (_atoms.length < MAX_ATOMS) _atoms.push({ i: N2, j: N2, Z: 0, el: '' });
const atomLabels = _atoms.map(a => a.el);
let nucPos = _atoms.map(a => [a.i, a.j, a.k !== undefined ? a.k : N2]);
const molNucPos = nucPos.map(p => [...p]);

let E_min = Infinity;
let screenAu = window.USER_SCREEN || 10;
let hv = screenAu / NN, h2v = hv * hv, h3v = hv * hv * hv;
const dv = 0.12;
let dtv = dv * h2v, half_dv = 0.5 * dv;
const PX = 400 / NN;
const INTERIOR = (NN - 1) * (NN - 1) * (NN - 1);
const STEPS_PER_FRAME = 100;
const W_STEPS_PER_FRAME = 100;
const NORM_INTERVAL = 20;
const POISSON_INTERVAL = 50;

// Multigrid coarse grid
if (NN % 2 !== 0) throw new Error("NN must be even for multigrid");
const NC = Math.floor(NN / 2);
const SC = NC + 1, SC2 = SC * SC, SC3 = SC * SC * SC;
const INTERIOR_C = (NC - 1) * (NC - 1) * (NC - 1);

// Reduce workgroup size: NRED * REDUCE_WG * 4 must be <= 16384 bytes
const REDUCE_WG = 1 << Math.floor(Math.log2(Math.min(128, Math.floor(4096 / NRED))));

// ===== WGSL SHADERS =====

// Param struct: 16 common fields = 64 bytes. Atom data in separate storage buffer.
const PARAM_BYTES = 64;
const paramStructWGSL = `
struct P {
  NN: u32, S: u32, S2: u32, S3: u32,
  N2: u32, voronoi: f32, R_out: f32, TWO_PI: f32,
  h: f32, h2: f32, inv_h: f32, inv_h2: f32,
  dt: f32, half_d: f32, h3: f32, _pad0: f32,
}`;

const ATOM_STRIDE = 8; // 8 f32s per atom (posI, posJ, posK, Z, rc, pad, pad, pad)
const ATOM_BUF_BYTES = MAX_ATOMS * ATOM_STRIDE * 4;
const atomStructWGSL = `
struct Atom {
  posI: u32, posJ: u32, posK: u32, Z: f32,
  rc: f32, _p0: f32, _p1: f32, _p2: f32,
}`;

// U update — single density field, smooth W for weak Neumann BC
const updateU_WGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Ui: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<f32>;
@group(0) @binding(4) var<storage, read> Pi: array<f32>;
@group(0) @binding(5) var<storage, read_write> Uo: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }

  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let wc = W[id];
  // Smooth face weights: geometric mean, scaled to [0.5, 1.0] for weak Neumann
  let wip = (sqrt(wc * W[id + p.S2]) + 1.0) * 0.5;
  let wim = (sqrt(wc * W[id - p.S2]) + 1.0) * 0.5;
  let wjp = (sqrt(wc * W[id + p.S])  + 1.0) * 0.5;
  let wjm = (sqrt(wc * W[id - p.S])  + 1.0) * 0.5;
  let wkp = (sqrt(wc * W[id + 1u])   + 1.0) * 0.5;
  let wkm = (sqrt(wc * W[id - 1u])   + 1.0) * 0.5;

  let uc  = Ui[id];
  let uip = Ui[id + p.S2]; let uim = Ui[id - p.S2];
  let ujp = Ui[id + p.S];  let ujm = Ui[id - p.S];
  let ukp = Ui[id + 1u];   let ukm = Ui[id - 1u];

  Uo[id] = uc
    + p.half_d * ((uip - uc) * wip - (uc - uim) * wim)
    + p.half_d * ((ujp - uc) * wjp - (uc - ujm) * wjm)
    + p.half_d * ((ukp - uc) * wkp - (uc - ukm) * wkm)
    + p.dt * (K[id] - 2.0 * Pi[id]) * uc;
}
`;

// W update — diffuse W only at boundary cells, interior pinned to 1.0
const updateW_WGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Wi: array<f32>;
@group(0) @binding(2) var<storage, read_write> Wo: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }

  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let myLabel = label[id];
  let sameAll = f32(label[id + p.S2] == myLabel)
              * f32(label[id - p.S2] == myLabel)
              * f32(label[id + p.S]  == myLabel)
              * f32(label[id - p.S]  == myLabel)
              * f32(label[id + 1u]   == myLabel)
              * f32(label[id - 1u]   == myLabel);

  // Interior cells (all neighbors same label): pin to 1.0
  if (sameAll > 0.5) {
    Wo[id] = 1.0;
    return;
  }

  // Boundary cells: diffuse with clamping
  let wc = Wi[id];
  let lap = Wi[id + p.S2] + Wi[id - p.S2]
          + Wi[id + p.S]  + Wi[id - p.S]
          + Wi[id + 1u]   + Wi[id - 1u] - 6.0 * wc;
  Wo[id] = clamp(wc + 0.1 * lap, 0.0, 1.0);
}
`;

// Jacobi smoother for single-field Poisson: Lap(P) = -2*pi*rho
const jacobiSmoothWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pin: array<f32>;
@group(0) @binding(2) var<storage, read_write> Pout: array<f32>;
@group(0) @binding(3) var<storage, read> rhoTotal: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let Pc = Pin[id];
  let sum_nbr = Pin[id + p.S2] + Pin[id - p.S2]
              + Pin[id + p.S]  + Pin[id - p.S]
              + Pin[id + 1u]   + Pin[id - 1u];
  let rhs = p.h2 * p.TWO_PI * rhoTotal[id];
  Pout[id] = 0.3333 * Pc + (sum_nbr + rhs) / 9.0;
}
`;

// === Multigrid V-cycle shaders ===

// Compute rho_total from single density field + labels
const computeRhoWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhoTotal: array<f32>;
@group(0) @binding(3) var<storage, read> atoms: array<Atom>;
@group(0) @binding(4) var<storage, read> label: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  let u = U[id];
  rhoTotal[id] = atoms[label[id]].Z * u * u;
}
`;

// Compute residual: r = -2*pi*rho - lap(P)
const computeResidualWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pin: array<f32>;
@group(0) @binding(2) var<storage, read_write> Rout: array<f32>;
@group(0) @binding(3) var<storage, read> rhoTotal: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let Pc = Pin[id];
  let lap = (Pin[id + p.S2] + Pin[id - p.S2]
           + Pin[id + p.S]  + Pin[id - p.S]
           + Pin[id + 1u]   + Pin[id - 1u]
           - 6.0 * Pc) * p.inv_h2;
  Rout[id] = -p.TWO_PI * rhoTotal[id] - lap;
}
`;

// Full-weighting 3D restriction (27-point stencil) — single field
const restrictWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> fine: array<f32>;
@group(0) @binding(2) var<storage, read_write> coarse: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NCM = ${NC - 1}u;
  let tot = NCM * NCM * NCM;
  if (gid.x >= tot) { return; }
  let K2 = (gid.x % NCM) + 1u;
  let J = ((gid.x / NCM) % NCM) + 1u;
  let I = (gid.x / (NCM * NCM)) + 1u;
  let fi = 2u * I; let fj = 2u * J; let fk = 2u * K2;

  var val: f32 = 8.0 * fine[fi*p.S2 + fj*p.S + fk];
  val += 4.0 * (fine[(fi+1u)*p.S2 + fj*p.S + fk] + fine[(fi-1u)*p.S2 + fj*p.S + fk]
              + fine[fi*p.S2 + (fj+1u)*p.S + fk] + fine[fi*p.S2 + (fj-1u)*p.S + fk]
              + fine[fi*p.S2 + fj*p.S + (fk+1u)] + fine[fi*p.S2 + fj*p.S + (fk-1u)]);
  val += 2.0 * (fine[(fi+1u)*p.S2 + (fj+1u)*p.S + fk] + fine[(fi+1u)*p.S2 + (fj-1u)*p.S + fk]
              + fine[(fi-1u)*p.S2 + (fj+1u)*p.S + fk] + fine[(fi-1u)*p.S2 + (fj-1u)*p.S + fk]
              + fine[(fi+1u)*p.S2 + fj*p.S + (fk+1u)] + fine[(fi+1u)*p.S2 + fj*p.S + (fk-1u)]
              + fine[(fi-1u)*p.S2 + fj*p.S + (fk+1u)] + fine[(fi-1u)*p.S2 + fj*p.S + (fk-1u)]
              + fine[fi*p.S2 + (fj+1u)*p.S + (fk+1u)] + fine[fi*p.S2 + (fj+1u)*p.S + (fk-1u)]
              + fine[fi*p.S2 + (fj-1u)*p.S + (fk+1u)] + fine[fi*p.S2 + (fj-1u)*p.S + (fk-1u)]);
  val += fine[(fi+1u)*p.S2 + (fj+1u)*p.S + (fk+1u)] + fine[(fi+1u)*p.S2 + (fj+1u)*p.S + (fk-1u)]
       + fine[(fi+1u)*p.S2 + (fj-1u)*p.S + (fk+1u)] + fine[(fi+1u)*p.S2 + (fj-1u)*p.S + (fk-1u)]
       + fine[(fi-1u)*p.S2 + (fj+1u)*p.S + (fk+1u)] + fine[(fi-1u)*p.S2 + (fj+1u)*p.S + (fk-1u)]
       + fine[(fi-1u)*p.S2 + (fj-1u)*p.S + (fk+1u)] + fine[(fi-1u)*p.S2 + (fj-1u)*p.S + (fk-1u)];

  coarse[I * ${SC2}u + J * ${SC}u + K2] = val * 0.015625;
}
`;

// Weighted Jacobi on coarse grid — single field
const coarseSmoothWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Ein: array<f32>;
@group(0) @binding(2) var<storage, read_write> Eout: array<f32>;
@group(0) @binding(3) var<storage, read> coarseRhs: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NCM = ${NC - 1}u;
  let tot = NCM * NCM * NCM;
  if (gid.x >= tot) { return; }
  let K2 = (gid.x % NCM) + 1u;
  let J = ((gid.x / NCM) % NCM) + 1u;
  let I = (gid.x / (NCM * NCM)) + 1u;
  let cid = I * ${SC2}u + J * ${SC}u + K2;

  let ec = Ein[cid];
  let sum_nbr = Ein[cid + ${SC2}u] + Ein[cid - ${SC2}u]
              + Ein[cid + ${SC}u]  + Ein[cid - ${SC}u]
              + Ein[cid + 1u]      + Ein[cid - 1u];
  let hc2 = 4.0 * p.h2;
  let f = coarseRhs[cid];
  Eout[cid] = 0.3333 * ec + (sum_nbr + hc2 * f) / 9.0;
}
`;

// Trilinear prolongation + additive correction to P — single field
const prolongCorrectWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Ec: array<f32>;
@group(0) @binding(2) var<storage, read_write> Pf: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;

  let ci = i / 2u; let cj = j / 2u; let ck = k / 2u;
  let ci1 = min(ci + (i & 1u), ${NC}u);
  let cj1 = min(cj + (j & 1u), ${NC}u);
  let ck1 = min(ck + (k & 1u), ${NC}u);
  let wi = select(1.0, 0.5, (i & 1u) == 1u);
  let wj = select(1.0, 0.5, (j & 1u) == 1u);
  let wk = select(1.0, 0.5, (k & 1u) == 1u);
  let wi1 = 1.0 - wi; let wj1 = 1.0 - wj; let wk1 = 1.0 - wk;

  var corr: f32 = 0.0;
  corr += wi  * wj  * wk  * Ec[ci  * ${SC2}u + cj  * ${SC}u + ck];
  corr += wi  * wj  * wk1 * Ec[ci  * ${SC2}u + cj  * ${SC}u + ck1];
  corr += wi  * wj1 * wk  * Ec[ci  * ${SC2}u + cj1 * ${SC}u + ck];
  corr += wi  * wj1 * wk1 * Ec[ci  * ${SC2}u + cj1 * ${SC}u + ck1];
  corr += wi1 * wj  * wk  * Ec[ci1 * ${SC2}u + cj  * ${SC}u + ck];
  corr += wi1 * wj  * wk1 * Ec[ci1 * ${SC2}u + cj  * ${SC}u + ck1];
  corr += wi1 * wj1 * wk  * Ec[ci1 * ${SC2}u + cj1 * ${SC}u + ck];
  corr += wi1 * wj1 * wk1 * Ec[ci1 * ${SC2}u + cj1 * ${SC}u + ck1];

  let fid = i * p.S2 + j * p.S + k;
  Pf[fid] += 0.5 * corr;
}
`;

const reduceWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> label: array<u32>;
@group(0) @binding(3) var<storage, read> Pv: array<f32>;
@group(0) @binding(4) var<storage, read> K: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;
@group(0) @binding(6) var<storage, read> atoms: array<Atom>;

var<workgroup> sn: array<f32, ${NRED * REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;

  for (var x: u32 = 0u; x < ${NRED}u; x++) { sn[lid * ${NRED}u + x] = 0.0; }

  if (gid.x < tot) {
    let k = (gid.x % NM) + 1u;
    let j = ((gid.x / NM) % NM) + 1u;
    let i = (gid.x / (NM * NM)) + 1u;
    let id = i * p.S2 + j * p.S + k;

    let v = U[id];
    let m = label[id];
    let Zm = atoms[m].Z;

    // Per-electron norm
    sn[lid * ${NRED}u + m] = v * v * p.h3;

    // Kinetic energy
    let a = U[id + p.S2] - v;
    let b = U[id + p.S] - v;
    let c = U[id + 1u] - v;
    sn[lid * ${NRED}u + ${NELEC}u] = Zm * 0.5 * (a * a + b * b + c * c) * p.h;

    // Electron-nuclear potential
    sn[lid * ${NRED}u + ${NELEC + 1}u] = -Zm * K[id] * v * v * p.h3;

    // Electron-electron potential
    sn[lid * ${NRED}u + ${NELEC + 2}u] = Zm * Pv[id] * v * v * p.h3;
  }

  workgroupBarrier();

  for (var s: u32 = ${REDUCE_WG >> 1}u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var x: u32 = 0u; x < ${NRED}u; x++) {
        sn[lid * ${NRED}u + x] += sn[(lid + s) * ${NRED}u + x];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    let base = wgid.x * ${NRED}u;
    for (var x: u32 = 0u; x < ${NRED}u; x++) {
      partials[base + x] = sn[x];
    }
  }
}
`;

const finalizeWGSL = `
struct NWG { count: u32 }
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> sums: array<f32>;
@group(0) @binding(2) var<uniform> nwg: NWG;

var<workgroup> wg: array<f32, ${NRED * REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(local_invocation_index) lid: u32) {
  for (var x: u32 = 0u; x < ${NRED}u; x++) { wg[lid * ${NRED}u + x] = 0.0; }

  for (var i: u32 = lid; i < nwg.count; i += ${REDUCE_WG}u) {
    for (var x: u32 = 0u; x < ${NRED}u; x++) {
      wg[lid * ${NRED}u + x] += partials[i * ${NRED}u + x];
    }
  }

  workgroupBarrier();

  for (var s: u32 = ${REDUCE_WG >> 1}u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var x: u32 = 0u; x < ${NRED}u; x++) {
        wg[lid * ${NRED}u + x] += wg[(lid + s) * ${NRED}u + x];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    for (var x: u32 = 0u; x < ${NRED}u; x++) {
      sums[x] = wg[x];
    }
  }
}
`;

const normalizeWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> U: array<f32>;
@group(0) @binding(2) var<storage, read> sums: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (g.x >= tot) { return; }

  let k = (g.x % NM) + 1u;
  let j = ((g.x / NM) % NM) + 1u;
  let i = (g.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let n = sums[label[id]];
  if (n > 0.0) { U[id] *= inverseSqrt(n); }
}
`;

const extractWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> label: array<u32>;
@group(0) @binding(3) var<storage, read> Pv: array<f32>;
@group(0) @binding(4) var<storage, read> K: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  let j = g.y;
  let SS = p.NN + 1u;
  if (i > p.NN || j > p.NN) { return; }

  let idx = i * p.S2 + j * p.S + p.N2;
  let u = U[idx];
  let lbl = label[idx];
  for (var m: u32 = 0u; m < ${NELEC}u; m++) {
    out[m * SS * SS + i * SS + j] = select(0.0, u * u, lbl == m);
  }

  if (j == 0u) {
    let b = ${NELEC}u * SS * SS;
    for (var m: u32 = 0u; m < ${NELEC}u; m++) {
      let wIdx = i * p.S2 + (p.N2 + 8u) * p.S + p.N2;
      out[b + m * SS + i] = f32(label[wIdx] == m);
      let uIdx = i * p.S2 + (p.N2 + 5u) * p.S + p.N2;
      out[b + ${NELEC}u * SS + m * SS + i] = U[uIdx] * f32(label[uIdx] == m);
      out[b + ${NELEC * 2}u * SS + m * SS + i] = Pv[i * p.S2 + p.N2 * p.S + p.N2];
    }
    out[b + ${NELEC * 3}u * SS + i] = K[i * p.S2 + p.N2 * p.S + p.N2];
  }
}
`;

// ===== GPU STATE =====
let device, paramsBuf, atomBuf, K_buf, sumsBuf, sumsReadBuf, sliceBuf, sliceReadBuf, partialsBuf, numWGBuf;
let U_buf = [], P_buf = [], labelBuf, W_buf = [];
let rhoTotalBuf, residualBuf, Pc_buf = [], coarseRhsBuf;
let updatePL, jacobiSmoothPL, reducePL, finalizePL, normalizePL, extractPL;
let updateWPL;
let computeRhoPL, computeResidualPL, restrictPL, coarseSmoothPL, prolongCorrectPL;
let updateBG = [], jacobiFineBG = [], reduceBG = [], finalizeBG, normalizeBG = [], extractBG = [];
let updateWBG = [];
let computeRhoBG = [], residualBG = [], prolongCorrectBG;
let restrictBG, coarseSmoothBG = [];
let cur = 0, gpuReady = false, computing = false;
let tStep = 0, E = 0, lastMs = 0;
let E_T = 0, E_eK = 0, E_ee = 0, E_KK = 0;
let gpuError = null;

// Single phase run
let phase = 0, phaseSteps = 0;
const TOTAL_STEPS = window.USER_STEPS || 20000;
let addNucRepulsion = true;
let vcycleEnabled = true;
let vcycleCount = 0;

const SLICE_SIZE = (NELEC * S * S + (3 * NELEC + 1) * S) * 4;
const WG_UPDATE = Math.ceil(INTERIOR / 256);
const WG_REDUCE = Math.ceil(INTERIOR / REDUCE_WG);
const WG_NORM = Math.ceil(INTERIOR / 256);
const WG_EXTRACT = Math.ceil(S / 16);
const WG_COARSE = Math.ceil(INTERIOR_C / 256);
const SUMS_BYTES = NRED * 4;

let sliceData = null;

function smoothCut(r, rc) {
  if (r >= rc) return 1;
  const edge = rc - 3 * hv;
  const t = Math.max(0, Math.min(1, (r - edge) / (rc - edge)));
  return t * t * (3 - 2 * t);
}

function fillParamsBuf(pb) {
  const pu = new Uint32Array(pb);
  const pf = new Float32Array(pb);
  pu[0] = NN; pu[1] = S; pu[2] = S2; pu[3] = S3;
  pu[4] = N2; pf[5] = 1.0; pf[6] = R_out; pf[7] = 2 * Math.PI;
  pf[8] = hv; pf[9] = h2v; pf[10] = 1 / hv; pf[11] = 1 / h2v;
  pf[12] = dtv; pf[13] = half_dv; pf[14] = h3v; pf[15] = 0;
}

function fillAtomBuf() {
  const ab = new ArrayBuffer(ATOM_BUF_BYTES);
  const au = new Uint32Array(ab);
  const af = new Float32Array(ab);
  for (let n = 0; n < MAX_ATOMS; n++) {
    const off = n * ATOM_STRIDE;
    au[off] = nucPos[n][0];
    au[off + 1] = nucPos[n][1];
    au[off + 2] = nucPos[n][2];
    af[off + 3] = Z[n];
    af[off + 4] = r_cut[n];
  }
  device.queue.writeBuffer(atomBuf, 0, ab);
}

function uploadInitialData() {
  console.log("Init: nuclei at", nucPos.map((p,i) => i+"=("+p+")").join(" "));

  const Kd = new Float32Array(S3);
  const Ud = new Float32Array(S3);
  const Ld = new Uint32Array(S3);
  const Wd = new Float32Array(S3);
  const Pd = new Float32Array(S3);
  const soft = 0.04 * h2v;
  const NA = NELEC;
  const SMOOTH_WIDTH = 3.0 * hv; // smoothing width in au

  for (let i = 0; i <= NN; i++) {
    const dx = [];
    for (let n = 0; n < NA; n++) dx[n] = (i - nucPos[n][0]) * hv;
    for (let j = 0; j <= NN; j++) {
      const dy = [];
      for (let n = 0; n < NA; n++) dy[n] = (j - nucPos[n][1]) * hv;
      for (let k = 0; k <= NN; k++) {
        const dz = [];
        for (let n = 0; n < NA; n++) dz[n] = (k - nucPos[n][2]) * hv;
        const id = i * S2 + j * S + k;

        const r = [], ir = [], u = [];
        for (let n = 0; n < NA; n++) {
          r[n] = Math.sqrt(dx[n]*dx[n] + dy[n]*dy[n] + dz[n]*dz[n] + soft);
          ir[n] = 1 / r[n];
          u[n] = Math.exp(-Z[n] * r[n]);
        }

        let Kval = 0;
        for (let n = 0; n < NA; n++) Kval += Z[n] * ir[n];
        Kd[id] = Kval;

        // Assign to nearest atom (simple distance, non-overlapping spheres)
        let best = -1, bestD = Infinity, secondD = Infinity;
        for (let n = 0; n < NA; n++) {
          if (Z[n] > 0) {
            if (r[n] < bestD) { secondD = bestD; bestD = r[n]; best = n; }
            else if (r[n] < secondD) { secondD = r[n]; }
          }
        }
        if (best >= 0) {
          Ld[id] = best;
          Ud[id] = u[best]; // exp(-Z*r) sphere for assigned atom
          // Smooth W: 1 deep inside cell, 0 at boundary
          const margin = secondD - bestD;
          const t = Math.min(1, margin / SMOOTH_WIDTH);
          Wd[id] = t * t * (3 - 2 * t); // smoothstep
        }

        // Initial potential estimate
        let pAvg = 0;
        for (let n = 0; n < NA; n++) pAvg += Z[n] * ir[n];
        Pd[id] = pAvg / NA;
      }
    }
  }

  console.log("Uploading to GPU...");
  device.queue.writeBuffer(K_buf, 0, Kd);
  device.queue.writeBuffer(labelBuf, 0, Ld);
  for (let i = 0; i < 2; i++) device.queue.writeBuffer(W_buf[i], 0, Wd);
  for (let i = 0; i < 2; i++) {
    device.queue.writeBuffer(U_buf[i], 0, Ud);
    device.queue.writeBuffer(P_buf[i], 0, Pd);
  }
  fillAtomBuf();
  cur = 0;
}

function updateParamsBuf() {
  const pb = new ArrayBuffer(PARAM_BYTES);
  fillParamsBuf(pb);
  device.queue.writeBuffer(paramsBuf, 0, pb);
  fillAtomBuf();
}

function startMolPhase() {
  nucPos = molNucPos.map(p => [...p]);
  Z = [...Z_orig]; Ne = [...Ne_orig];
  R_out = 2.0;
  addNucRepulsion = true;
  updateParamsBuf();
  uploadInitialData();
  tStep = 0;
  phaseSteps = 0;
  E_min = Infinity;
  cur = 0;
  phase = 0;
  console.log("=== Molecule: " + atomLabels.join("-") + " ===");
}

function setup() {
  createCanvas(400, 400);
  textSize(9);
  initGPU();
}

async function initGPU() {
  try {
    if (!navigator.gpu) {
      gpuError = "WebGPU not supported. Use Chrome 113+ or Safari 17+.";
      return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { gpuError = "No GPU adapter found."; return; }

    try {
      const info = await adapter.requestAdapterInfo();
      console.log("GPU:", info.vendor, info.architecture, info.description);
    } catch (e) {}

    const maxBuf = S3 * 4;
    device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.max(maxBuf, adapter.limits.maxStorageBufferBindingSize),
        maxBufferSize: Math.max(maxBuf, adapter.limits.maxBufferSize)
      }
    });
    console.log("WebGPU device ready, maxStorage=" + device.limits.maxStorageBufferBindingSize);

    device.lost.then((info) => {
      gpuError = "GPU device lost: " + info.message;
      gpuReady = false;
    });

    const bs = S3 * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    K_buf = device.createBuffer({ size: bs, usage });
    atomBuf = device.createBuffer({ size: ATOM_BUF_BYTES, usage });
    for (let i = 0; i < 2; i++) {
      U_buf[i] = device.createBuffer({ size: bs, usage: usage | GPUBufferUsage.COPY_SRC });
      P_buf[i] = device.createBuffer({ size: bs, usage: usage | GPUBufferUsage.COPY_SRC });
    }
    labelBuf = device.createBuffer({ size: bs, usage: usage });
    for (let i = 0; i < 2; i++) {
      W_buf[i] = device.createBuffer({ size: bs, usage: usage });
    }

    // Multigrid buffers
    const cBufSize = SC3 * 4;
    rhoTotalBuf = device.createBuffer({ size: bs, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    residualBuf = device.createBuffer({ size: bs, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    for (let i = 0; i < 2; i++) {
      Pc_buf[i] = device.createBuffer({ size: cBufSize, usage: usage });
    }
    coarseRhsBuf = device.createBuffer({ size: cBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    console.log("Multigrid: fine=" + NN + " coarse=" + NC);

    const partialSize = WG_REDUCE * NRED * 4;
    partialsBuf = device.createBuffer({ size: partialSize, usage: GPUBufferUsage.STORAGE });
    sumsBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sumsReadBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    sliceBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sliceReadBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    numWGBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(numWGBuf, 0, new Uint32Array([WG_REDUCE, 0, 0, 0]));

    const pb = new ArrayBuffer(PARAM_BYTES);
    fillParamsBuf(pb);
    paramsBuf = device.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuf, 0, pb);

    uploadInitialData();

    async function compileShader(name, code) {
      const module = device.createShaderModule({ code });
      try {
        const getInfo = module.getCompilationInfo || module.compilationInfo;
        if (getInfo) {
          const info = await getInfo.call(module);
          for (const msg of info.messages) {
            if (msg.type === 'error') {
              throw new Error("Shader '" + name + "': " + msg.message + " (line " + msg.lineNum + ")");
            }
          }
        }
      } catch (e) {
        if (e.message.startsWith("Shader '")) throw e;
      }
      console.log("Shader '" + name + "' OK");
      return module;
    }

    const updateMod = await compileShader('updateU', updateU_WGSL);
    const updateWMod = await compileShader('updateW', updateW_WGSL);
    const jacobiSmoothMod = await compileShader('jacobiSmooth', jacobiSmoothWGSL);
    const computeRhoMod = await compileShader('computeRho', computeRhoWGSL);
    const computeResidualMod = await compileShader('computeResidual', computeResidualWGSL);
    const restrictMod = await compileShader('restrict', restrictWGSL);
    const coarseSmoothMod = await compileShader('coarseSmooth', coarseSmoothWGSL);
    const prolongCorrectMod = await compileShader('prolongCorrect', prolongCorrectWGSL);
    const reduceMod = await compileShader('reduce', reduceWGSL);
    const finalizeMod = await compileShader('finalize', finalizeWGSL);
    const normalizeMod = await compileShader('normalize', normalizeWGSL);
    const extractMod = await compileShader('extract', extractWGSL);

    updatePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: updateMod, entryPoint: 'main' } });
    updateWPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: updateWMod, entryPoint: 'main' } });
    jacobiSmoothPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: jacobiSmoothMod, entryPoint: 'main' } });
    computeRhoPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: computeRhoMod, entryPoint: 'main' } });
    computeResidualPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: computeResidualMod, entryPoint: 'main' } });
    restrictPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: restrictMod, entryPoint: 'main' } });
    coarseSmoothPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: coarseSmoothMod, entryPoint: 'main' } });
    prolongCorrectPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: prolongCorrectMod, entryPoint: 'main' } });
    reducePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: reduceMod, entryPoint: 'main' } });
    finalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: finalizeMod, entryPoint: 'main' } });
    normalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: normalizeMod, entryPoint: 'main' } });
    extractPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: extractMod, entryPoint: 'main' } });

    for (let c = 0; c < 2; c++) {
      const n = 1 - c;
      // U update: reads smooth W[0], P_buf[0]
      updateBG[c] = device.createBindGroup({ layout: updatePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: K_buf } },
        { binding: 2, resource: { buffer: U_buf[c] } },
        { binding: 3, resource: { buffer: W_buf[0] } },
        { binding: 4, resource: { buffer: P_buf[0] } },
        { binding: 5, resource: { buffer: U_buf[n] } },
      ]});
      reduceBG[c] = device.createBindGroup({ layout: reducePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: labelBuf } },
        { binding: 3, resource: { buffer: P_buf[0] } },
        { binding: 4, resource: { buffer: K_buf } },
        { binding: 5, resource: { buffer: partialsBuf } },
        { binding: 6, resource: { buffer: atomBuf } },
      ]});
      normalizeBG[c] = device.createBindGroup({ layout: normalizePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: sumsBuf } },
        { binding: 3, resource: { buffer: labelBuf } },
      ]});
      extractBG[c] = device.createBindGroup({ layout: extractPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: labelBuf } },
        { binding: 3, resource: { buffer: P_buf[0] } },
        { binding: 4, resource: { buffer: K_buf } },
        { binding: 5, resource: { buffer: sliceBuf } },
      ]});
      // Multigrid bind groups (per cur for U dependency)
      computeRhoBG[c] = device.createBindGroup({ layout: computeRhoPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: rhoTotalBuf } },
        { binding: 3, resource: { buffer: atomBuf } },
        { binding: 4, resource: { buffer: labelBuf } },
      ]});
    }
    // W update: ping-pong diffusion at boundaries
    for (let d = 0; d < 2; d++) {
      updateWBG[d] = device.createBindGroup({ layout: updateWPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: W_buf[d] } },
        { binding: 2, resource: { buffer: W_buf[1 - d] } },
        { binding: 3, resource: { buffer: labelBuf } },
      ]});
    }
    // Residual (cur-independent now — single field P)
    residualBG[0] = device.createBindGroup({ layout: computeResidualPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: P_buf[0] } },
      { binding: 2, resource: { buffer: residualBuf } },
      { binding: 3, resource: { buffer: rhoTotalBuf } },
    ]});
    // Jacobi fine-grid smoother: 2 directions (no U dependency)
    jacobiFineBG[0] = device.createBindGroup({ layout: jacobiSmoothPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: P_buf[0] } },
      { binding: 2, resource: { buffer: P_buf[1] } },
      { binding: 3, resource: { buffer: rhoTotalBuf } },
    ]});
    jacobiFineBG[1] = device.createBindGroup({ layout: jacobiSmoothPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: P_buf[1] } },
      { binding: 2, resource: { buffer: P_buf[0] } },
      { binding: 3, resource: { buffer: rhoTotalBuf } },
    ]});
    // Prolongation always corrects P_buf[0]
    prolongCorrectBG = device.createBindGroup({ layout: prolongCorrectPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: Pc_buf[0] } },
      { binding: 2, resource: { buffer: P_buf[0] } },
    ]});
    // Restriction and coarse solve (cur-independent)
    restrictBG = device.createBindGroup({ layout: restrictPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: residualBuf } },
      { binding: 2, resource: { buffer: coarseRhsBuf } },
    ]});
    for (let dir = 0; dir < 2; dir++) {
      coarseSmoothBG[dir] = device.createBindGroup({ layout: coarseSmoothPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: Pc_buf[dir] } },
        { binding: 2, resource: { buffer: Pc_buf[1 - dir] } },
        { binding: 3, resource: { buffer: coarseRhsBuf } },
      ]});
    }

    finalizeBG = device.createBindGroup({ layout: finalizePL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: partialsBuf } },
      { binding: 1, resource: { buffer: sumsBuf } },
      { binding: 2, resource: { buffer: numWGBuf } },
    ]});

    console.log("Ready! dispatch(" + WG_UPDATE + ") single-field + multigrid V-cycle");
    gpuReady = true;
    startMolPhase();

  } catch (e) {
    gpuError = e.message || String(e);
    console.error("GPU init failed:", e);
  }
}

async function doSteps(n) {
  const t0 = performance.now();
  const enc = device.createCommandEncoder();

  // --- Phase 1: U evolution for n steps ---
  for (let s = 0; s < n; s++) {
    const next = 1 - cur;
    let vp = enc.beginComputePass();
    vp.setPipeline(computeRhoPL);
    vp.setBindGroup(0, computeRhoBG[cur]);
    vp.dispatchWorkgroups(WG_UPDATE);
    vp.end();

    for (let js = 0; js < 2; js++) {
      vp = enc.beginComputePass();
      vp.setPipeline(jacobiSmoothPL);
      vp.setBindGroup(0, jacobiFineBG[js]);
      vp.dispatchWorkgroups(WG_UPDATE);
      vp.end();
    }

    if (vcycleEnabled && s > 0 && s % POISSON_INTERVAL === 0) {
      vcycleCount++;
      vp = enc.beginComputePass();
      vp.setPipeline(computeResidualPL);
      vp.setBindGroup(0, residualBG[0]);
      vp.dispatchWorkgroups(WG_UPDATE);
      vp.end();
      vp = enc.beginComputePass();
      vp.setPipeline(restrictPL);
      vp.setBindGroup(0, restrictBG);
      vp.dispatchWorkgroups(WG_COARSE);
      vp.end();
      enc.clearBuffer(Pc_buf[0]);
      for (let cs = 0; cs < 10; cs++) {
        vp = enc.beginComputePass();
        vp.setPipeline(coarseSmoothPL);
        vp.setBindGroup(0, coarseSmoothBG[cs % 2]);
        vp.dispatchWorkgroups(WG_COARSE);
        vp.end();
      }
      vp = enc.beginComputePass();
      vp.setPipeline(prolongCorrectPL);
      vp.setBindGroup(0, prolongCorrectBG);
      vp.dispatchWorkgroups(WG_UPDATE);
      vp.end();
    }

    let cp = enc.beginComputePass();
    cp.setPipeline(updatePL);
    cp.setBindGroup(0, updateBG[cur]);
    cp.dispatchWorkgroups(WG_UPDATE);
    cp.end();

    if ((s + 1) % NORM_INTERVAL === 0 || s === n - 1) {
      cp = enc.beginComputePass();
      cp.setPipeline(reducePL);
      cp.setBindGroup(0, reduceBG[next]);
      cp.dispatchWorkgroups(WG_REDUCE);
      cp.end();

      cp = enc.beginComputePass();
      cp.setPipeline(finalizePL);
      cp.setBindGroup(0, finalizeBG);
      cp.dispatchWorkgroups(1);
      cp.end();

      cp = enc.beginComputePass();
      cp.setPipeline(normalizePL);
      cp.setBindGroup(0, normalizeBG[next]);
      cp.dispatchWorkgroups(WG_NORM);
      cp.end();
    }

    cur = next;
  }

  // --- W smoothing at boundaries (even count → lands back in W_buf[0]) ---
  const wSteps = W_STEPS_PER_FRAME - (W_STEPS_PER_FRAME % 2);
  for (let s = 0; s < wSteps; s++) {
    let wp = enc.beginComputePass();
    wp.setPipeline(updateWPL);
    wp.setBindGroup(0, updateWBG[s % 2]);
    wp.dispatchWorkgroups(WG_UPDATE);
    wp.end();
  }

  let cp = enc.beginComputePass();
  cp.setPipeline(extractPL);
  cp.setBindGroup(0, extractBG[cur]);
  cp.dispatchWorkgroups(WG_EXTRACT, WG_EXTRACT);
  cp.end();

  enc.copyBufferToBuffer(sumsBuf, 0, sumsReadBuf, 0, SUMS_BYTES);
  enc.copyBufferToBuffer(sliceBuf, 0, sliceReadBuf, 0, SLICE_SIZE);
  device.queue.submit([enc.finish()]);

  await sumsReadBuf.mapAsync(GPUMapMode.READ);
  const sumsData = new Float32Array(sumsReadBuf.getMappedRange().slice(0));
  sumsReadBuf.unmap();
  E_T = sumsData[NELEC];
  E_eK = sumsData[NELEC + 1];
  E_ee = sumsData[NELEC + 2];

  await sliceReadBuf.mapAsync(GPUMapMode.READ);
  sliceData = new Float32Array(sliceReadBuf.getMappedRange().slice(0));
  sliceReadBuf.unmap();

  tStep += n;
  lastMs = performance.now() - t0;

  E_KK = 0;
  if (addNucRepulsion) {
    const soft_nuc = 0.04 * h2v;
    for (let a = 0; a < NELEC; a++) {
      for (let b = a + 1; b < NELEC; b++) {
        if (Z[a] === 0 || Z[b] === 0) continue;
        const d = Math.sqrt(
          ((nucPos[a][0]-nucPos[b][0])*hv)**2 +
          ((nucPos[a][1]-nucPos[b][1])*hv)**2 +
          ((nucPos[a][2]-nucPos[b][2])*hv)**2 + soft_nuc);
        E_KK += Z[a]*Z[b]/d;
      }
    }
  }
  E = E_T + E_eK + E_ee + E_KK;

  if (!isFinite(E)) {
    gpuError = "Numerical instability at step " + tStep;
    return;
  }

  console.log("Step " + tStep + ": E=" + E.toFixed(6) + " (" + lastMs.toFixed(0) + "ms/" + n + "steps)");
}

function draw() {
  background(0);
  noStroke();

  if (gpuError) {
    fill(200, 0, 0);
    textSize(11);
    text("GPU Error:", 10, 180);
    textSize(9);
    const lines = gpuError.match(/.{1,55}/g) || [gpuError];
    for (let i = 0; i < Math.min(lines.length, 8); i++) {
      text(lines[i], 10, 198 + i * 14);
    }
    return;
  }

  if (!gpuReady) {
    fill(255);
    text("Initializing WebGPU...", 10, 200);
    return;
  }

  if (!computing && phase === 0) {
    computing = true;
    doSteps(STEPS_PER_FRAME).then(() => {
      computing = false;
      phaseSteps += STEPS_PER_FRAME;
      if (isFinite(E) && E < E_min) E_min = E;

      if (phaseSteps >= TOTAL_STEPS && TOTAL_STEPS > 0) {
        console.log("=== DONE: E=" + E.toFixed(6) + " ===");
        phase = 1;  // done
      }
    }).catch((e) => {
      gpuError = e.message || String(e);
      console.error("GPU step failed:", e);
      computing = false;
    });
  }

  if (sliceData) {
    const SS = S;
    loadPixels();
    const d = pixelDensity();
    const W = 400 * d, H = 400 * d;
    for (let p = 0; p < W * H * 4; p += 4) {
      pixels[p] = 0; pixels[p+1] = 0; pixels[p+2] = 0; pixels[p+3] = 255;
    }
    // Element colors: H=white, O=red, N=blue, C=green
    const zRGB = {1:[1,1,0], 2:[1,0,0], 3:[0,0.5,1], 4:[0,1,1]};
    const eRGB = Z.slice(0, NELEC).map(z => zRGB[z] || [0.5,0.5,0.5]);
    // Per-atom auto-scale: find max density per electron
    const maxPerAtom = new Float32Array(NELEC);
    for (let i = 1; i < NN; i++) {
      for (let j = 1; j < NN; j++) {
        const b = i * SS + j;
        for (let m = 0; m < NELEC; m++) {
          const v = sliceData[m * SS * SS + b];
          if (v > maxPerAtom[m]) maxPerAtom[m] = v;
        }
      }
    }

    for (let i = 1; i < NN; i++) {
      const px0 = Math.floor(PX * i * d);
      const px1 = Math.floor(PX * (i + 1) * d);
      for (let j = 1; j < NN; j++) {
        const py0 = Math.floor(PX * j * d);
        const py1 = Math.floor(PX * (j + 1) * d);
        const b = i * SS + j;
        let ri = 0, gi = 0, bi = 0;
        for (let m = 0; m < NELEC; m++) {
          const s = maxPerAtom[m] > 0 ? 1.0 / maxPerAtom[m] : 1.0;
          const ev = 255 * Math.sqrt(sliceData[m * SS * SS + b] * s);
          ri += ev * eRGB[m][0];
          gi += ev * eRGB[m][1];
          bi += ev * eRGB[m][2];
        }
        ri = Math.min(255, Math.floor(ri));
        gi = Math.min(255, Math.floor(gi));
        bi = Math.min(255, Math.floor(bi));
        for (let py = py0; py < py1 && py < H; py++) {
          for (let px = px0; px < px1 && px < W; px++) {
            const idx = (py * W + px) * 4;
            pixels[idx] = ri;
            pixels[idx + 1] = gi;
            pixels[idx + 2] = bi;
            pixels[idx + 3] = 255;
          }
        }
      }
    }
    updatePixels();

    // Line plots
    const lb = NELEC * SS * SS;
    const zCol = {1:[255,255,255], 2:[255,0,0], 3:[0,100,255], 4:[0,255,0]};
    const colors = Z.slice(0, NELEC).map(z => zCol[z] || [128,128,128]);
    for (let i = 1; i < NN - 10; i++) {
      for (let m = 0; m < NELEC; m++) {
        if (Z[m] === 0) continue;
        fill(255); ellipse(PX * i, 300 - 100 * sliceData[lb + m * SS + i], 2);
        fill(colors[m][0], colors[m][1], colors[m][2]);
        ellipse(PX * i, 300 - 100 * sliceData[lb + NELEC * SS + m * SS + i], 3);
        fill(0, 255, 255, 200); ellipse(PX * i, 300 - 30 * sliceData[lb + 2 * NELEC * SS + m * SS + i], 2);
      }
      fill(0, 0, 255, 200); ellipse(PX * i, 300 - 30 * sliceData[lb + 3 * NELEC * SS + i], 2);
    }
  }

  // Draw nuclear positions
  fill(255); stroke(255); strokeWeight(1);
  for (let n = 0; n < NELEC; n++) {
    if (Z[n] > 0) circle(nucPos[n][0] * PX, nucPos[n][1] * PX, 6);
  }
  // Screen boundary
  noFill(); stroke(100); strokeWeight(1);
  rect(0, 0, 400, 400);
  noStroke();

  fill(255);
  const labels = atomLabels.map((el, i) => [el, Z_orig[i]]).filter(x => x[1] > 0).map(x => x[0] + "(Z=" + x[1] + ")").join(" ");
  const pLabel = phase === 0 ? "running" : "DONE";
  text("Molecule: " + labels + " | " + screenAu + " au | " + pLabel + " | " + NN + "^3", 5, 20);
  text("step " + tStep + " (" + phaseSteps + "/" + TOTAL_STEPS + ")  E=" + E.toFixed(6) + "  E_min=" + E_min.toFixed(6), 5, 35);
  if (lastMs > 0) text((lastMs / STEPS_PER_FRAME).toFixed(1) + "ms/step", 300, 35);

  fill(200);
  text("T=" + E_T.toFixed(4) + " V_eK=" + E_eK.toFixed(4) + " V_ee=" + E_ee.toFixed(4) + " V_KK=" + E_KK.toFixed(4), 5, 50);
  fill(vcycleEnabled ? [0,255,0] : [255,100,0]);
  text("V-cycle: " + (vcycleEnabled ? "ON" : "OFF") + " (" + vcycleCount + " cycles)  [press V to toggle]", 5, 65);
}

function keyPressed() {
  if (key === 'v' || key === 'V') {
    vcycleEnabled = !vcycleEnabled;
    vcycleCount = 0;
    console.log("V-cycle " + (vcycleEnabled ? "ENABLED" : "DISABLED"));
  }
}
