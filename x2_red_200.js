// Li2 Quantum Simulation — WebGPU Compute Shaders (x2_red_200)
// 2 Li atoms (Z=3 each), 6 electrons: 2 inner + 1 outer (2s) per atom
// Fire-and-forget batching

const NN = 200;
const S = NN + 1;
const S2 = S * S;
const S3 = S * S * S;
const N2 = Math.round(NN / 2);

// Geometry: Li1 fixed, Li2 moves right
const Li1_pos = 50;
let Li2_pos, D_grid, d_au, nucRepulsion;
const nucPos = { Li1: [Li1_pos, N2, N2], Li2: [0, N2, N2] };
const R1 = 10;
const R_inner = 3.0;  // inner shell radius (au)
const hv = 20 / NN, h2v = hv * hv, h3v = hv * hv * hv;
const dv = 0.12;
const dtv = dv * h2v, half_dv = 0.5 * dv;
const PX = 400 / NN;
const INTERIOR = (NN - 1) * (NN - 1) * (NN - 1);
const STEPS_PER_BATCH = 50;
const VIS_INTERVAL = 5;
const NORM_INTERVAL = 20;
let frameCount = 0;
const NELEC = 6;
const NRED = NELEC + 1;  // 7: 6 norms + 1 energy

// Sweep configuration
const sweepDistances = [4.0, 5.0, 6.0, 8.0];  // au
const STEPS_PER_POINT = 20000;
let sweepIdx = 0;
let sweepResults = [];
let E_min = Infinity;

// Reference energies
const E_Li_ref = -7.3249;

function setDistance(dist_au) {
  d_au = dist_au;
  D_grid = Math.round(dist_au / hv);
  Li2_pos = Li1_pos + D_grid;
  nucPos.Li2[0] = Li2_pos;
  nucRepulsion = 9.0 / dist_au;  // Z1*Z2/d = 3*3/d
  console.log("Distance=" + dist_au + " au, D=" + D_grid + " grid, Li1=" + Li1_pos + " Li2=" + Li2_pos + " nuc=" + nucRepulsion.toFixed(4));
}

// ===== WGSL SHADERS =====

const paramStructWGSL = `
struct P {
  NN: u32, S: u32, S2: u32, S3: u32,
  N2: u32, li1I: u32,
  h: f32, h2: f32, inv_h: f32, inv_h2: f32,
  dt: f32, half_d: f32, _pad1: f32, TWO_PI: f32, h3: f32,
  li2I: u32
}`;

const updateWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Ui: array<f32>;
@group(0) @binding(3) var<storage, read> Wi: array<f32>;
@group(0) @binding(4) var<storage, read> Pi: array<f32>;
@group(0) @binding(5) var<storage, read_write> Uo: array<f32>;
@group(0) @binding(6) var<storage, read_write> Wo: array<f32>;
@group(0) @binding(7) var<storage, read_write> Po: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }

  let m = gid.y;
  let o = m * p.S3;

  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  // c[m]: other electrons' density
  var cm: f32 = 0.0;
  for (var e: u32 = 0u; e < 6u; e++) {
    cm -= Ui[e * p.S3 + id];
  }
  cm += Ui[o + id];
  cm = 0.5 * (cm + Ui[o + id]);

  let wc  = Wi[o + id];
  let wip = Wi[o + id + p.S2]; let wim = Wi[o + id - p.S2];
  let wjp = Wi[o + id + p.S];  let wjm = Wi[o + id - p.S];
  let wkp = Wi[o + id + 1u];   let wkm = Wi[o + id - 1u];

  let lw = (wip + wim + wjp + wjm + wkp + wkm - 6.0 * wc) * p.inv_h2;
  let gx = (wip - wim) * p.inv_h;
  let gy = (wjp - wjm) * p.inv_h;
  let gz = (wkp - wkm) * p.inv_h;
  let dw = 0.5 * p.dt * abs(cm) * lw + 5.0 * p.dt * cm * sqrt(gx * gx + gy * gy + gz * gz);

  // W update: inner electrons (m=0,1 near Li1; m=3,4 near Li2) are stable near their nucleus
  var nw = wc;
  let di1 = abs(i32(i) - i32(p.li1I));
  let di2 = abs(i32(i) - i32(p.li2I));
  if (m < 2u) {
    // Li1 inner: only update W away from Li1
    if (di1 > 3) { nw = clamp(wc + dw, 0.0, 1.0); }
  } else if (m >= 3u && m < 5u) {
    // Li2 inner: only update W away from Li2
    if (di2 > 3) { nw = clamp(wc + dw, 0.0, 1.0); }
  } else {
    // Outer electrons (m=2, m=5): update W everywhere
    nw = clamp(wc + dw, 0.0, 1.0);
  }
  Wo[o + id] = nw;

  let uc  = Ui[o + id];
  let uip = Ui[o + id + p.S2]; let uim = Ui[o + id - p.S2];
  let ujp = Ui[o + id + p.S];  let ujm = Ui[o + id - p.S];
  let ukp = Ui[o + id + 1u];   let ukm = Ui[o + id - 1u];

  Uo[o + id] = uc
    + p.half_d * ((uip - uc) * (wip + nw) * 0.5 - (uc - uim) * (nw + wim) * 0.5)
    + p.half_d * ((ujp - uc) * (wjp + nw) * 0.5 - (uc - ujm) * (nw + wjm) * 0.5)
    + p.half_d * ((ukp - uc) * (wkp + nw) * 0.5 - (uc - ukm) * (nw + wkm) * 0.5)
    + p.dt * (K[id] - 2.0 * Pi[o + id]) * uc * wc;

  let Pc = Pi[o + id];
  var rho: f32 = 0.0;
  for (var e: u32 = 0u; e < 6u; e++) {
    let ue = Ui[e * p.S3 + id];
    rho += ue * ue;
  }
  let self_u = Ui[o + id];
  rho -= self_u * self_u;

  Po[o + id] = Pc
    + p.dt * (Pi[o + id + p.S2] + Pi[o + id - p.S2]
            + Pi[o + id + p.S]  + Pi[o + id - p.S]
            + Pi[o + id + 1u]   + Pi[o + id - 1u]
            - 6.0 * Pc) * p.inv_h2
    + p.TWO_PI * p.dt * rho;
}
`;

const reduceWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> Pv: array<f32>;
@group(0) @binding(4) var<storage, read> K: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;

var<workgroup> sn: array<f32, 1792>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;

  for (var x: u32 = 0u; x < 7u; x++) { sn[lid * 7u + x] = 0.0; }

  if (gid.x < tot) {
    let k = (gid.x % NM) + 1u;
    let j = ((gid.x / NM) % NM) + 1u;
    let i = (gid.x / (NM * NM)) + 1u;
    let id = i * p.S2 + j * p.S + k;

    var en: f32 = 0.0;
    for (var m: u32 = 0u; m < 6u; m++) {
      let o = m * p.S3;
      let v = U[o + id];
      sn[lid * 7u + m] = v * v * p.h3;
      if (W[o + id] > 0.7) {
        let a = U[o + id + p.S2] - v;
        let b = U[o + id + p.S] - v;
        let c = U[o + id + 1u] - v;
        en += 0.5 * (a * a + b * b + c * c) * p.h;
      }
      en += (Pv[o + id] - K[id]) * v * v * p.h3;
    }
    sn[lid * 7u + 6u] = en;
  }

  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var x: u32 = 0u; x < 7u; x++) {
        sn[lid * 7u + x] += sn[(lid + s) * 7u + x];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    let base = wgid.x * 7u;
    for (var x: u32 = 0u; x < 7u; x++) {
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

var<workgroup> wg: array<f32, 1792>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
  for (var x: u32 = 0u; x < 7u; x++) { wg[lid * 7u + x] = 0.0; }

  for (var i: u32 = lid; i < nwg.count; i += 256u) {
    for (var x: u32 = 0u; x < 7u; x++) {
      wg[lid * 7u + x] += partials[i * 7u + x];
    }
  }

  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var x: u32 = 0u; x < 7u; x++) {
        wg[lid * 7u + x] += wg[(lid + s) * 7u + x];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    for (var x: u32 = 0u; x < 7u; x++) {
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (g.x >= tot) { return; }

  let k = (g.x % NM) + 1u;
  let j = ((g.x / NM) % NM) + 1u;
  let i = (g.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  for (var m: u32 = 0u; m < 6u; m++) {
    let n = sums[m];
    if (n > 0.0) { U[m * p.S3 + id] *= inverseSqrt(n); }
  }
}
`;

const extractWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> Pv: array<f32>;
@group(0) @binding(4) var<storage, read> K: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  let j = g.y;
  let SS = p.NN + 1u;
  if (i > p.NN || j > p.NN) { return; }

  // Extract 6 density slices (U*W at k=N2)
  for (var m: u32 = 0u; m < 6u; m++) {
    out[m * SS * SS + i * SS + j] = U[m * p.S3 + i * p.S2 + j * p.S + p.N2] * W[m * p.S3 + i * p.S2 + j * p.S + p.N2];
  }

  if (j == 0u) {
    let b = 6u * SS * SS;
    // K along bond axis
    out[b + i] = K[i * p.S2 + p.N2 * p.S + p.N2];
  }
}
`;

// ===== GPU STATE =====
let device, paramsBuf, K_buf, sumsBuf, sumsReadBuf, sliceBuf, sliceReadBuf, partialsBuf, numWGBuf;
let U_buf = [], W_buf = [], P_buf = [];
let updatePL, reducePL, finalizePL, normalizePL, extractPL;
let updateBG = [], reduceBG = [], finalizeBG, normalizeBG = [], extractBG = [];
let cur = 0, gpuReady = false, computing = false;
let tStep = 0, E = 0, lastMs = 0;
let gpuError = null;
let sweepDone = false;

const SLICE_SIZE = (6 * S * S + S) * 4;  // 6 density slices + 1 K line
const WG_UPDATE = Math.ceil(INTERIOR / 256);
const WG_REDUCE = Math.ceil(INTERIOR / 256);
const WG_NORM = Math.ceil(INTERIOR / 256);
const WG_EXTRACT = Math.ceil(S / 16);
const SUMS_BYTES = NRED * 4;  // 28 bytes

let sliceData = null;

function uploadInitialData() {
  const [li1I, li1J, li1K] = nucPos.Li1;
  const [li2I, li2J, li2K] = nucPos.Li2;
  console.log("Init: Li1=(" + li1I + "," + li1J + "," + li1K + ") Li2=(" + li2I + "," + li2J + "," + li2K + ")");

  const Kd = new Float32Array(S3);
  const Ud = new Float32Array(NELEC * S3);
  const Wd = new Float32Array(NELEC * S3);
  const Pd = new Float32Array(NELEC * S3);
  const soft = h2v;

  const mid = Math.round((li1I + li2I) / 2);

  for (let i = 0; i <= NN; i++) {
    const dxL1 = (i - li1I) * hv;
    const dxL2 = (i - li2I) * hv;
    for (let j = 0; j <= NN; j++) {
      const dyL1 = (j - li1J) * hv;
      const dyL2 = (j - li2J) * hv;
      for (let k = 0; k <= NN; k++) {
        const dzL1 = (k - li1K) * hv;
        const dzL2 = (k - li2K) * hv;
        const id = i * S2 + j * S + k;

        const rL1 = Math.sqrt(dxL1*dxL1 + dyL1*dyL1 + dzL1*dzL1 + soft);
        const rL2 = Math.sqrt(dxL2*dxL2 + dyL2*dyL2 + dzL2*dzL2 + soft);
        const irL1 = 1/rL1, irL2 = 1/rL2;

        // Nuclear potential: Z=3 for both
        Kd[id] = 3*irL1 + 3*irL2;

        // Li1 electrons (m=0,1,2)
        // m=0: Li1 inner, i < li1I
        if (rL1 < R_inner && i < li1I) {
          Ud[0*S3+id] = Math.exp(-3*rL1); Wd[0*S3+id] = 1;
        }
        // m=1: Li1 inner, i > li1I and i < mid
        if (rL1 < R_inner && i > li1I && i < mid) {
          Ud[1*S3+id] = Math.exp(-3*rL1); Wd[1*S3+id] = 1;
        }
        // m=2: Li1 outer (2s), i < mid
        if (rL1 > R_inner && i < mid) {
          Ud[2*S3+id] = Math.exp(-rL1); Wd[2*S3+id] = 1;
        }

        // Li2 electrons (m=3,4,5)
        // m=3: Li2 inner, i > li2I
        if (rL2 < R_inner && i > li2I) {
          Ud[3*S3+id] = Math.exp(-3*rL2); Wd[3*S3+id] = 1;
        }
        // m=4: Li2 inner, i < li2I and i >= mid
        if (rL2 < R_inner && i < li2I && i >= mid) {
          Ud[4*S3+id] = Math.exp(-3*rL2); Wd[4*S3+id] = 1;
        }
        // m=5: Li2 outer (2s), i >= mid
        if (rL2 > R_inner && i >= mid) {
          Ud[5*S3+id] = Math.exp(-rL2); Wd[5*S3+id] = 1;
        }

        // Initial Poisson
        for (let m = 0; m < NELEC; m++) {
          Pd[m*S3+id] = 1.5*irL1 + 1.5*irL2;
        }
      }
    }
  }

  console.log("Uploading to GPU...");
  device.queue.writeBuffer(K_buf, 0, Kd);
  for (let i = 0; i < 2; i++) {
    device.queue.writeBuffer(U_buf[i], 0, Ud);
    device.queue.writeBuffer(W_buf[i], 0, Wd);
    device.queue.writeBuffer(P_buf[i], 0, Pd);
  }
  cur = 0;
}

function uploadParams() {
  const pb = new ArrayBuffer(64);
  const pu = new Uint32Array(pb);
  const pf = new Float32Array(pb);
  pu[0] = NN; pu[1] = S; pu[2] = S2; pu[3] = S3; pu[4] = N2; pu[5] = Li1_pos;
  pf[6] = hv; pf[7] = h2v; pf[8] = 1 / hv; pf[9] = 1 / h2v;
  pf[10] = dtv; pf[11] = half_dv; pf[12] = 0; pf[13] = 2 * Math.PI;
  pf[14] = h3v;
  pu[15] = Li2_pos;
  device.queue.writeBuffer(paramsBuf, 0, pb);
}

function startSweepPoint(idx) {
  sweepIdx = idx;
  setDistance(sweepDistances[idx]);
  uploadParams();
  uploadInitialData();
  tStep = 0;
  E = 0;
  E_min = Infinity;
  console.log("=== Sweep point " + (idx + 1) + "/" + sweepDistances.length + ": d=" + d_au + " au ===");
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
      console.error(gpuError);
      return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      gpuError = "No GPU adapter found.";
      console.error(gpuError);
      return;
    }

    try {
      const info = await adapter.requestAdapterInfo();
      console.log("GPU:", info.vendor, info.architecture, info.description);
    } catch (e) { console.log("Could not get adapter info"); }

    // 6 electrons at 200^3 needs >128MB per binding — request higher limits
    device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: 256 * 1024 * 1024,
        maxBufferSize: 256 * 1024 * 1024,
      }
    });
    console.log("WebGPU device ready (256MB buffer limit)");

    device.lost.then((info) => {
      gpuError = "GPU device lost: " + info.message;
      console.error(gpuError);
      gpuReady = false;
    });

    const bs = S3 * 4, bN = NELEC * S3 * 4;
    console.log("Buffer sizes: K=" + (bs/1024/1024).toFixed(1) + "MB, U/W/P=" + (bN/1024/1024).toFixed(1) + "MB each");
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    K_buf = device.createBuffer({ size: bs, usage });
    for (let i = 0; i < 2; i++) {
      U_buf[i] = device.createBuffer({ size: bN, usage: usage | GPUBufferUsage.COPY_SRC });
      W_buf[i] = device.createBuffer({ size: bN, usage: usage | GPUBufferUsage.COPY_SRC });
      P_buf[i] = device.createBuffer({ size: bN, usage: usage | GPUBufferUsage.COPY_SRC });
    }

    const partialSize = WG_REDUCE * NRED * 4;
    partialsBuf = device.createBuffer({ size: partialSize, usage: GPUBufferUsage.STORAGE });
    sumsBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sumsReadBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    sliceBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sliceReadBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    numWGBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(numWGBuf, 0, new Uint32Array([WG_REDUCE, 0, 0, 0]));

    paramsBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    setDistance(sweepDistances[0]);
    uploadParams();
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

    const updateMod = await compileShader('update', updateWGSL);
    const reduceMod = await compileShader('reduce', reduceWGSL);
    const finalizeMod = await compileShader('finalize', finalizeWGSL);
    const normalizeMod = await compileShader('normalize', normalizeWGSL);
    const extractMod = await compileShader('extract', extractWGSL);

    updatePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: updateMod, entryPoint: 'main' } });
    reducePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: reduceMod, entryPoint: 'main' } });
    finalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: finalizeMod, entryPoint: 'main' } });
    normalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: normalizeMod, entryPoint: 'main' } });
    extractPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: extractMod, entryPoint: 'main' } });

    for (let c = 0; c < 2; c++) {
      const n = 1 - c;
      updateBG[c] = device.createBindGroup({ layout: updatePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: K_buf } },
        { binding: 2, resource: { buffer: U_buf[c] } },
        { binding: 3, resource: { buffer: W_buf[c] } },
        { binding: 4, resource: { buffer: P_buf[c] } },
        { binding: 5, resource: { buffer: U_buf[n] } },
        { binding: 6, resource: { buffer: W_buf[n] } },
        { binding: 7, resource: { buffer: P_buf[n] } },
      ]});
      reduceBG[c] = device.createBindGroup({ layout: reducePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: W_buf[c] } },
        { binding: 3, resource: { buffer: P_buf[c] } },
        { binding: 4, resource: { buffer: K_buf } },
        { binding: 5, resource: { buffer: partialsBuf } },
      ]});
      normalizeBG[c] = device.createBindGroup({ layout: normalizePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: sumsBuf } },
      ]});
      extractBG[c] = device.createBindGroup({ layout: extractPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: W_buf[c] } },
        { binding: 3, resource: { buffer: P_buf[c] } },
        { binding: 4, resource: { buffer: K_buf } },
        { binding: 5, resource: { buffer: sliceBuf } },
      ]});
    }

    finalizeBG = device.createBindGroup({ layout: finalizePL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: partialsBuf } },
      { binding: 1, resource: { buffer: sumsBuf } },
      { binding: 2, resource: { buffer: numWGBuf } },
    ]});

    console.log("Ready! dispatch(" + WG_UPDATE + ",6,1)");
    gpuReady = true;

  } catch (e) {
    gpuError = e.message || String(e);
    console.error("GPU init failed:", e);
  }
}

function encodeBatch(enc, n) {
  for (let s = 0; s < n; s++) {
    const next = 1 - cur;

    let cp = enc.beginComputePass();
    cp.setPipeline(updatePL);
    cp.setBindGroup(0, updateBG[cur]);
    cp.dispatchWorkgroups(WG_UPDATE, NELEC, 1);
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
}

async function doSteps(readback) {
  const t0 = performance.now();
  const enc = device.createCommandEncoder();
  encodeBatch(enc, STEPS_PER_BATCH);

  if (readback) {
    let cp = enc.beginComputePass();
    cp.setPipeline(extractPL);
    cp.setBindGroup(0, extractBG[cur]);
    cp.dispatchWorkgroups(WG_EXTRACT, WG_EXTRACT);
    cp.end();
    enc.copyBufferToBuffer(sumsBuf, 0, sumsReadBuf, 0, SUMS_BYTES);
    enc.copyBufferToBuffer(sliceBuf, 0, sliceReadBuf, 0, SLICE_SIZE);
  }

  device.queue.submit([enc.finish()]);
  tStep += STEPS_PER_BATCH;

  if (!readback) {
    lastMs = performance.now() - t0;
    return;
  }

  await sumsReadBuf.mapAsync(GPUMapMode.READ);
  const sumsData = new Float32Array(sumsReadBuf.getMappedRange().slice(0));
  sumsReadBuf.unmap();
  E = sumsData[6] + nucRepulsion;  // energy at index 6 (after 6 norms)

  await sliceReadBuf.mapAsync(GPUMapMode.READ);
  sliceData = new Float32Array(sliceReadBuf.getMappedRange().slice(0));
  sliceReadBuf.unmap();

  lastMs = performance.now() - t0;

  if (!isFinite(E)) {
    gpuError = "Numerical instability at step " + tStep;
    return;
  }

  if (isFinite(E) && E < E_min) E_min = E;

  console.log("Step " + tStep + ": E=" + E.toFixed(6) + " min=" + E_min.toFixed(6) + " (" + lastMs.toFixed(0) + "ms)");
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

  if (!computing && !sweepDone) {
    computing = true;
    frameCount++;
    const readback = (frameCount % VIS_INTERVAL === 0);
    doSteps(readback).then(() => {
      computing = false;
      if (tStep >= STEPS_PER_POINT) {
        const bind = E_min - 2 * E_Li_ref;
        sweepResults.push({ d: d_au, E_min: E_min, bind: bind });
        console.log("Sweep " + (sweepIdx+1) + ": d=" + d_au.toFixed(2) + " E_min=" + E_min.toFixed(6) + " bind=" + bind.toFixed(4));
        if (sweepIdx + 1 < sweepDistances.length) {
          startSweepPoint(sweepIdx + 1);
        } else {
          sweepDone = true;
          console.log("=== Sweep complete ===");
          for (const r of sweepResults) console.log("d=" + r.d.toFixed(2) + " E=" + r.E_min.toFixed(6) + " bind=" + r.bind.toFixed(4));
        }
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
    for (let i = 1; i < NN; i++) {
      const px0 = Math.floor(PX * i * d);
      const px1 = Math.floor(PX * (i + 1) * d);
      for (let j = 1; j < NN; j++) {
        const py0 = Math.floor(PX * j * d);
        const py1 = Math.floor(PX * (j + 1) * d);
        const b = i * SS + j;
        // Li1: inner=red (m=0,1), outer=yellow (m=2)
        // Li2: inner=blue (m=3,4), outer=cyan (m=5)
        const li1_in = 500 * (sliceData[0*SS*SS+b] + sliceData[1*SS*SS+b]);
        const li1_out = 2000 * sliceData[2*SS*SS+b];
        const li2_in = 500 * (sliceData[3*SS*SS+b] + sliceData[4*SS*SS+b]);
        const li2_out = 2000 * sliceData[5*SS*SS+b];
        const ri = Math.min(255, Math.floor(li1_in + li1_out));
        const gi = Math.min(255, Math.floor(li1_out + li2_out));
        const bi = Math.min(255, Math.floor(li2_in + li2_out));
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

    // Ridge plot
    const nLines = 10;
    const jStep = 4;
    const baseY = 380;
    const lineSpacing = 18;
    const scale = 3000;
    for (let L = 0; L < nLines; L++) {
      const jOff = N2 - L * jStep;
      if (jOff < 1) break;
      const yBase = baseY - L * lineSpacing;
      const alpha = Math.max(60, 255 - L * 20);
      noStroke(); fill(0);
      beginShape();
      vertex(PX, yBase);
      for (let i = 1; i < NN; i++) {
        let tot = 0;
        for (let m = 0; m < 6; m++) {
          const v = sliceData[m * SS * SS + i * SS + jOff];
          tot += v * v;
        }
        vertex(PX * i, yBase - scale * tot);
      }
      vertex(PX * (NN - 1), yBase);
      endShape(CLOSE);
      stroke(100, 180, 255, alpha); strokeWeight(1); noFill();
      beginShape();
      for (let i = 1; i < NN; i++) {
        let tot = 0;
        for (let m = 0; m < 6; m++) {
          const v = sliceData[m * SS * SS + i * SS + jOff];
          tot += v * v;
        }
        vertex(PX * i, yBase - scale * tot);
      }
      endShape();
    }
    noStroke();
  }

  // Draw nuclear positions
  fill(255, 200, 200); stroke(255); strokeWeight(1);
  circle(nucPos.Li1[0] * PX, nucPos.Li1[1] * PX, 8);
  circle(nucPos.Li2[0] * PX, nucPos.Li2[1] * PX, 8);
  noStroke();

  fill(255);
  text("Li2 6e sweep " + (sweepIdx+1) + "/" + sweepDistances.length + " | d=" + d_au.toFixed(2) + " au, " + NN + "^3", 5, 20);
  text("step " + tStep + "/" + STEPS_PER_POINT + "  E=" + E.toFixed(6) + "  min=" + (isFinite(E_min) ? E_min.toFixed(6) : "---"), 5, 35);
  if (lastMs > 0) text((lastMs / STEPS_PER_BATCH).toFixed(1) + "ms/step", 300, 35);

  // Show completed sweep results
  if (sweepResults.length > 0) {
    fill(200, 180, 100); textSize(10);
    text("d(au)   E_min      bind(Ha)", 10, 55);
    textSize(9);
    for (let r = 0; r < sweepResults.length; r++) {
      const sr = sweepResults[r];
      fill(sr.bind < 0 ? [100, 255, 100] : [255, 100, 100]);
      text(sr.d.toFixed(2) + "    " + sr.E_min.toFixed(6) + "   " + sr.bind.toFixed(4), 10, 70 + r * 14);
    }
  }
}
