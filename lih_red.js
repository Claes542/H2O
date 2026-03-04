// H2-like Quantum Simulation — WebGPU Compute Shaders
// e1 around k1 (Z=1), e2 around k2 (Z=1), w2=0 inside r_cut of k2

const NN = 200;
const S = NN + 1;
const S2 = S * S;
const S3 = S * S * S;
const N2 = Math.round(NN / 2);
const D_bond = 60;   // 3 au
const r_cut = 1.0;   // au

function makeBondPos(D) {
  // Two atoms along i-axis, centered at [N2, N2, N2]
  const half = Math.round(D / 2);
  return {
    H0: [N2 - half, N2, N2],
    H1: [N2 + half, N2, N2]
  };
}
const nucPos = makeBondPos(D_bond);
const R1 = 100;  // large, no outer cutoff
let E_min = Infinity;
const hv = 10 / NN, h2v = hv * hv, h3v = hv * hv * hv;
const dv = 0.12;
const dtv = dv * h2v, half_dv = 0.5 * dv;
const PX = 400 / NN;
const INTERIOR = (NN - 1) * (NN - 1) * (NN - 1);
const STEPS_PER_FRAME = 500;
const NORM_INTERVAL = 20;
const NELEC = 2;
const NRED = NELEC + 1;  // 3 values: 2 norms + 1 energy

// ===== WGSL SHADERS =====

const paramStructWGSL = `
struct P {
  NN: u32, S: u32, S2: u32, S3: u32,
  N2: u32, h1I: u32,
  h: f32, h2: f32, inv_h: f32, inv_h2: f32,
  dt: f32, half_d: f32, r_cut: f32, TWO_PI: f32, h3: f32,
  _pad2: u32
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

  // c[m]: other electron's density
  var cm: f32 = 0.0;
  cm -= Ui[0u * p.S3 + id] + Ui[1u * p.S3 + id];
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
  var nw = wc + 0.5 * p.dt * abs(cm) * lw + 5.0 * p.dt * cm * sqrt(gx * gx + gy * gy + gz * gz);
  nw = clamp(nw, 0.0, 1.0);

  // inside r_cut of k2: smooth cutoff for w2, hard freeze for w1/u1
  let di = f32(i) - f32(p.h1I);
  let dj = f32(j) - f32(p.N2);
  let dkk = f32(k) - f32(p.N2);
  let r_k2 = sqrt(di * di + dj * dj + dkk * dkk) * p.h;
  let in_cut = r_k2 < p.r_cut;
  if (m == 0u && in_cut) { nw = wc; }
  if (m == 1u && in_cut) {
    let edge = p.r_cut - 3.0 * p.h;
    let t = clamp((r_k2 - edge) / (p.r_cut - edge), 0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    nw = min(nw, s);
  }
  let inside_cut = m == 0u && in_cut;
  Wo[o + id] = nw;

  let uc  = Ui[o + id];
  let uip = Ui[o + id + p.S2]; let uim = Ui[o + id - p.S2];
  let ujp = Ui[o + id + p.S];  let ujm = Ui[o + id - p.S];
  let ukp = Ui[o + id + 1u];   let ukm = Ui[o + id - 1u];

  if (inside_cut) {
    Uo[o + id] = uc;
  } else {
    Uo[o + id] = uc
      + p.half_d * ((uip - uc) * (wip + nw) * 0.5 - (uc - uim) * (nw + wim) * 0.5)
      + p.half_d * ((ujp - uc) * (wjp + nw) * 0.5 - (uc - ujm) * (nw + wjm) * 0.5)
      + p.half_d * ((ukp - uc) * (wkp + nw) * 0.5 - (uc - ukm) * (nw + wkm) * 0.5)
      + p.dt * (K[id] - 2.0 * Pi[o + id]) * uc * wc;
  }

  let Pc = Pi[o + id];
  var rho: f32 = 0.0;
  let u0 = Ui[0u * p.S3 + id]; let u1 = Ui[1u * p.S3 + id];
  rho = u0*u0 + u1*u1;
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

var<workgroup> sn: array<f32, 768>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;

  for (var x: u32 = 0u; x < 3u; x++) { sn[lid * 3u + x] = 0.0; }

  if (gid.x < tot) {
    let k = (gid.x % NM) + 1u;
    let j = ((gid.x / NM) % NM) + 1u;
    let i = (gid.x / (NM * NM)) + 1u;
    let id = i * p.S2 + j * p.S + k;

    var en: f32 = 0.0;
    for (var m: u32 = 0u; m < 2u; m++) {
      let o = m * p.S3;
      let v = U[o + id];
      sn[lid * 3u + m] = v * v * p.h3;
      if (W[o + id] > 0.7) {
        let a = U[o + id + p.S2] - v;
        let b = U[o + id + p.S] - v;
        let c = U[o + id + 1u] - v;
        en += 0.5 * (a * a + b * b + c * c) * p.h;
      }
      en += (Pv[o + id] - K[id]) * v * v * p.h3;
    }
    sn[lid * 3u + 2u] = en;
  }

  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var x: u32 = 0u; x < 3u; x++) {
        sn[lid * 3u + x] += sn[(lid + s) * 3u + x];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    let base = wgid.x * 3u;
    for (var x: u32 = 0u; x < 3u; x++) {
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

var<workgroup> wg: array<f32, 768>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
  for (var x: u32 = 0u; x < 3u; x++) { wg[lid * 3u + x] = 0.0; }

  for (var i: u32 = lid; i < nwg.count; i += 256u) {
    for (var x: u32 = 0u; x < 3u; x++) {
      wg[lid * 3u + x] += partials[i * 3u + x];
    }
  }

  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var x: u32 = 0u; x < 3u; x++) {
        wg[lid * 3u + x] += wg[(lid + s) * 3u + x];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    for (var x: u32 = 0u; x < 3u; x++) {
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

  for (var m: u32 = 0u; m < 2u; m++) {
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

  for (var m: u32 = 0u; m < 2u; m++) {
    let idx = m * p.S3 + i * p.S2 + j * p.S + p.N2;
    out[m * SS * SS + i * SS + j] = select(0.0, U[idx], W[idx] > 0.0);
  }

  if (j == 0u) {
    let b = 2u * SS * SS;
    for (var m: u32 = 0u; m < 2u; m++) {
      out[b + m * SS + i] = W[m * p.S3 + i * p.S2 + (p.N2 + 8u) * p.S + p.N2];
      let uIdx = m * p.S3 + i * p.S2 + (p.N2 + 5u) * p.S + p.N2;
      out[b + 2u * SS + m * SS + i] = select(0.0, U[uIdx], W[uIdx] > 0.0);
      out[b + 4u * SS + m * SS + i] = Pv[m * p.S3 + i * p.S2 + p.N2 * p.S + p.N2];
    }
    out[b + 6u * SS + i] = K[i * p.S2 + p.N2 * p.S + p.N2];
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
const STEPS_PHASE = 5000;

const SLICE_SIZE = (2 * S * S + 7 * S) * 4;
const WG_UPDATE = Math.ceil(INTERIOR / 256);
const WG_REDUCE = Math.ceil(INTERIOR / 256);
const WG_NORM = Math.ceil(INTERIOR / 256);
const WG_EXTRACT = Math.ceil(S / 16);
const SUMS_BYTES = NRED * 4;  // 12 bytes

let sliceData = null;

function uploadInitialData(pos) {
  const [h0I, h0J, h0K] = pos.H0;
  const [h1I, h1J, h1K] = pos.H1;
  console.log("Init: H0=(" + h0I + "," + h0J + "," + h0K + ") H1=(" + h1I + "," + h1J + "," + h1K + ")");

  const Kd = new Float32Array(S3);
  const Ud = new Float32Array(NELEC * S3);
  const Wd = new Float32Array(NELEC * S3);
  const Pd = new Float32Array(NELEC * S3);
  const soft = 0.04 * h2v;

  for (let i = 0; i <= NN; i++) {
    const dxH0 = (i - h0I) * hv, dxH1 = (i - h1I) * hv;
    for (let j = 0; j <= NN; j++) {
      const dyH0 = (j - h0J) * hv, dyH1 = (j - h1J) * hv;
      for (let k = 0; k <= NN; k++) {
        const dzH0 = (k - h0K) * hv, dzH1 = (k - h1K) * hv;
        const id = i * S2 + j * S + k;

        const r1 = Math.sqrt(dxH0*dxH0 + dyH0*dyH0 + dzH0*dzH0 + soft);
        const r2 = Math.sqrt(dxH1*dxH1 + dyH1*dyH1 + dzH1*dzH1 + soft);
        const ir1 = 1/r1, ir2 = 1/r2;

        // Both point charges +1
        Kd[id] = ir1 + ir2;

        // u1/w1 excluded inside r_cut of k2; u2 free everywhere
        const u1 = Math.exp(-r1);
        const u2 = Math.exp(-r2);
        const R_out = 2.5;  // au
        if (r2 <= r_cut) {
          // m=0: excluded (u0=0, w0=0)
          // m=1: smooth w2 inside cutoff, u2 free
          const edge = r_cut - 3 * hv;
          const t = Math.max(0, Math.min(1, (r2 - edge) / (r_cut - edge)));
          const s = t * t * (3 - 2 * t);
          Ud[1*S3+id] = u2;
          Wd[1*S3+id] = s;
        } else if (u1 >= u2) {
          Ud[0*S3+id] = u1; Wd[0*S3+id] = r1 <= R_out ? 1 : 0;
        } else {
          Ud[1*S3+id] = u2; Wd[1*S3+id] = r2 <= R_out ? 1 : 0;
        }

        for (let m = 0; m < NELEC; m++) {
          Pd[m*S3+id] = 0.5 * ir1 + 0.5 * ir2;
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

    device = await adapter.requestDevice();
    console.log("WebGPU device ready");

    device.lost.then((info) => {
      gpuError = "GPU device lost: " + info.message;
      console.error(gpuError);
      gpuReady = false;
    });

    const bs = S3 * 4, bN = NELEC * S3 * 4;
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

    const pb = new ArrayBuffer(64);
    const pu = new Uint32Array(pb);
    const pf = new Float32Array(pb);
    pu[0] = NN; pu[1] = S; pu[2] = S2; pu[3] = S3; pu[4] = N2; pu[5] = nucPos.H1[0];
    console.log("h1I=" + nucPos.H1[0] + " N2=" + N2 + " hv=" + hv + " 1au=" + (1/hv).toFixed(1) + " grid pts");
    pf[6] = hv; pf[7] = h2v; pf[8] = 1 / hv; pf[9] = 1 / h2v;
    pf[10] = dtv; pf[11] = half_dv; pf[12] = r_cut; pf[13] = 2 * Math.PI;
    pf[14] = h3v;
    pu[15] = 0;
    paramsBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuf, 0, pb);

    uploadInitialData(nucPos);

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

    console.log("Ready! dispatch(" + WG_UPDATE + ",2,1)");
    gpuReady = true;

  } catch (e) {
    gpuError = e.message || String(e);
    console.error("GPU init failed:", e);
  }
}

async function doSteps(n) {
  const t0 = performance.now();
  const enc = device.createCommandEncoder();

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
  E = sumsData[2];  // energy at index 2 (after 2 norms)

  await sliceReadBuf.mapAsync(GPUMapMode.READ);
  sliceData = new Float32Array(sliceReadBuf.getMappedRange().slice(0));
  sliceReadBuf.unmap();

  tStep += n;
  lastMs = performance.now() - t0;

  // Nuclear repulsion: Z=1 for both
  const soft_nuc = 0.04 * h2v;
  const d01 = Math.sqrt(
    ((nucPos.H0[0]-nucPos.H1[0])*hv)**2 +
    ((nucPos.H0[1]-nucPos.H1[1])*hv)**2 +
    ((nucPos.H0[2]-nucPos.H1[2])*hv)**2 + soft_nuc);
  E += 1/d01;

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

  if (!computing) {
    computing = true;
    doSteps(STEPS_PER_FRAME).then(() => {
      computing = false;
      if (isFinite(E) && E < E_min) E_min = E;
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
        // 2 electrons: red, green
        const u0 = 500 * sliceData[0 * SS * SS + b];
        const u1 = 500 * sliceData[1 * SS * SS + b];
        const ri = Math.min(255, Math.floor(u0));
        const gi = Math.min(255, Math.floor(u1));
        for (let py = py0; py < py1 && py < H; py++) {
          for (let px = px0; px < px1 && px < W; px++) {
            const idx = (py * W + px) * 4;
            pixels[idx] = ri;
            pixels[idx + 1] = gi;
            pixels[idx + 2] = 0;
            pixels[idx + 3] = 255;
          }
        }
      }
    }
    updatePixels();

    // Line plots
    const lb = 2 * SS * SS;
    for (let i = 1; i < NN - 10; i++) {
      for (let m = 0; m < 2; m++) {
        fill(255); ellipse(PX * i, 300 - 100 * sliceData[lb + m * SS + i], 3);
        fill(m === 0 ? 255 : 0, m === 1 ? 255 : 0, 0); ellipse(PX * i, 300 - 100 * sliceData[lb + 2 * SS + m * SS + i], 3);
        fill(0, 255, 255, 255); ellipse(PX * i, 300 - 30 * sliceData[lb + 4 * SS + m * SS + i], 2);
      }
      fill(0, 0, 255, 255); ellipse(PX * i, 300 - 30 * sliceData[lb + 6 * SS + i], 3);
    }
  }

  // Draw nuclear positions
  fill(255); stroke(255); strokeWeight(1);
  circle(nucPos.H0[0] * PX, nucPos.H0[1] * PX, 6);
  circle(nucPos.H1[0] * PX, nucPos.H1[1] * PX, 6);
  noStroke();

  fill(255);
  const r = D_bond * hv;
  text("e1+e2 k1+k2 w2=0 inside r_cut | WebGPU, " + NN + "^3", 5, 20);
  text("D=" + D_bond + " r=" + r.toFixed(2) + " rc=" + r_cut, 5, 35);
  text("step " + tStep + "  E=" + E.toFixed(6) + "  min=" + (isFinite(E_min) ? E_min.toFixed(6) : "---"), 5, 50);
  if (lastMs > 0) text((lastMs / STEPS_PER_FRAME).toFixed(1) + "ms/step", 300, 50);
}
