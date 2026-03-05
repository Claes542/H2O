// 3D visualization overlay for molecule.js
// Load after molecule.js, before p5.js

const STRIDE_3D = 8;
const SG = Math.floor(NN / STRIDE_3D) + 1;
const SG3 = SG * SG * SG;
const SLICE3D_SIZE = NELEC * SG3 * 4;

let extract3DPL, extract3DBG = [], slice3DBuf, slice3DReadBuf;
let slice3DData = null;

const extract3DWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let SG = ${SG}u;
  if (g.x >= SG || g.y >= SG || g.z >= SG) { return; }

  let i = g.x * ${STRIDE_3D}u;
  let j = g.y * ${STRIDE_3D}u;
  let k = g.z * ${STRIDE_3D}u;

  let outIdx = g.x * SG * SG + g.y * SG + g.z;
  let sg3 = SG * SG * SG;

  for (var m: u32 = 0u; m < ${NELEC}u; m++) {
    let idx = m * p.S3 + i * p.S2 + j * p.S + k;
    let u = U[idx];
    out[m * sg3 + outIdx] = select(0.0, u * u, W[idx] > 0.0);
  }
}
`;

const WG_3D = Math.ceil(SG / 4);

// Save original functions
const _origInitGPU = initGPU;
const _origDoSteps = doSteps;

// Override setup for WEBGL
setup = function() {
  createCanvas(500, 500, WEBGL);
  textFont('monospace');
  textSize(11);
  initGPU3D();
};

async function initGPU3D() {
  await _origInitGPU();
  if (!gpuReady) return;

  const mod = device.createShaderModule({ code: extract3DWGSL });
  extract3DPL = await device.createComputePipelineAsync({
    layout: 'auto', compute: { module: mod, entryPoint: 'main' }
  });

  slice3DBuf = device.createBuffer({
    size: SLICE3D_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  slice3DReadBuf = device.createBuffer({
    size: SLICE3D_SIZE,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });

  for (let c = 0; c < 2; c++) {
    extract3DBG[c] = device.createBindGroup({
      layout: extract3DPL.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: W_buf[c] } },
        { binding: 3, resource: { buffer: slice3DBuf } },
      ]
    });
  }
  console.log("3D extract ready: SG=" + SG + " stride=" + STRIDE_3D);
}

// Wrap doSteps to add 3D extract
doSteps = async function(n) {
  await _origDoSteps(n);

  if (!extract3DPL) return;

  const enc = device.createCommandEncoder();
  const cp = enc.beginComputePass();
  cp.setPipeline(extract3DPL);
  cp.setBindGroup(0, extract3DBG[cur]);
  cp.dispatchWorkgroups(WG_3D, WG_3D, WG_3D);
  cp.end();
  enc.copyBufferToBuffer(slice3DBuf, 0, slice3DReadBuf, 0, SLICE3D_SIZE);
  device.queue.submit([enc.finish()]);

  await slice3DReadBuf.mapAsync(GPUMapMode.READ);
  slice3DData = new Float32Array(slice3DReadBuf.getMappedRange().slice(0));
  slice3DReadBuf.unmap();
};

// Override draw for 3D rendering
draw = function() {
  background(0);

  if (gpuError) {
    push(); translate(-width/2, -height/2);
    fill(255, 0, 0); textSize(11);
    text("GPU Error: " + gpuError, 10, 200);
    pop(); return;
  }
  if (!gpuReady) {
    push(); translate(-width/2, -height/2);
    fill(255); text("Initializing WebGPU...", 10, 200);
    pop(); return;
  }

  // Simulation step
  if (!computing && phase === 0) {
    computing = true;
    doSteps(STEPS_PER_FRAME).then(() => {
      computing = false;
      phaseSteps += STEPS_PER_FRAME;
      if (phaseSteps >= 5000) {
        device.queue.writeBuffer(paramsBuf, 27 * 4, new Float32Array([0.0]));
      }
      if (isFinite(E) && E < E_min) E_min = E;
      if (phaseSteps >= TOTAL_STEPS) {
        console.log("DONE: E=" + E.toFixed(6));
        phase = 1;
      }
    }).catch((e) => {
      gpuError = e.message || String(e);
      computing = false;
    });
  }

  // 3D scene
  orbitControl();
  scale(1.2);

  if (slice3DData) {
    const half = NN * PX / 2;
    const sp = STRIDE_3D * PX;
    noFill();
    strokeWeight(sp * 0.9);

    const zC = {1:[255,255,255], 2:[255,50,50], 3:[50,100,255], 4:[50,255,50]};
    const eCol = Z.slice(0, NELEC).map(z => zC[z] || [128,128,128]);

    for (let m = 0; m < NELEC; m++) {
      if (Z[m] === 0) continue;
      const c = eCol[m];
      const base = m * SG3;
      beginShape(POINTS);
      for (let gi = 0; gi < SG; gi++) {
        const xp = gi * sp - half;
        const gi_off = gi * SG * SG;
        for (let gj = 0; gj < SG; gj++) {
          const yp = gj * sp - half;
          const gj_off = gi_off + gj * SG;
          for (let gk = 0; gk < SG; gk++) {
            const v = slice3DData[base + gj_off + gk];
            if (v < 0.002) continue;
            const a = Math.min(200, v * 600);
            stroke(c[0], c[1], c[2], a);
            vertex(xp, yp, gk * sp - half);
          }
        }
      }
      endShape();
    }

    // Nuclei as white dots
    strokeWeight(8);
    stroke(255);
    beginShape(POINTS);
    for (let n = 0; n < NELEC; n++) {
      if (Z[n] === 0) continue;
      vertex(
        nucPos[n][0] * PX - half,
        nucPos[n][1] * PX - half,
        nucPos[n][2] * PX - half
      );
    }
    endShape();

    // Wireframe cube
    noFill(); stroke(80); strokeWeight(1);
    box(NN * PX);
  }

  // HUD text
  push();
  translate(-width/2 + 10, -height/2 + 20, 0);
  fill(255); noStroke();
  const pLabel = phase === 0 ? "running" : "DONE";
  text(pLabel + " step " + tStep + "/" + TOTAL_STEPS + "  E=" + E.toFixed(4), 0, 0);
  text("T=" + E_T.toFixed(3) + " VeK=" + E_eK.toFixed(3) + " Vee=" + E_ee.toFixed(3) + " VKK=" + E_KK.toFixed(3), 0, 14);
  pop();
};
