// Web Worker for HOH quantum simulation computation
// Uses flat Float64Arrays for ~10-20x speedup over nested JS arrays

const NN = 100;
const S = NN + 1;       // 101
const S2 = S * S;       // 10201
const S3 = S * S * S;   // 1030301
const N2 = 50;
const N3 = 50;
const N4 = 45;
const D = 15;
const M = 6;   // electrons 1..5
const R1 = 2;
const R2 = 0.9;
const R3 = 2;
const d = 0.2;
const h = 10 / NN;
const h2 = h * h;
const h3 = h * h * h;
const inv_h = 1 / h;
const inv_h2 = 1 / h2;
const dt = d * h2;
const TWO_PI = 2 * Math.PI;
const half_d = 0.5 * d;

// Strides for neighbor access
const di = S2;  // stride in i
const dj = S;   // stride in j
const dk = 1;   // stride in k

// Flat typed arrays: u[m], w[m], P[m] for m=1..5, K is shared
const u = new Array(M);
const w = new Array(M);
const P = new Array(M);
const K = new Float64Array(S3);
const charge = new Float64Array(M);
const norm = new Float64Array(M);
const c = new Float64Array(M);

for (let m = 1; m < M; m++) {
  u[m] = new Float64Array(S3);
  w[m] = new Float64Array(S3);
  P[m] = new Float64Array(S3);
  charge[m] = 1;
}

function idx(i, j, k) {
  return i * S2 + j * S + k;
}

// Setup: initialize arrays
function setup() {
  const softening = 0.16 * h2;

  for (let i = 0; i <= NN; i++) {
    const di_o = (i - N3) * h;
    const di_h1 = (i - N3 - D) * h;
    for (let j = 0; j <= NN; j++) {
      const dj_o = (j - N2) * h;
      const dj_h1 = (j - N2 + D) * h;
      const dj_h2 = (j - N2 - D) * h;
      for (let k = 0; k <= NN; k++) {
        const dk_o = (k - N2) * h;
        const id = idx(i, j, k);

        // Distance to oxygen
        const r2_o = di_o * di_o + dj_o * dj_o + dk_o * dk_o + softening;
        const r = Math.sqrt(r2_o);
        const inv_r = 1 / r;

        // Distance to H1
        const r2_h1 = di_h1 * di_h1 + dj_h1 * dj_h1 + dk_o * dk_o + softening;
        const r1 = Math.sqrt(r2_h1);
        const inv_r1 = 1 / r1;

        // Distance to H2
        const r2_h2 = di_h1 * di_h1 + dj_h2 * dj_h2 + dk_o * dk_o + softening;
        const r4 = Math.sqrt(r2_h2);
        const inv_r4 = 1 / r4;

        // Initial wavefunctions
        if (r > R2 && r < R3 && i < N4) {
          u[1][id] = Math.exp(-3 * r);
          w[1][id] = 1;
        }
        if (r > R2 && r < R3 && i > N4 && j < N2) {
          u[2][id] = Math.exp(-3 * r);
          w[2][id] = 1;
        }
        if (r > R2 && r < R3 && i > N4 && j > N2) {
          u[3][id] = Math.exp(-3 * r);
          w[3][id] = 1;
        }
        if (r1 < R1 && r > R3) {
          u[4][id] = Math.exp(-r1);
          w[4][id] = 1;
        }
        if (r4 < R1 && r > R3) {
          u[5][id] = Math.exp(-r4);
          w[5][id] = 1;
        }

        // Potentials
        P[1][id] = inv_r + 0.5 * inv_r1 + 0.5 * inv_r4;
        P[2][id] = inv_r + 0.5 * inv_r1 + 0.5 * inv_r4;
        P[3][id] = inv_r + 0.5 * inv_r1 + 0.5 * inv_r4;
        P[4][id] = 1.5 * inv_r + 0.5 * inv_r1 + 0.5 * inv_r4;
        P[5][id] = 1.5 * inv_r + 0.5 * inv_r1 + 0.5 * inv_r4;

        // Kernel
        K[id] = 3 * inv_r + inv_r1 + inv_r4;
      }
    }
  }
}

// One time step
function timeStep() {
  let E = 0;

  for (let i = 1; i < NN; i++) {
    for (let j = 1; j < NN; j++) {
      for (let k = 1; k < NN; k++) {
        const id = idx(i, j, k);

        // Precompute c[m] for all electrons at this grid point
        for (let m = 1; m < M; m++) {
          let cm = 0;
          for (let n = 1; n < M; n++) {
            if (n !== m) cm -= u[n][id];
          }
          c[m] = 0.5 * (cm + u[m][id]);
        }

        // Precompute r^2 (only depends on i,j,k, not m)
        const di_o = (i - N3) * h;
        const dj_o = (j - N2) * h;
        const dk_o = (k - N2) * h;
        const r2val = di_o * di_o + dj_o * dj_o + dk_o * dk_o + 0.16 * h2;
        const rCheck = r2val > R2;  // compare r^2 > R2 (note: R2=0.9, r^2 vs r)

        // Precompute neighbor indices
        const id_ip = id + di;
        const id_im = id - di;
        const id_jp = id + dj;
        const id_jm = id - dj;
        const id_kp = id + dk;
        const id_km = id - dk;

        const Kval = K[id];

        for (let m = 1; m < M; m++) {
          const um = u[m];
          const wm = w[m];
          const Pm = P[m];

          const u_c = um[id];
          const w_c = wm[id];
          const w_ip = wm[id_ip];
          const w_im = wm[id_im];
          const w_jp = wm[id_jp];
          const w_jm = wm[id_jm];
          const w_kp = wm[id_kp];
          const w_km = wm[id_km];

          const u_ip = um[id_ip];
          const u_im = um[id_im];
          const u_jp = um[id_jp];
          const u_jm = um[id_jm];
          const u_kp = um[id_kp];
          const u_km = um[id_km];

          // Front tracking of characteristic functions
          if (rCheck) {
            const lap_w = (w_ip + w_im + w_jp + w_jm + w_kp + w_km - 6 * w_c) * inv_h2;
            const gx = (w_ip - w_im) * inv_h;
            const gy = (w_jp - w_jm) * inv_h;
            const gz = (w_kp - w_km) * inv_h;
            const grad_mag = Math.sqrt(gx * gx + gy * gy + gz * gz);
            const cm = c[m];
            wm[id] = w_c + dt * Math.abs(cm) * lap_w + 5 * dt * cm * grad_mag;
          }

          // Update electron density (Neumann diffusion)
          const du_ip = u_ip - u_c;
          const du_im = u_c - u_im;
          const du_jp = u_jp - u_c;
          const du_jm = u_c - u_jm;
          const du_kp = u_kp - u_c;
          const du_km = u_c - u_km;

          um[id] = u_c
            + half_d * (du_ip * (w_ip + w_c) * 0.5 - du_im * (w_c + w_im) * 0.5)
            + half_d * (du_jp * (w_jp + w_c) * 0.5 - du_jm * (w_c + w_jm) * 0.5)
            + half_d * (du_kp * (w_kp + w_c) * 0.5 - du_km * (w_c + w_km) * 0.5)
            + dt * (Kval - 2 * Pm[id]) * u_c * w_c;

          // Poisson equation for electron potentials
          let cm2 = 0;
          for (let n = 1; n < M; n++) {
            if (n !== m) {
              const un = u[n][id];
              cm2 += un * un;
            }
          }
          const P_c = Pm[id];
          Pm[id] = P_c + dt * (Pm[id_ip] + Pm[id_im] + Pm[id_jp] + Pm[id_jm] + Pm[id_kp] + Pm[id_km] - 6 * P_c) * inv_h2 + TWO_PI * dt * cm2;

          // Energy
          if (w_c > 0.44) {
            E += 0.5 * (du_ip * du_ip + du_jp * du_jp + du_kp * du_kp) * h;
          }
          const u_new = um[id];
          E += (Pm[id] - Kval) * u_new * u_new * h3;
        }
      }
    }
  }

  // Normalisation
  for (let m = 1; m < M; m++) {
    let nrm = 0;
    const um = u[m];
    for (let i = 1; i < NN; i++) {
      for (let j = 1; j < NN; j++) {
        const base = i * S2 + j * S;
        for (let k = 1; k < NN; k++) {
          const val = um[base + k];
          nrm += val * val;
        }
      }
    }
    nrm *= h3;
    if (nrm > 0) {
      const scale = Math.sqrt(charge[m]) / Math.sqrt(nrm);
      for (let i = 1; i < NN; i++) {
        for (let j = 1; j < NN; j++) {
          const base = i * S2 + j * S;
          for (let k = 1; k < NN; k++) {
            um[base + k] *= scale;
          }
        }
      }
    }
  }

  return E;
}

// Extract a 2D slice for rendering: u[m][i][j][N2] for all i,j
function extractSlice(arr, kSlice) {
  const slice = new Float64Array(S * S);
  for (let i = 0; i <= NN; i++) {
    for (let j = 0; j <= NN; j++) {
      slice[i * S + j] = arr[i * S2 + j * S + kSlice];
    }
  }
  return slice;
}

// Extract a 1D line for diagnostic plots
function extractLine(arr, j0, k0) {
  const line = new Float64Array(S);
  for (let i = 0; i <= NN; i++) {
    line[i] = arr[i * S2 + j0 * S + k0];
  }
  return line;
}

// Message handler
self.onmessage = function(e) {
  if (e.data.type === 'init') {
    setup();
    self.postMessage({ type: 'ready' });
  } else if (e.data.type === 'step') {
    const t0 = performance.now();
    const E = timeStep();
    const elapsed = performance.now() - t0;

    // Extract rendering data
    const slices = {};
    const wLines = {};
    const uLines = {};
    const pLines = {};
    for (let m = 1; m < M; m++) {
      slices[m] = extractSlice(u[m], N2);
      wLines[m] = extractLine(w[m], N2 + 8, N2);
      uLines[m] = extractLine(u[m], N2 + 5, N2);
      pLines[m] = extractLine(P[m], N2, N2);
    }
    const kLine = extractLine(K, N2, N2);

    self.postMessage({
      type: 'result',
      E: E,
      elapsed: elapsed,
      slices: slices,
      wLines: wLines,
      uLines: uLines,
      pLines: pLines,
      kLine: kLine,
      S: S
    });
  }
};
