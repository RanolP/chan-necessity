// WGSL for the Qwen3-ASR audio tower. Activations stay in f32 storage so the
// decoder can consume the result directly; r16() rounds at every edge the
// fp16 ONNX graph stores in fp16, so the result tracks that graph closely.
// Weights are the graph's fp16 initializers read as packed u32 pairs; linear
// weights keep the MatMul layout [in, out].
const head = `
fn f16at(w: u32, hi: u32) -> f32 { return unpack2x16float(w)[hi]; }
fn r16(x: f32) -> f32 { return unpack2x16float(pack2x16float(vec2<f32>(x, 0.0))).x; }
fn erf_(v: f32) -> f32 {
  let s = select(-1.0, 1.0, v >= 0.0); let a = abs(v); let t = 1.0 / (1.0 + 0.3275911 * a);
  let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return s * (1.0 - p * exp(-a * a));
}
// The graph's GELU: x * 0.5 * (1 + erf(x / 1.4140625)), each step in fp16.
fn gelu16(x: f32) -> f32 { return r16(x * r16(0.5 * r16(1.0 + r16(erf_(r16(x / 1.4140625)))))); }
`;
const flatIndex = `@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>`;
const flatI = `let oi = gid.x + gid.y * nw.x * 256u;`;

// Conv2d (3x3, pad 1) + GELU over [chunk, inC, inH, inW] -> [chunk, outC, outH, outW].
export function conv2dWGSL(): string {
  return head + `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
struct U { chunks: u32, inC: u32, inH: u32, inW: u32, outC: u32, outH: u32, outW: u32, stride: u32 }
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(${flatIndex}) {
  ${flatI}
  if (oi >= u.chunks * u.outC * u.outH * u.outW) { return; }
  let ow = oi % u.outW; let t0 = oi / u.outW;
  let oh = t0 % u.outH; let t1 = t0 / u.outH;
  let oc = t1 % u.outC; let c = t1 / u.outC;
  var acc = 0.0;
  for (var ic = 0u; ic < u.inC; ic++) {
    for (var kh = 0u; kh < 3u; kh++) {
      let iy = i32(oh * u.stride + kh) - 1;
      if (iy < 0 || iy >= i32(u.inH)) { continue; }
      for (var kw = 0u; kw < 3u; kw++) {
        let ix = i32(ow * u.stride + kw) - 1;
        if (ix < 0 || ix >= i32(u.inW)) { continue; }
        let xi = ((c * u.inC + ic) * u.inH + u32(iy)) * u.inW + u32(ix);
        let wi = ((oc * u.inC + ic) * 3u + kh) * 3u + kw;
        acc += r16(x[xi]) * f16at(w[wi >> 1u], wi & 1u);
      }
    }
  }
  y[oi] = gelu16(r16(acc + f16at(b[oc >> 1u], oc & 1u)));
}`;
}

// conv_out (no bias) over the [time, channel * freq] view of each chunk, plus
// the stored positional embedding for the frame's place in its chunk. Row r of
// the valid sequence is frame r % time of chunk r / time.
export function projectWGSL({ hidden, channels, freq, time }: { hidden: number; channels: number; freq: number; time: number }): string {
  return head + `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> pos: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
struct U { frames: u32 }
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(${flatIndex}) {
  ${flatI}
  if (oi >= u.frames * ${hidden}u) { return; }
  let d = oi % ${hidden}u; let row = oi / ${hidden}u;
  let chunk = row / ${time}u; let t = row % ${time}u;
  var acc = 0.0;
  for (var j = 0u; j < ${channels * freq}u; j++) {
    let xi = (chunk * ${channels * freq}u + j) * ${time}u + t;
    let wi = j * ${hidden}u + d;
    acc += x[xi] * f16at(w[wi >> 1u], wi & 1u);
  }
  let pi = t * ${hidden}u + d;
  y[oi] = r16(r16(acc) + f16at(pos[pi >> 1u], pi & 1u));
}`;
}

export function layerNormWGSL({ hidden, eps }: { hidden: number; eps: number }): string {
  return head + `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
struct U { rows: u32 }
@group(0) @binding(4) var<uniform> u: U;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wg.x; if (row >= u.rows) { return; }
  var sum = 0.0;
  for (var i = li; i < ${hidden}u; i += 256u) { sum += x[row * ${hidden}u + i]; }
  red[li] = sum; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let mean = red[0] / ${hidden}.0;
  workgroupBarrier();
  var ss = 0.0;
  for (var i = li; i < ${hidden}u; i += 256u) { let z = x[row * ${hidden}u + i] - mean; ss += z * z; }
  red[li] = ss; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / ${hidden}.0 + ${eps});
  for (var i = li; i < ${hidden}u; i += 256u) {
    let z = (x[row * ${hidden}u + i] - mean) * inv;
    y[row * ${hidden}u + i] = r16(z * f16at(w[i >> 1u], i & 1u) + f16at(b[i >> 1u], i & 1u));
  }
}`;
}

// y[row, d] = x[row, :] . W[:, d] + b[d], W stored [k, n]. One thread per
// output; neighbouring threads read neighbouring weights.
export function linearWGSL({ k, n }: { k: number; n: number }): string {
  return head + `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
struct U { rows: u32 }
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wg.x; let d = wg.y * 64u + li; if (row >= u.rows || d >= ${n}u) { return; }
  var acc = 0.0;
  for (var i = 0u; i < ${k}u; i++) { let wi = i * ${n}u + d; acc += x[row * ${k}u + i] * f16at(w[wi >> 1u], wi & 1u); }
  y[row * ${n}u + d] = r16(r16(acc) + f16at(b[d >> 1u], d & 1u));
}`;
}

// Attention within fixed windows of the frame sequence: frame f attends to the
// frames of window f / window (the last window holds the remainder).
export function attentionWGSL({ hidden, heads, headDim, window }: { hidden: number; heads: number; headDim: number; window: number }): string {
  if (window > 128) throw new Error(`attention window ${window} exceeds shader score storage`);
  return head + `
@group(0) @binding(0) var<storage, read> qv: array<f32>;
@group(0) @binding(1) var<storage, read> kv: array<f32>;
@group(0) @binding(2) var<storage, read> vv: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
struct U { frames: u32 }
@group(0) @binding(4) var<uniform> u: U;
var<workgroup> query: array<f32, ${headDim}>;
var<workgroup> scores: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let flat = wg.x + wg.y * nw.x; let frame = flat / ${heads}u; let h = flat % ${heads}u;
  if (frame >= u.frames) { return; }
  if (li < ${headDim}u) { query[li] = qv[frame * ${hidden}u + h * ${headDim}u + li]; }
  workgroupBarrier();
  let first = (frame / ${window}u) * ${window}u; let count = min(${window}u, u.frames - first);
  var s = -3.4e38;
  if (li < count) {
    var dot = 0.0;
    for (var d = 0u; d < ${headDim}u; d++) { dot += query[d] * kv[(first + li) * ${hidden}u + h * ${headDim}u + d]; }
    s = r16(r16(dot) * ${1 / Math.sqrt(headDim)});
  }
  scores[li] = s; red[li] = s;
  workgroupBarrier();
  for (var z = 64u; z > 0u; z >>= 1u) { if (li < z) { red[li] = max(red[li], red[li + z]); } workgroupBarrier(); }
  let mx = red[0];
  workgroupBarrier();
  let e = select(0.0, exp(scores[li] - mx), li < count);
  scores[li] = e; red[li] = e;
  workgroupBarrier();
  for (var z = 64u; z > 0u; z >>= 1u) { if (li < z) { red[li] += red[li + z]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  if (li < ${headDim}u) {
    var acc = 0.0;
    for (var t = 0u; t < count; t++) { acc += r16(scores[t] * inv) * vv[(first + t) * ${hidden}u + h * ${headDim}u + li]; }
    y[frame * ${hidden}u + h * ${headDim}u + li] = r16(acc);
  }
}`;
}

export function geluWGSL(): string {
  return head + `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
struct U { n: u32 }
@group(0) @binding(1) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(${flatIndex}) { ${flatI} if (oi < u.n) { x[oi] = gelu16(x[oi]); } }
`;
}

export function addWGSL(): string {
  return head + `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
struct U { n: u32 }
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(${flatIndex}) { ${flatI} if (oi < u.n) { y[oi] = r16(a[oi] + b[oi]); } }
`;
}
