// WGSL for the Qwen3-ASR 1.7B decoder step. Shapes are baked in as constants.
// f16 data is read as packed u32 (unpack2x16float) so no shader-f16 is needed;
// all accumulation is f32.
//
// State buffer `st` (u32): [0] tok, [1] pos, [2] done, [3] n generated,
// [4] cap, [5] eos step, [16 + i] generated token i.

const head = `
@group(0) @binding(0) var<storage, read_write> st: array<u32>;
var<workgroup> wflag: u32;
fn halted(li: u32) -> bool {
  if (li == 0u) { wflag = st[2]; }
  return workgroupUniformLoad(&wflag) != 0u;
}
fn f16lo(w: u32, hi: u32) -> f32 { return unpack2x16float(w)[hi]; }
fn r16(x: f32) -> f32 { return unpack2x16float(pack2x16float(vec2<f32>(x, 0.0))).x; }
`;
// When on, every op boundary that is an f16 tensor in the ORT q4f16 graph is rounded to f16
// (math inside a kernel stays f32), so near-tied argmax steps break the way ORT's do.
export const prec = { r16: true };
const q = (e: string) => (prec.r16 ? `r16(${e})` : `(${e})`);

export function embedWGSL(rowsPerPart: number): string {
  return head + `
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<storage, read> e0: array<u32>;
@group(0) @binding(3) var<storage, read> e1: array<u32>;
@group(0) @binding(4) var<storage, read> e2: array<u32>;
@group(0) @binding(5) var<storage, read> e3: array<u32>;
@group(0) @binding(6) var<storage, read> esc: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (halted(li)) { return; }
  let tok = st[0]; let part = tok / ${rowsPerPart}u; let row = tok % ${rowsPerPart}u;
  let d = gid.x; let wi = row * 512u + (d >> 2u);
  var w = 0u;
  switch part { case 0u: { w = e0[wi]; } case 1u: { w = e1[wi]; } case 2u: { w = e2[wi]; } default: { w = e3[wi]; } }
  let b = i32(w << (24u - 8u * (d & 3u))) >> 24u;
  // ORT receives input_embeds as f16: round the same way.
  h[d] = unpack2x16float(pack2x16float(vec2<f32>(f32(b) * esc[tok], 0.0))).x;
}`;
}

// q4 GEMV, 8 rows per workgroup, 32 lanes per row, one 32-element block per lane-iteration.
// mode: "plain" y=W·x, "resid" y+=W·x, "silu" y[r]=silu(W[r]·x)*W[r+N]·x, "lm" logits + per-WG argmax.
// norm: RMSNorm(x)*nw applied on the fly (K must be 2048).
export type GemvMode = "plain" | "resid" | "silu" | "lm";
export function gemvWGSL({ K, N, mode, norm }: { K: number; N: number; mode: GemvMode; norm: boolean }): string {
  const NB = K / 32, ZB = Math.ceil(NB / 2);
  const two = mode === "silu";
  return head + `
@group(0) @binding(1) var<storage, read> x: array<f32>;
${norm ? "@group(0) @binding(2) var<storage, read> nw: array<u32>;" : ""}
@group(0) @binding(3) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> ws: array<u32>;
@group(0) @binding(5) var<storage, read> wz: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
${mode === "lm" ? "@group(0) @binding(7) var<storage, read_write> part: array<u32>;\nstruct U { rowBase: u32, partBase: u32, a: u32, b: u32 }\n@group(1) @binding(0) var<uniform> u: U;" : ""}
var<workgroup> xs: array<f32, ${norm ? K : 1}>;
var<workgroup> red: array<f32, 256>;
var<workgroup> red2: array<f32, 256>;
fn scale(i: u32) -> f32 { return f16lo(ws[i >> 1u], i & 1u); }
fn zp(row: u32, b: u32) -> f32 {
  let byte = row * ${ZB}u + (b >> 1u);
  let v = (wz[byte >> 2u] >> ((byte & 3u) * 8u)) & 255u;
  return f32((v >> ((b & 1u) * 4u)) & 15u);
}
fn xin(k: u32) -> f32 { ${norm ? "return xs[k];" : "return x[k];"} }
fn qdot(row: u32, b: u32) -> f32 {
  let q = wq[row * ${NB}u + b];
  var dq = 0.0; var sx = 0.0;
  for (var w = 0u; w < 4u; w++) {
    let qw = q[w];
    for (var j = 0u; j < 8u; j++) {
      let xv = xin(b * 32u + w * 8u + j);
      dq += f32((qw >> (4u * j)) & 15u) * xv; sx += xv;
    }
  }
  return scale(row * ${NB}u + b) * (dq - zp(row, b) * sx);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (halted(li)) { return; }
  ${norm ? `
  var ss = 0.0;
  for (var i = li; i < ${K}u; i += 256u) { let v = x[i]; ss += v * v; }
  red[li] = ss; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / ${K}.0 + 1e-6);
  for (var i = li; i < ${K}u; i += 256u) { xs[i] = ${q("x[i] * inv * f16lo(nw[i >> 1u], i & 1u)")}; }
  workgroupBarrier();` : ""}
  let lane = li & 31u; let slot = li >> 5u;
  let r = wg.x * 8u + slot;
  var acc = 0.0; var acc2 = 0.0;
  if (r < ${N}u) {
    for (var b = lane; b < ${NB}u; b += 32u) {
      acc += qdot(r, b);
      ${two ? `acc2 += qdot(r + ${N}u, b);` : ""}
    }
  }
  red[li] = acc; red2[li] = acc2; workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { red[li] += red[li + s]; red2[li] += red2[li + s]; }
    workgroupBarrier();
  }
  ${mode === "lm" ? `
  if (li == 0u) {
    var bv = -3.4e38; var bi = 0u;
    for (var k = 0u; k < 8u; k++) {
      let rr = wg.x * 8u + k;
      if (rr < ${N}u) { let v = ${q("red[k * 32u]")}; let gr = u.rowBase + rr; y[gr] = v; if (v > bv) { bv = v; bi = gr; } }
    }
    part[(u.partBase + wg.x) * 2u] = bitcast<u32>(bv); part[(u.partBase + wg.x) * 2u + 1u] = bi;
  }` : `
  if (lane == 0u && r < ${N}u) {
    let v = ${q("red[li]")};
    ${mode === "plain" ? "y[r] = v;" : mode === "resid" ? `y[r] = ${q("y[r] + v")};` : `let g = v; let sg = ${q("g * " + q("1.0 / (1.0 + exp(-g))"))}; y[r] = ${q("sg * " + q("red2[li]"))};`}
  }`}
}`;
}

// QK-norm + rope for q (heads 0..15) and k (16..23), KV append for k and v (24..31).
export function ropeWGSL({ LMAX, theta }: { LMAX: number; theta: number }): string {
  const inv = Array.from({ length: 64 }, (_, i) => Math.fround(Math.pow(theta, -(2 * i) / 128)));
  return head + `
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> qn: array<u32>;
@group(0) @binding(3) var<storage, read> kn: array<u32>;
@group(0) @binding(4) var<storage, read_write> qo: array<f32>;
@group(0) @binding(5) var<storage, read_write> kc: array<u32>;
@group(0) @binding(6) var<storage, read_write> vc: array<u32>;
struct U { layer: u32, a: u32, b: u32, c: u32 }
@group(1) @binding(0) var<uniform> u: U;
var<private> INVF: array<f32, 64> = array<f32, 64>(${inv.map((v) => v.toPrecision(9)).join(", ")});
var<workgroup> red: array<f32, 128>;
var<workgroup> xo: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) d: u32) {
  if (halted(d)) { return; }
  let hd = wg.x; let pos = st[1];
  let v = qkv[hd * 128u + d];
  var outv = v;
  if (hd < 24u) {
    red[d] = v * v; workgroupBarrier();
    for (var s = 64u; s > 0u; s >>= 1u) { if (d < s) { red[d] += red[d + s]; } workgroupBarrier(); }
    var w = 0.0;
    if (hd < 16u) { w = f16lo(qn[d >> 1u], d & 1u); } else { w = f16lo(kn[d >> 1u], d & 1u); }
    let xn = v * inverseSqrt(red[0] / 128.0 + 1e-6) * w;
    xo[d] = ${q("xn")}; workgroupBarrier();
    let ang = f32(pos) * INVF[d & 63u];
    var rot = 0.0;
    if (d < 64u) { rot = -xo[d + 64u]; } else { rot = xo[d - 64u]; }
    outv = ${q("xo[d] * cos(ang) + rot * sin(ang)")};
  }
  if (hd < 16u) { qo[hd * 128u + d] = outv; return; }
  xo[d] = outv;
  workgroupBarrier();
  if ((d & 1u) == 0u) {
    let pk = pack2x16float(vec2<f32>(xo[d], xo[d + 1u]));
    let kvh = select(hd - 24u, hd - 16u, hd < 24u);
    let idx = ((u.layer * 8u + kvh) * ${LMAX}u + pos) * 64u + (d >> 1u);
    if (hd < 24u) { kc[idx] = pk; } else { vc[idx] = pk; }
  }
}`;
}

// One workgroup per query head; scores over positions 0..pos of its KV head.
export function attnWGSL({ LMAX, scale }: { LMAX: number; scale: number }): string {
  return head + `
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<u32>;
@group(0) @binding(3) var<storage, read> vc: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
struct U { layer: u32, a: u32, b: u32, c: u32 }
@group(1) @binding(0) var<uniform> u: U;
var<workgroup> qs: array<f32, 128>;
var<workgroup> sc: array<f32, ${LMAX}>;
var<workgroup> red: array<f32, 256>;
var<workgroup> acc: array<vec2<f32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (halted(li)) { return; }
  let hq = wg.x; let T = st[1] + 1u;
  let base = (u.layer * 8u + hq / 2u) * ${LMAX}u * 64u;
  if (li < 128u) { qs[li] = q[hq * 128u + li]; }
  workgroupBarrier();
  var mx = -3.4e38;
  for (var t = li; t < T; t += 256u) {
    var s = 0.0;
    for (var w = 0u; w < 64u; w++) { let k = unpack2x16float(kc[base + t * 64u + w]); s += k.x * qs[2u * w] + k.y * qs[2u * w + 1u]; }
    s *= ${scale};
    sc[t] = s; mx = max(mx, s);
  }
  red[li] = mx; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] = max(red[li], red[li + s]); } workgroupBarrier(); }
  let m = red[0]; workgroupBarrier();
  var sum = 0.0;
  for (var t = li; t < T; t += 256u) { let p = exp(sc[t] - m); sc[t] = p; sum += p; }
  red[li] = sum; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let tot = red[0];
  let w = li & 63u; let g = li >> 6u;
  var a = vec2<f32>(0.0);
  for (var t = g; t < T; t += 4u) { a += sc[t] * unpack2x16float(vc[base + t * 64u + w]); }
  acc[li] = a; workgroupBarrier();
  if (g == 0u) {
    let r = (acc[w] + acc[w + 64u] + acc[w + 128u] + acc[w + 192u]) / tot;
    o[hq * 128u + 2u * w] = ${q("r.x")}; o[hq * 128u + 2u * w + 1u] = ${q("r.y")};
  }
}`;
}

// Reduces the LM-head per-workgroup partials, feeds the token back, sets EOS/cap flags.
export function argmaxWGSL({ NP, eos, LMAX }: { NP: number; eos: readonly number[]; LMAX: number }): string {
  return head + `
@group(0) @binding(1) var<storage, read> part: array<u32>;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  if (halted(li)) { return; }
  var v = -3.4e38; var i = 0xffffffffu;
  for (var p = li; p < ${NP}u; p += 256u) {
    let pv = bitcast<f32>(part[2u * p]); let pi = part[2u * p + 1u];
    if (pv > v || (pv == v && pi < i)) { v = pv; i = pi; }
  }
  bv[li] = v; bi[li] = i; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { let ov = bv[li + s]; let oi = bi[li + s]; if (ov > bv[li] || (ov == bv[li] && oi < bi[li])) { bv[li] = ov; bi[li] = oi; } }
    workgroupBarrier();
  }
  if (li == 0u) {
    let tok = bi[0]; let n = st[3];
    st[16u + n] = tok; st[3] = n + 1u;
    if (${eos.map((e) => `tok == ${e}u`).join(" || ")}) { st[2] = 1u; st[5] = n; return; }
    st[0] = tok; st[1] = st[1] + 1u;
    if (n + 1u >= st[4] || st[1] >= ${LMAX}u) { st[2] = 2u; }
  }
}`;
}
