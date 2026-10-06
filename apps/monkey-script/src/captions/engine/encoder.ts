// Model-specific Qwen3-ASR audio encoder, mirroring the encoder.fp16.onnx
// graph: the mel is cut into 100-frame chunks that are convolved separately,
// the valid conv frames of all chunks are concatenated, and attention runs in
// fixed 104-frame windows over that sequence. Weights are uploaded once from
// the ONNX bytes (ranges found by parseEncoderManifest) and the result stays a
// GPU buffer, [frames, outputDim] f32, which the decoder prefill reads directly.
import * as K from "./encoder-kernels.ts";
import { parseEncoderManifest } from "./encoder-onnx.ts";

export interface EncoderTensor {
  offset: number;
  length: number;
  dtype: string;
  shape: number[];
}
interface Linear { weight: string; bias: string }
export interface EncoderLayerRoles {
  index: number;
  q: Linear; k: Linear; v: Linear; o: Linear;
  ln1: Linear; fc1: Linear; fc2: Linear; ln2: Linear;
}
export interface EncoderManifest {
  schema: number;
  source: string;
  config: {
    mel: number;
    chunkFrames: number;
    attentionWindowFrames: number;
    convFrames: number;
    convChannels: number;
    convFreq: number;
    hidden: number;
    intermediate: number;
    heads: number;
    headDim: number;
    layers: number;
    outputDim: number;
    layerNormEps: number;
  };
  roles: {
    conv: Linear[];
    convOut: string;
    positions: string;
    lnPost: Linear;
    proj1: Linear;
    proj2: Linear;
    layers: EncoderLayerRoles[];
  };
  tensors: Record<string, EncoderTensor>;
}
export interface EncoderResult {
  // Owned by the caller, who destroys it once the decoder has read it.
  buffer: GPUBuffer;
  frames: number;
  dim: number;
}

const S = GPUBufferUsage.STORAGE;
const CD = GPUBufferUsage.COPY_DST;
const CS = GPUBufferUsage.COPY_SRC;
const align4 = (n: number) => Math.max(4, (n + 3) & ~3);
const MAX_GROUPS = 65535;

export function encoderFrames(T: number, chunkFrames = 100, convFrames = 13) {
  const chunks = Math.ceil(T / chunkFrames);
  return (chunks - 1) * convFrames + convLength(T - (chunks - 1) * chunkFrames);
}

// `wait` wraps the one completion wait per run; the worker passes the
// engine's makePump, since Firefox settles queue promises only while polled.
export async function createEncoder(device: GPUDevice, model: ArrayBuffer | Uint8Array, { wait = <T,>(p: Promise<T>) => p }: { wait?: <T>(p: Promise<T>) => Promise<T> } = {}) {
  const bytes = model instanceof Uint8Array ? model : new Uint8Array(model);
  const manifest = parseEncoderManifest(bytes);
  const C = manifest.config;
  if (C.convFrames !== 13 || C.convFreq !== 16 || C.hidden % C.heads) throw new Error(`unsupported encoder dimensions ${JSON.stringify(C)}`);
  const window = Math.round(C.attentionWindowFrames / C.chunkFrames * C.convFrames);

  const weights = new Map<string, GPUBuffer>();
  const weight = (name: string) => {
    const hit = weights.get(name);
    if (hit) return hit;
    const t = manifest.tensors[name];
    if (!t) throw new Error(`encoder.fp16.onnx has no tensor ${name}`);
    if (t.dtype !== "float16") throw new Error(`${name}: expected float16, got ${t.dtype}`);
    if (t.length % 4) throw new Error(`${name}: byte length ${t.length} is not 4-byte aligned`);
    const b = device.createBuffer({ size: t.length, usage: S | CD });
    device.queue.writeBuffer(b, 0, bytes, t.offset, t.length);
    weights.set(name, b);
    return b;
  };
  const R = manifest.roles;
  for (const c of R.conv) { weight(c.weight); weight(c.bias); }
  weight(R.convOut); weight(R.positions);
  for (const l of [R.lnPost, R.proj1, R.proj2, ...R.layers.flatMap((x) => [x.q, x.k, x.v, x.o, x.ln1, x.fc1, x.fc2, x.ln2])]) { weight(l.weight); weight(l.bias); }

  const mk = async (name: string, code: string) => {
    const module = device.createShaderModule({ code });
    try {
      return await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    } catch (error) {
      const info = await module.getCompilationInfo();
      const notes = info.messages.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join("\n");
      throw new Error(`encoder pipeline ${name}: ${(error as Error).message}\n${notes}`);
    }
  };
  const [conv, project, norm, linH, attnP, gelu, add, fc1, fc2, proj2] = await Promise.all([
    mk("conv2d", K.conv2dWGSL()),
    mk("conv_out", K.projectWGSL({ hidden: C.hidden, channels: C.convChannels, freq: C.convFreq, time: C.convFrames })),
    mk("layer_norm", K.layerNormWGSL({ hidden: C.hidden, eps: C.layerNormEps })),
    mk("linear", K.linearWGSL({ k: C.hidden, n: C.hidden })),
    mk("window_attention", K.attentionWGSL({ hidden: C.hidden, heads: C.heads, headDim: C.headDim, window })),
    mk("gelu", K.geluWGSL()),
    mk("residual", K.addWGSL()),
    mk("fc1", K.linearWGSL({ k: C.hidden, n: C.intermediate })),
    mk("fc2", K.linearWGSL({ k: C.intermediate, n: C.hidden })),
    mk("proj2", K.linearWGSL({ k: C.hidden, n: C.outputDim })),
  ]);

  let serial: Promise<unknown> = Promise.resolve();

  async function run(mel: Float32Array, T: number): Promise<EncoderResult> {
    if (!Number.isSafeInteger(T) || T <= 0) throw new Error(`T must be a positive integer, got ${T}`);
    if (mel.length < C.mel * T) throw new Error(`mel has ${mel.length} values, expected at least ${C.mel * T}`);
    const chunks = Math.ceil(T / C.chunkFrames);
    const frames = encoderFrames(T, C.chunkFrames, C.convFrames);
    const temp: GPUBuffer[] = [];
    const buf = (n: number, usage = S | CD) => { const b = device.createBuffer({ size: align4(n * 4), usage }); temp.push(b); return b; };
    const uni = (values: number[]) => {
      const b = buf(Math.max(4, values.length), GPUBufferUsage.UNIFORM | CD);
      device.queue.writeBuffer(b, 0, new Uint32Array(values));
      return b;
    };
    const bg = (p: GPUComputePipeline, bs: GPUBuffer[]) => device.createBindGroup({
      layout: p.getBindGroupLayout(0), entries: bs.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });

    // Mel input is [mel, T]; the graph pads T to whole chunks with zeros and
    // reshapes to [chunk, 1, mel, chunkFrames].
    const melChunks = new Float32Array(chunks * C.mel * C.chunkFrames);
    for (let m = 0; m < C.mel; m++) for (let t = 0; t < T; t++) {
      const c = Math.floor(t / C.chunkFrames);
      melChunks[(c * C.mel + m) * C.chunkFrames + (t % C.chunkFrames)] = mel[m * T + t];
    }
    const input = buf(melChunks.length);
    device.queue.writeBuffer(input, 0, melChunks);

    const output = device.createBuffer({ size: align4(frames * C.outputDim * 4), usage: S | CS | CD });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    // 1D kernels index gid.x + gid.y * num_workgroups.x * 256, so a large
    // element count is split across y to stay under the per-dimension limit.
    const flat = (p: GPUComputePipeline, bs: GPUBuffer[], n: number) => {
      const groups = Math.ceil(n / 256);
      const x = Math.min(groups, MAX_GROUPS);
      pass.setPipeline(p); pass.setBindGroup(0, bg(p, bs)); pass.dispatchWorkgroups(x, Math.ceil(groups / x));
    };
    const rows = (p: GPUComputePipeline, bs: GPUBuffer[], x: number, y = 1) => {
      pass.setPipeline(p); pass.setBindGroup(0, bg(p, bs)); pass.dispatchWorkgroups(x, y);
    };

    let shape = [1, C.mel, C.chunkFrames];
    let previous = input;
    for (const r of R.conv) {
      const out = [C.convChannels, half(shape[1]), half(shape[2])];
      const n = chunks * out[0] * out[1] * out[2];
      const y = buf(n);
      flat(conv, [previous, weight(r.weight), weight(r.bias), y, uni([chunks, ...shape, ...out, 2])], n);
      previous = y; shape = out;
    }
    if (shape[1] !== C.convFreq || shape[2] !== C.convFrames) throw new Error(`conv output ${shape} disagrees with the encoder config`);

    const H = C.hidden;
    const x0 = buf(frames * H), x1 = buf(frames * H), normed = buf(frames * H), q = buf(frames * H), k = buf(frames * H), v = buf(frames * H), a = buf(frames * H), ffn = buf(frames * Math.max(C.intermediate, H));
    flat(project, [previous, weight(R.convOut), weight(R.positions), x0, uni([frames])], frames * H);

    const fu = uni([frames]);
    const hu = uni([frames * H]);
    const iu = uni([frames * C.intermediate]);
    const groupsH = Math.ceil(H / 64);
    let cur = x0, alt = x1;
    for (const l of R.layers) {
      rows(norm, [cur, weight(l.ln1.weight), weight(l.ln1.bias), normed, fu], frames);
      for (const [w, y] of [[l.q, q], [l.k, k], [l.v, v]] as const) rows(linH, [normed, weight(w.weight), weight(w.bias), y, fu], frames, groupsH);
      const heads = frames * C.heads;
      rows(attnP, [q, k, v, a, fu], Math.min(heads, MAX_GROUPS), Math.ceil(heads / MAX_GROUPS));
      rows(linH, [a, weight(l.o.weight), weight(l.o.bias), normed, fu], frames, groupsH);
      flat(add, [cur, normed, alt, hu], frames * H);
      [cur, alt] = [alt, cur];

      rows(norm, [cur, weight(l.ln2.weight), weight(l.ln2.bias), normed, fu], frames);
      rows(fc1, [normed, weight(l.fc1.weight), weight(l.fc1.bias), ffn, fu], frames, Math.ceil(C.intermediate / 64));
      flat(gelu, [ffn, iu], frames * C.intermediate);
      rows(fc2, [ffn, weight(l.fc2.weight), weight(l.fc2.bias), normed, fu], frames, groupsH);
      flat(add, [cur, normed, alt, hu], frames * H);
      [cur, alt] = [alt, cur];
    }
    rows(norm, [cur, weight(R.lnPost.weight), weight(R.lnPost.bias), normed, fu], frames);
    rows(linH, [normed, weight(R.proj1.weight), weight(R.proj1.bias), ffn, fu], frames, groupsH);
    flat(gelu, [ffn, hu], frames * H);
    rows(proj2, [ffn, weight(R.proj2.weight), weight(R.proj2.bias), output, fu], frames, Math.ceil(C.outputDim / 64));
    pass.end();
    device.queue.submit([enc.finish()]);
    await wait(device.queue.onSubmittedWorkDone());
    for (const b of temp) b.destroy();
    return { buffer: output, frames, dim: C.outputDim };
  }

  function encode(mel: Float32Array, T: number): Promise<EncoderResult> {
    const next = serial.then(() => run(mel, T));
    serial = next.catch(() => undefined);
    return next;
  }
  function destroy() { for (const b of weights.values()) b.destroy(); weights.clear(); }
  return { config: C, encode, destroy };
}

// One stride-2, kernel-3, pad-1 convolution.
const half = (n: number) => (n > 0 ? Math.floor((n - 1) / 2) + 1 : 0);
const convLength = (n: number) => half(half(half(n)));

export type Encoder = Awaited<ReturnType<typeof createEncoder>>;
