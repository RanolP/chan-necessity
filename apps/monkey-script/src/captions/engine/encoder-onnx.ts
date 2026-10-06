// Builds the encoder manifest straight from encoder.fp16.onnx, so byte ranges
// always describe the file actually cached. The torch.onnx export names most
// MatMul weights "val_<n>", so a linear layer's role is read off the graph:
// the MatMul whose output feeds the Add that consumes "<module>.bias".
import type { EncoderManifest, EncoderTensor } from "./encoder.ts";

interface Field { field: number; wire: number; start: number; end: number }

function varint(a: Uint8Array, p: number, end: number): [number, number] {
  let n = 0;
  for (let shift = 0; p < end && shift < 70; shift += 7) {
    const b = a[p++];
    n += (b & 0x7f) * 2 ** shift;
    if (!(b & 0x80)) return [n, p];
  }
  throw new Error(`encoder.fp16.onnx: bad varint before byte ${p}`);
}

function* fields(a: Uint8Array, start: number, end: number): Generator<Field> {
  let p = start;
  while (p < end) {
    const [key, q] = varint(a, p, end);
    const field = Math.floor(key / 8), wire = key & 7;
    p = q;
    let s = p;
    if (wire === 0) p = varint(a, p, end)[1];
    else if (wire === 1) p += 8;
    else if (wire === 5) p += 4;
    else if (wire === 2) { const [n, r] = varint(a, p, end); s = r; p = r + n; }
    else throw new Error(`encoder.fp16.onnx: protobuf wire type ${wire} at byte ${p}`);
    if (p > end) throw new Error(`encoder.fp16.onnx: field ${field} overruns its message at byte ${s}`);
    yield { field, wire, start: s, end: p };
  }
}

const text = (a: Uint8Array, f: Field) => new TextDecoder().decode(a.subarray(f.start, f.end));
const DTYPE: Record<number, string> = { 1: "float32", 7: "int64", 10: "float16" };

export function parseEncoderManifest(bytes: Uint8Array): EncoderManifest {
  let graph: Field | undefined;
  for (const f of fields(bytes, 0, bytes.byteLength)) if (f.field === 7 && f.wire === 2) { graph = f; break; }
  if (!graph) throw new Error("encoder.fp16.onnx: ModelProto has no graph");

  const tensors: Record<string, EncoderTensor> = {};
  const producer = new Map<string, { op: string; inputs: string[] }>();
  const adds: string[][] = [];
  const matmuls: string[][] = [];
  for (const f of fields(bytes, graph.start, graph.end)) {
    if (f.wire !== 2) continue;
    if (f.field === 5) {
      let name = "", dtype = 0, raw: Field | undefined;
      const shape: number[] = [];
      for (const t of fields(bytes, f.start, f.end)) {
        if (t.field === 1 && t.wire === 0) shape.push(varint(bytes, t.start, t.end)[0]);
        else if (t.field === 1 && t.wire === 2) for (let p = t.start; p < t.end;) { const [v, q] = varint(bytes, p, t.end); shape.push(v); p = q; }
        else if (t.field === 2 && t.wire === 0) dtype = varint(bytes, t.start, t.end)[0];
        else if (t.field === 8 && t.wire === 2) name = text(bytes, t);
        else if (t.field === 9 && t.wire === 2) raw = t;
      }
      if (raw) tensors[name] = { offset: raw.start, length: raw.end - raw.start, dtype: DTYPE[dtype] ?? `onnx:${dtype}`, shape };
    } else if (f.field === 1) {
      let op = "";
      const inputs: string[] = [], outputs: string[] = [];
      for (const n of fields(bytes, f.start, f.end)) {
        if (n.wire !== 2) continue;
        if (n.field === 1) inputs.push(text(bytes, n));
        else if (n.field === 2) outputs.push(text(bytes, n));
        else if (n.field === 4) op = text(bytes, n);
      }
      for (const o of outputs) producer.set(o, { op, inputs });
      if (op === "Add") adds.push(inputs);
      if (op === "MatMul") matmuls.push(inputs);
    }
  }

  const need = (name: string) => {
    if (!tensors[name]) throw new Error(`encoder.fp16.onnx: initializer ${name} is missing or not inline`);
    return name;
  };
  // "<module>.bias" -> the [in, out] weight of the MatMul feeding that Add.
  const linearWeight = new Map<string, string>();
  for (const inputs of adds) {
    const bias = inputs.find((x) => x.endsWith(".bias") && tensors[x]);
    const other = inputs.find((x) => x !== bias);
    const mm = other ? producer.get(other) : undefined;
    if (bias && mm?.op === "MatMul" && tensors[mm.inputs[1]]) linearWeight.set(bias.slice(0, -5), mm.inputs[1]);
  }
  const linear = (module: string) => {
    const weight = linearWeight.get(module);
    if (!weight) throw new Error(`encoder.fp16.onnx: no MatMul feeds ${module}.bias`);
    return { weight, bias: need(module + ".bias") };
  };
  const norm = (module: string) => ({ weight: need(module + ".weight"), bias: need(module + ".bias") });

  const conv = [1, 2, 3].map((i) => norm(`conv2d${i}`));
  const convChannels = tensors[conv[0].weight].shape[0];
  const convFreq = 16;
  // conv_out has no bias, so it is the one MatMul weight shaped [channels * freq, hidden].
  const convOut = matmuls.map((x) => x[1]).find((w) => tensors[w]?.shape.length === 2 && tensors[w].shape[0] === convChannels * convFreq);
  if (!convOut) throw new Error(`encoder.fp16.onnx: no conv_out MatMul of input width ${convChannels * convFreq}`);

  const layerCount = Object.keys(tensors).filter((n) => /^layers\.\d+\.fc1\.bias$/.test(n)).length;
  const layers = Array.from({ length: layerCount }, (_, index) => {
    const p = `layers.${index}.`;
    return {
      index,
      q: linear(p + "self_attn.q_proj"), k: linear(p + "self_attn.k_proj"), v: linear(p + "self_attn.v_proj"), o: linear(p + "self_attn.out_proj"),
      ln1: norm(p + "self_attn_layer_norm"), fc1: linear(p + "fc1"), fc2: linear(p + "fc2"), ln2: norm(p + "final_layer_norm"),
    };
  });
  const roles = {
    conv, convOut,
    positions: need("positional_embedding.positional_embedding"),
    lnPost: norm("ln_post"), proj1: linear("proj1"), proj2: linear("proj2"),
    layers,
  };
  const hidden = tensors[layers[0].q.weight].shape[1];
  const heads = 16;
  return {
    schema: 2,
    source: "encoder.fp16.onnx",
    config: {
      mel: 128, chunkFrames: 100, attentionWindowFrames: 800, convFrames: 13, convChannels, convFreq,
      hidden, intermediate: tensors[layers[0].fc1.weight].shape[1], heads, headDim: hidden / heads,
      layers: layerCount, outputDim: tensors[roles.proj2.weight].shape[1], layerNormEps: 1e-5,
    },
    roles,
    tensors,
  };
}
