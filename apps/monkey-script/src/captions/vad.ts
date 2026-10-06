/**
 * The 16 kHz Silero VAD v5 streaming path without an inference runtime.
 *
 * The ONNX file is used as a weight container. `parseSileroVadWeights` reads
 * its GraphProto initializer messages and `createVad` evaluates the fixed
 * 576-sample path (64 samples of context plus 512 new samples).
 */

const FRAME_SAMPLES = 576;
const STFT_FRAMES = 4;
const FFT_SIZE = 256;
const HOP_SIZE = 128;
const FFT_BINS = 129;
const HIDDEN_SIZE = 128;
const GATE_SIZE = HIDDEN_SIZE * 4;

export const SILERO_VAD_V5_URL = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.24/dist/silero_vad_v5.onnx";

export interface SileroVadWeights {
    readonly stftBasis: Float32Array;
    readonly encoder0Weight: Float32Array;
    readonly encoder0Bias: Float32Array;
    readonly encoder1Weight: Float32Array;
    readonly encoder1Bias: Float32Array;
    readonly encoder2Weight: Float32Array;
    readonly encoder2Bias: Float32Array;
    readonly encoder3Weight: Float32Array;
    readonly encoder3Bias: Float32Array;
    readonly rnnWeightInput: Float32Array;
    readonly rnnWeightHidden: Float32Array;
    readonly rnnBiasInput: Float32Array;
    readonly rnnBiasHidden: Float32Array;
    readonly decoderWeight: Float32Array;
    readonly decoderBias: Float32Array;
}

interface TensorInitializer {
    readonly name: string;
    readonly dataType: number;
    readonly dims: number[];
    readonly rawData?: Uint8Array;
    readonly floatData: number[];
}

class ProtoReader {
    private offset = 0;
    private readonly bytes: Uint8Array;
    private readonly end: number;

    constructor(bytes: Uint8Array, end = bytes.length) {
        this.bytes = bytes;
        this.end = end;
    }

    get done(): boolean {
        return this.offset >= this.end;
    }

    field(): [number, number] {
        const key = this.varint();
        return [Math.floor(key / 8), key & 7];
    }

    varint(): number {
        let value = 0;
        let shift = 0;
        for (;;) {
            if (this.offset >= this.end || shift > 49) throw new Error("Invalid ONNX protobuf varint");
            const byte = this.bytes[this.offset++];
            value += (byte & 0x7f) * 2 ** shift;
            if ((byte & 0x80) === 0) return value;
            shift += 7;
        }
    }

    bytesField(): Uint8Array {
        const length = this.varint();
        const start = this.offset;
        const end = start + length;
        if (!Number.isSafeInteger(length) || end > this.end) throw new Error("Invalid ONNX protobuf length");
        this.offset = end;
        return this.bytes.subarray(start, end);
    }

    fixed32(): number {
        if (this.offset + 4 > this.end) throw new Error("Invalid ONNX protobuf fixed32");
        const value = new DataView(this.bytes.buffer, this.bytes.byteOffset + this.offset, 4).getFloat32(0, true);
        this.offset += 4;
        return value;
    }

    skip(wireType: number): void {
        switch (wireType) {
            case 0:
                this.varint();
                return;
            case 1:
                this.offset += 8;
                return;
            case 2:
                {
                    const length = this.varint();
                    this.offset += length;
                }
                return;
            case 5:
                this.offset += 4;
                return;
            default:
                throw new Error(`Unsupported ONNX protobuf wire type ${wireType}`);
        }
    }
}

function parseTensorInitializer(bytes: Uint8Array): TensorInitializer {
    const reader = new ProtoReader(bytes);
    let name = "";
    let dataType = 0;
    const dims: number[] = [];
    let rawData: Uint8Array | undefined;
    const floatData: number[] = [];

    while (!reader.done) {
        const [field, wireType] = reader.field();
        if (field === 1 && wireType === 0) dims.push(reader.varint());
        else if (field === 1 && wireType === 2) {
            const packedDims = new ProtoReader(reader.bytesField());
            while (!packedDims.done) dims.push(packedDims.varint());
        }
        else if (field === 2 && wireType === 0) dataType = reader.varint();
        else if (field === 4 && wireType === 5) {
            floatData.push(reader.fixed32());
        } else if (field === 4 && wireType === 2) {
            const packed = reader.bytesField();
            if (packed.byteLength % 4 !== 0) throw new Error(`Invalid float_data for ONNX initializer ${name}`);
            const view = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
            for (let i = 0; i < packed.byteLength; i += 4) floatData.push(view.getFloat32(i, true));
        } else if (field === 8 && wireType === 2) {
            name = new TextDecoder().decode(reader.bytesField());
        } else if (field === 9 && wireType === 2) {
            rawData = reader.bytesField();
        } else {
            reader.skip(wireType);
        }
    }
    return { name, dataType, dims, rawData, floatData };
}

// The v5 export branches on the sample rate (If), and each branch holds its
// weights as Constant nodes named "If_0_then_branch__Inline_0__<module path>"
// rather than as graph initializers, so every tensor in the model and its
// subgraphs is collected.
function readTensors(model: Uint8Array): TensorInitializer[] {
    const out: TensorInitializer[] = [];
    const modelReader = new ProtoReader(model);
    while (!modelReader.done) {
        const [field, wireType] = modelReader.field();
        if (field === 7 && wireType === 2) readGraph(modelReader.bytesField(), out);
        else modelReader.skip(wireType);
    }
    return out;
}

function readGraph(bytes: Uint8Array, out: TensorInitializer[]): void {
    const reader = new ProtoReader(bytes);
    while (!reader.done) {
        const [field, wireType] = reader.field();
        if (field === 5 && wireType === 2) out.push(parseTensorInitializer(reader.bytesField()));
        else if (field === 1 && wireType === 2) readNode(reader.bytesField(), out);
        else reader.skip(wireType);
    }
}

function readNode(bytes: Uint8Array, out: TensorInitializer[]): void {
    const reader = new ProtoReader(bytes);
    let output = "";
    let op = "";
    const attrs: Uint8Array[] = [];
    while (!reader.done) {
        const [field, wireType] = reader.field();
        if (field === 2 && wireType === 2 && !output) output = new TextDecoder().decode(reader.bytesField());
        else if (field === 4 && wireType === 2) op = new TextDecoder().decode(reader.bytesField());
        else if (field === 5 && wireType === 2) attrs.push(reader.bytesField());
        else reader.skip(wireType);
    }
    for (const attr of attrs) {
        const a = new ProtoReader(attr);
        while (!a.done) {
            const [field, wireType] = a.field();
            if (field === 5 && wireType === 2 && op === "Constant") out.push({ ...parseTensorInitializer(a.bytesField()), name: output });
            else if (field === 6 && wireType === 2) readGraph(a.bytesField(), out);
            else a.skip(wireType);
        }
    }
}

// Matched by module-path suffix and element count, preferring the If's
// then_branch: the 16 kHz (then) and 8 kHz (else) branches hold same-named
// tensors, several of them the same size.
function tensorFloat(tensors: readonly TensorInitializer[], name: string, size: number): Float32Array {
    const matches = tensors.filter((t) => (t.name === name || t.name.endsWith("__" + name) || t.name.endsWith("." + name)) && t.dims.reduce((product, dim) => product * dim, 1) === size);
    const tensor = matches.find((t) => t.name.includes("then_branch")) ?? matches[0];
    if (!tensor) throw new Error(`Silero VAD tensor ${name} with ${size} values is missing`);
    if (tensor.dataType !== 1) throw new Error(`Silero VAD initializer ${name} is not float32`);

    const result = new Float32Array(size);
    if (tensor.rawData) {
        if (tensor.rawData.byteLength !== size * 4) throw new Error(`Silero VAD initializer ${name} has invalid raw_data`);
        const view = new DataView(tensor.rawData.buffer, tensor.rawData.byteOffset, tensor.rawData.byteLength);
        for (let i = 0; i < size; i++) result[i] = view.getFloat32(i * 4, true);
    } else if (tensor.floatData.length === size) {
        result.set(tensor.floatData);
    } else {
        throw new Error(`Silero VAD initializer ${name} has no float data`);
    }
    return result;
}

export function parseSileroVadWeights(model: ArrayBuffer | Uint8Array): SileroVadWeights {
    const bytes = model instanceof Uint8Array ? model : new Uint8Array(model);
    const tensors = readTensors(bytes);
    const tensor = (name: string, size: number) => tensorFloat(tensors, name, size);
    return {
        stftBasis: tensor("stft.forward_basis_buffer", 258 * 256),
        encoder0Weight: tensor("encoder.0.reparam_conv.weight", 128 * 129 * 3),
        encoder0Bias: tensor("encoder.0.reparam_conv.bias", 128),
        encoder1Weight: tensor("encoder.1.reparam_conv.weight", 64 * 128 * 3),
        encoder1Bias: tensor("encoder.1.reparam_conv.bias", 64),
        encoder2Weight: tensor("encoder.2.reparam_conv.weight", 64 * 64 * 3),
        encoder2Bias: tensor("encoder.2.reparam_conv.bias", 64),
        encoder3Weight: tensor("encoder.3.reparam_conv.weight", 128 * 64 * 3),
        encoder3Bias: tensor("encoder.3.reparam_conv.bias", 128),
        rnnWeightInput: tensor("decoder.rnn.weight_ih", GATE_SIZE * HIDDEN_SIZE),
        rnnWeightHidden: tensor("decoder.rnn.weight_hh", GATE_SIZE * HIDDEN_SIZE),
        rnnBiasInput: tensor("decoder.rnn.bias_ih", GATE_SIZE),
        rnnBiasHidden: tensor("decoder.rnn.bias_hh", GATE_SIZE),
        decoderWeight: tensor("decoder.decoder.2.weight", HIDDEN_SIZE),
        decoderBias: tensor("decoder.decoder.2.bias", 1),
    };
}

export async function loadSileroVadWeights(url = SILERO_VAD_V5_URL): Promise<SileroVadWeights> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Silero VAD download failed: ${response.status} ${response.statusText}`);
    return parseSileroVadWeights(await response.arrayBuffer());
}

function sigmoid(value: number): number {
    return 1 / (1 + Math.exp(-value));
}

function convRelu(
    input: Float32Array,
    inputChannels: number,
    inputLength: number,
    output: Float32Array,
    outputChannels: number,
    outputLength: number,
    weights: Float32Array,
    bias: Float32Array,
    stride: number,
): void {
    for (let channel = 0; channel < outputChannels; channel++) {
        const weightBase = channel * inputChannels * 3;
        const outputBase = channel * outputLength;
        for (let time = 0; time < outputLength; time++) {
            let value = bias[channel];
            const inputBase = time * stride - 1;
            for (let inputChannel = 0; inputChannel < inputChannels; inputChannel++) {
                const weightOffset = weightBase + inputChannel * 3;
                for (let kernel = 0; kernel < 3; kernel++) {
                    const inputTime = inputBase + kernel;
                    if (inputTime >= 0 && inputTime < inputLength) {
                        value += weights[weightOffset + kernel] * input[inputChannel * inputLength + inputTime];
                    }
                }
            }
            output[outputBase + time] = value > 0 ? value : 0;
        }
    }
}

export interface SileroVad {
    reset(): void;
    prob(frame576: Float32Array): number;
}

export function createVad(weights: SileroVadWeights): SileroVad {
    const spectrum = new Float32Array(FFT_BINS * STFT_FRAMES);
    const encoder0 = new Float32Array(128 * STFT_FRAMES);
    const encoder1 = new Float32Array(64 * 2);
    const encoder2 = new Float32Array(64);
    const encoder3 = new Float32Array(HIDDEN_SIZE);
    const gates = new Float32Array(GATE_SIZE);
    const hidden = new Float32Array(HIDDEN_SIZE);
    const cell = new Float32Array(HIDDEN_SIZE);

    const reset = () => {
        hidden.fill(0);
        cell.fill(0);
    };

    const prob = (frame576: Float32Array): number => {
        if (frame576.length !== FRAME_SAMPLES) throw new RangeError(`Silero VAD expects ${FRAME_SAMPLES} samples`);

        for (let frame = 0; frame < STFT_FRAMES; frame++) {
            const inputOffset = frame * HOP_SIZE;
            const basisOffset = 0;
            for (let bin = 0; bin < FFT_BINS; bin++) {
                let real = 0;
                let imag = 0;
                const realBasisOffset = (basisOffset + bin) * FFT_SIZE;
                const imagBasisOffset = (basisOffset + FFT_BINS + bin) * FFT_SIZE;
                for (let sample = 0; sample < FFT_SIZE; sample++) {
                    const inputIndex = inputOffset + sample;
                    const value = inputIndex < FRAME_SAMPLES ? frame576[inputIndex] : frame576[2 * FRAME_SAMPLES - 2 - inputIndex];
                    real += weights.stftBasis[realBasisOffset + sample] * value;
                    imag += weights.stftBasis[imagBasisOffset + sample] * value;
                }
                spectrum[bin * STFT_FRAMES + frame] = Math.sqrt(real * real + imag * imag);
            }
        }

        convRelu(spectrum, FFT_BINS, STFT_FRAMES, encoder0, 128, STFT_FRAMES, weights.encoder0Weight, weights.encoder0Bias, 1);
        convRelu(encoder0, 128, STFT_FRAMES, encoder1, 64, 2, weights.encoder1Weight, weights.encoder1Bias, 2);
        convRelu(encoder1, 64, 2, encoder2, 64, 1, weights.encoder2Weight, weights.encoder2Bias, 2);
        convRelu(encoder2, 64, 1, encoder3, HIDDEN_SIZE, 1, weights.encoder3Weight, weights.encoder3Bias, 1);

        const input = encoder3;
        const inputWeights = weights.rnnWeightInput;
        const hiddenWeights = weights.rnnWeightHidden;
        const inputBias = weights.rnnBiasInput;
        const hiddenBias = weights.rnnBiasHidden;
        for (let gateIndex = 0; gateIndex < HIDDEN_SIZE; gateIndex++) {
            const inputOffset = gateIndex * HIDDEN_SIZE;
            const outputOffset = 384 * HIDDEN_SIZE + inputOffset;
            let inputGate = inputBias[gateIndex] + hiddenBias[gateIndex];
            let outputGate = inputBias[384 + gateIndex] + hiddenBias[384 + gateIndex];
            let forgetGate = inputBias[128 + gateIndex] + hiddenBias[128 + gateIndex];
            let cellGate = inputBias[256 + gateIndex] + hiddenBias[256 + gateIndex];
            for (let inputIndex = 0; inputIndex < HIDDEN_SIZE; inputIndex++) {
                inputGate += inputWeights[inputOffset + inputIndex] * input[inputIndex] + hiddenWeights[inputOffset + inputIndex] * hidden[inputIndex];
                outputGate += inputWeights[outputOffset + inputIndex] * input[inputIndex] + hiddenWeights[outputOffset + inputIndex] * hidden[inputIndex];
                forgetGate += inputWeights[128 * HIDDEN_SIZE + inputOffset + inputIndex] * input[inputIndex] + hiddenWeights[128 * HIDDEN_SIZE + inputOffset + inputIndex] * hidden[inputIndex];
                cellGate += inputWeights[256 * HIDDEN_SIZE + inputOffset + inputIndex] * input[inputIndex] + hiddenWeights[256 * HIDDEN_SIZE + inputOffset + inputIndex] * hidden[inputIndex];
            }
            gates[gateIndex] = sigmoid(inputGate);
            gates[HIDDEN_SIZE + gateIndex] = sigmoid(outputGate);
            gates[2 * HIDDEN_SIZE + gateIndex] = sigmoid(forgetGate);
            gates[3 * HIDDEN_SIZE + gateIndex] = Math.tanh(cellGate);
        }

        let probabilityLogit = weights.decoderBias[0];
        for (let i = 0; i < HIDDEN_SIZE; i++) {
            const nextCell = gates[2 * HIDDEN_SIZE + i] * cell[i] + gates[i] * gates[3 * HIDDEN_SIZE + i];
            cell[i] = nextCell;
            hidden[i] = gates[HIDDEN_SIZE + i] * Math.tanh(nextCell);
            const reluHidden = hidden[i] > 0 ? hidden[i] : 0;
            probabilityLogit += weights.decoderWeight[i] * reluHidden;
        }
        return sigmoid(probabilityLogit);
    };

    reset();
    return { reset, prob };
}
