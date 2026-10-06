// Whisper-compatible log-mel with a mixed-radix FFT for n_fft = 400.
const N = 400;
const HOP = 160;
const COS = new Float64Array(N);
const SIN = new Float64Array(N);
for (let i = 0; i < N; i++) {
    COS[i] = Math.cos((2 * Math.PI * i) / N);
    SIN[i] = Math.sin((2 * Math.PI * i) / N);
}
function fft(re: Float64Array, im: Float64Array, n: number) {
    if (n === 1) return;
    let p = 2;
    while (n % p) p++;
    const m = n / p;
    const subRe: Float64Array[] = [];
    const subIm: Float64Array[] = [];
    for (let r = 0; r < p; r++) {
        const sr = new Float64Array(m);
        const si = new Float64Array(m);
        for (let k = 0; k < m; k++) {
            sr[k] = re[k * p + r];
            si[k] = im[k * p + r];
        }
        fft(sr, si, m);
        subRe.push(sr);
        subIm.push(si);
    }
    const step = N / n;
    for (let k = 0; k < n; k++) {
        let accR = 0;
        let accI = 0;
        const km = k % m;
        for (let r = 0; r < p; r++) {
            const idx = ((r * k) % n) * step;
            const c = COS[idx];
            const s = -SIN[idx];
            const xr = subRe[r][km];
            const xi = subIm[r][km];
            accR += xr * c - xi * s;
            accI += xr * s + xi * c;
        }
        re[k] = accR;
        im[k] = accI;
    }
}
const HANN = new Float64Array(N);
for (let i = 0; i < N; i++) HANN[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
export function logMel(pcm: Float32Array, filters: number[][]) {
    const P = N / 2;
    const len = pcm.length;
    const x = new Float32Array(len + 2 * P);
    x.set(pcm, P);
    for (let i = 0; i < P; i++) {
        x[P - 1 - i] = pcm[i + 1];
        x[P + len + i] = pcm[len - 2 - i];
    }
    const T = Math.floor(len / HOP); // last frame dropped
    const out = new Float32Array(128 * T);
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    const pw = new Float64Array(201);
    let mx = -Infinity;
    for (let t = 0; t < T; t++) {
        for (let n = 0; n < N; n++) {
            re[n] = x[t * HOP + n] * HANN[n];
            im[n] = 0;
        }
        fft(re, im, N);
        for (let k = 0; k < 201; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
        for (let m = 0; m < 128; m++) {
            const f = filters[m];
            let s = 0;
            for (let k = 0; k < 201; k++) s += f[k] * pw[k];
            const v = Math.log10(Math.max(s, 1e-10));
            out[m * T + t] = v;
            if (v > mx) mx = v;
        }
    }
    for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], mx - 8) + 4) / 4;
    return { data: out, T };
}

