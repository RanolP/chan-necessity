// The picture's on-screen box: a <video> letterboxes its frame inside its
// own element box (object-fit: contain by default in the player), so the
// element box alone overstates it. Anything drawn around or over the
// picture positions from here.
export function videoRect(video: HTMLVideoElement): DOMRect {
    const box = video.getBoundingClientRect();
    const { videoWidth: w, videoHeight: h } = video;
    const fit = getComputedStyle(video).objectFit;
    if (!w || !h || !box.width || !box.height || (fit !== "contain" && fit !== "scale-down")) return box;
    let scale = Math.min(box.width / w, box.height / h);
    if (fit === "scale-down") scale = Math.min(scale, 1);
    const width = w * scale;
    const height = h * scale;
    return new DOMRect(box.x + (box.width - width) / 2, box.y + (box.height - height) / 2, width, height);
}
