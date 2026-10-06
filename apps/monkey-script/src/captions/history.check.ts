// Run: node src/captions/history.check.ts
// Catches a VOD stamp losing its hour or zero padding (1:02:03 shown as 62:03 or 1:2:3).
import { formatStamp } from "./history.ts";

const wall = new Date(2026, 0, 1, 9, 5, 7).getTime();
const at = (media: number | null) => ({ wall, media, open: null, text: "" });
const eq = (got: string, want: string) => {
    if (got !== want) throw new Error(`formatStamp: got ${got}, want ${want}`);
};
eq(formatStamp(at(3723.9), false), "1:02:03");
eq(formatStamp(at(65), false), "1:05");
eq(formatStamp(at(null), false), "--:--");
eq(formatStamp(at(null), true), "09:05:07");

// Catches live stamps reverting to the wall clock, losing the hour, or going negative when the watched moment precedes the open.
const since = (sec: number) => ({ wall, media: null, open: wall - sec * 1000, text: "" });
eq(formatStamp(since(65.9), true), "1:05");
eq(formatStamp(since(3723), true), "1:02:03");
eq(formatStamp(since(-30), true), "0:00");
console.log("history.check ok");
