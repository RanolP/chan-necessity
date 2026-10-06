// The site serves the userscript at its root, where @updateURL and
// @downloadURL point. It is a build product of apps/monkey-script, so it is
// copied into public/ (gitignored) before every build instead of committed.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const src = fileURLToPath(new URL("../../monkey-script/dist/chan-necessity.user.js", import.meta.url));
const dir = fileURLToPath(new URL("../public/", import.meta.url));
if (!existsSync(src)) {
    console.error(`${src} is missing; build apps/monkey-script first (the root "pnpm build" does both in order).`);
    process.exit(1);
}
mkdirSync(dir, { recursive: true });
copyFileSync(src, dir + "chan-necessity.user.js");
console.log(`copied ${src} → ${dir}chan-necessity.user.js`);
