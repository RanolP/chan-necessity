// Catches: the singleton guard touching the DOM again, which threw
// "Cannot read properties of null (reading 'dataset')" at @run-at
// document-start and kept the userscript from ever starting.
import assert from "node:assert/strict";
import { claimInstance } from "../src/shared/instance-guard.ts";

assert.equal(typeof document, "undefined", "check must run with no DOM, like document-start");
const page = {};
assert.equal(claimInstance(page, "0.5.0"), undefined, "first instance must start");
assert.equal(claimInstance(page, "0.6.0-dev"), "0.5.0", "second instance must see the first");
console.log("instance-guard: ok");
