import { readFile, readdir, mkdir, copyFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const html = await readFile(resolve(root, "src/feedback-form.html"), "utf8");
if (!html.includes("其他（写下你的真实感受）")) throw new Error("feedback label is missing");
if (html.includes("orthers")) throw new Error("invalid feedback field spelling");
const security = await readFile(resolve(root, "src/feedback-window-security.mjs"), "utf8");
if (!security.includes("connect-src 'none'")) throw new Error("feedback CSP is missing");
console.log("Desktop feedback build checks passed.");
await mkdir(resolve(root, 'dist'), { recursive: true });
// src contains product files only. Enumerate them so a new host module cannot
// pass syntax/tests yet be omitted from the staged application.
const files = (await readdir(resolve(root, 'src'), { withFileTypes: true }))
  .filter(entry => entry.isFile() && /\.(?:mjs|cjs|html|css)$/u.test(entry.name))
  .map(entry => entry.name).sort();
for (const file of files) await copyFile(resolve(root, 'src', file), resolve(root, 'dist', file));
console.log(`Desktop application sources staged: ${files.length} files. Source-run app: npm start.`);
