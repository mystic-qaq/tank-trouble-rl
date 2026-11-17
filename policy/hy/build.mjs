/* Generates the alternate entry points from policy.js.
   Run: node policy/hy/build.mjs
   Only files inside policy/hy are touched. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const src = fs.readFileSync(path.join(dir, 'policy.js'), 'utf8');

const esm = src + [
  '',
  'export default HY_POLICY_API;',
  'export const decide = HY_POLICY_API;',
  'export const act = HY_POLICY_API;',
  'export const policy = HY_POLICY_API;',
  'export const updateBot = HY_POLICY_API;',
  'export const name = "hy";',
  ''
].join('\n');

const targets = [
  ['policy.mjs', esm],
  ['index.js', src],
  ['index.mjs', esm],
  ['hy_policy.js', src],
  ['hy.mjs', esm]
];

for (const [file, content] of targets) {
  fs.writeFileSync(path.join(dir, file), content);
  console.log('wrote', file, content.length, 'bytes');
}
