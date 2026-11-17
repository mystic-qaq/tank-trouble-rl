/* ===========================================================
   生成 classic script 版：policy.global.js
   ---------------------------------------------------------------
   Arena 如果用 <script src="..."> 直接加载（非 module），
   就用这个版本：它把接口挂到 window.TTPolicy 上。
   源码只有 policy.js 一份，本脚本负责机械转换，避免两份代码走样。
   用法：node build-global.mjs
   =========================================================== */

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(HERE, 'policy.js'), 'utf8');

let out = src;

out = out.replace(/^export function /gm, 'function ');
out = out.replace(/^export const /gm, 'const ');
out = out.replace(/^export let /gm, 'let ');
out = out.replace(/^export default .*$/gm, '');
out = out.replace(/^export \{[^}]*\};?\s*$/gm, '');

/* 头部说明改成 classic 版 */
out = out.replace(
  '   对外接口（几种写法都支持）：',
  '   ★ 这是 classic script 版（无 import/export），用 <script src="policy.global.js"> 加载：\n' +
  '     window.TTPolicy.updateBot(tank, game, dt)\n\n' +
  '   对外接口（几种写法都支持）：'
);

/* 结尾补上 window 导出 */
out += `
/* ---------------- classic script 版：挂到 window 上 ---------------- */
if (typeof window !== 'undefined') {
  window.TTPolicy = {
    updateBot: updateBot,
    decide: decidePolicy,
    createPolicy: createPolicy,
    configure: configure,
    resetConfig: resetConfig,
    policy: updateBot,
  };
}
`;

writeFileSync(path.join(HERE, 'policy.global.js'), out, 'utf8');
console.log('已生成 policy.global.js（' + out.split('\n').length + ' 行）');
