/* ===========================================================
   build-vendor.mjs — 为浏览器页生成 ESM 策略的 classic script 版
   -----------------------------------------------------------
   为什么需要：
     浏览器里用 `import()` 动态加载 ES module，在 `file://` 协议下会被
     Chrome 拦掉（"Failed to fetch dynamically imported module"），
     双击打开 arena/index.html 就载入不了 kimi / minimax。

   做法：
     把 ESM 源码做**机械转换**（去掉 export 关键字、尾部挂到 window），
     输出到 arena/vendor/。policy/ 目录一个字节都不动。

   一致性保证：
     minimax 自带 policy/minimax/policy.global.js，它的生成逻辑就是
     policy/minimax/build-global.mjs。本脚本用同样的转换生成 minimax 的
     vendor 版，并与官方产物逐字节比对，确保转换逻辑没有走样。

   用法：node arena/build-vendor.mjs
   =========================================================== */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const OUT = path.join(HERE, 'vendor');

/* ESM → classic script 的机械转换（与 minimax/build-global.mjs 同一套规则） */
function esmToClassic(src, globalName, exports) {
  let out = src;
  out = out.replace(/^export function /gm, 'function ');
  out = out.replace(/^export const /gm, 'const ');
  out = out.replace(/^export let /gm, 'let ');
  out = out.replace(/^export default .*$/gm, '');
  out = out.replace(/^export \{[^}]*\};?\s*$/gm, '');

  /* 用 IIFE 包起来：classic script 的顶层 const/function 会进全局作用域，
     minimax 和 kimi 都声明了 `const TAU`，同时加载会直接抛
     "Identifier 'TAU' has already been declared"。包进 IIFE 后互不干扰。 */
  out = '(function () {\n' + out + `
/* ---------------- classic script 版：挂到 window 上（arena/build-vendor.mjs 生成） ---------------- */
if (typeof window !== 'undefined') {
  window.${globalName} = {
${exports.map((e) => `    ${e.name}: ${e.ref},`).join('\n')}
  };
}
})();
`;
  return out;
}

const TARGETS = [
  {
    id: 'kimi',
    src: 'policy/kimi/policy.mjs',
    out: 'kimi.classic.js',
    globalName: 'KIMI_TANK',
    exports: [{ name: 'update', ref: 'updateBot' }, { name: 'policy', ref: 'updateBot' }],
  },
  {
    id: 'minimax',
    src: 'policy/minimax/policy.js',
    out: 'minimax.classic.js',
    globalName: 'TTPolicy',
    exports: [
      { name: 'updateBot', ref: 'updateBot' },
      { name: 'decide', ref: 'decidePolicy' },
      { name: 'createPolicy', ref: 'createPolicy' },
      { name: 'configure', ref: 'configure' },
      { name: 'resetConfig', ref: 'resetConfig' },
      { name: 'policy', ref: 'updateBot' },
    ],
  },
];

fs.mkdirSync(OUT, { recursive: true });

let ok = true;
for (const t of TARGETS) {
  const src = fs.readFileSync(path.join(ROOT, t.src), 'utf8');
  const out = esmToClassic(src, t.globalName, t.exports);
  fs.writeFileSync(path.join(OUT, t.out), out, 'utf8');
  console.log(`✓ ${t.src} → arena/vendor/${t.out}  (${out.length} bytes)`);
}

/* 交叉校验：minimax 的 vendor 版必须与官方 policy.global.js 的**代码本体**一致。
   （官方版在文件头多插了 3 行说明文字，会造成后续行号整体位移，
     所以比对时要先去掉注释再比。）
   注：vendor 版比官方版多一层 IIFE 包裹（避免全局标识符冲突），
       比对时把包裹层剥掉。 */
const official = fs.readFileSync(path.join(ROOT, 'policy/minimax/policy.global.js'), 'utf8');
const mine = fs.readFileSync(path.join(OUT, 'minimax.classic.js'), 'utf8');

function stripComments(s) {
  return s
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/[^\n]*/g, '$1')
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.length)
    .join('\n');
}
function unwrapIIFE(s) {
  const t = s.trim();
  if (t.startsWith('(function () {')) {
    const i = t.indexOf('\n');
    const body = t.slice(i + 1, t.lastIndexOf('})();'));
    return body;
  }
  return t;
}
const same = stripComments(official) === stripComments(unwrapIIFE(mine));
console.log('\n' + (same
  ? '✓ 交叉校验通过：minimax 的 vendor 版与官方 policy.global.js 代码本体完全一致'
  : '✗ 交叉校验失败：minimax vendor 版与官方产物不一致'));
if (!same) ok = false;

console.log('\n（policy/ 目录未被修改，vendor 版只是浏览器用的机械转换副本）');
if (!ok) process.exit(1);
