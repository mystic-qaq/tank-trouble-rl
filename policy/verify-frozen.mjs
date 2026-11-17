#!/usr/bin/env node
/* ===========================================================
   verify-frozen.mjs — 校验 7 份策略是否被改动过
   -----------------------------------------------------------
   竞技场开跑前跑一次：任何一份策略被改过就退出码 1，
   保证战报里的数字对应的确实是冻结时的那一版代码。

   用法：node policy/verify-frozen.mjs
   =========================================================== */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MANIFEST = path.join(HERE, 'manifest.sha256.json');

if (!fs.existsSync(MANIFEST)) {
  console.error('✗ 缺少 policy/manifest.sha256.json —— 策略尚未冻结，先跑 node policy/freeze.mjs');
  process.exit(1);
}

const manifest = JSON.parse(fs.readFileSync(MANIFEST, 'utf8'));

/* 复用 freeze.mjs 的校验逻辑，避免两份实现走样 */
const { verifyManifest } = await import('./freeze.mjs');
const problems = verifyManifest(manifest);

if (problems.length) {
  console.error('✗ 检测到策略漂移（冻结后被修改）：');
  for (const p of problems) console.error('  - ' + p);
  console.error('\n若这是有意改动，请重新冻结：node policy/freeze.mjs');
  process.exit(1);
}

const n = Object.values(manifest.strategies).reduce((a, s) => a + s.fileCount, 0);
console.log(`✓ 策略冻结校验通过：${Object.keys(manifest.strategies).length} 个策略 / ${n} 个文件，与 ${manifest.frozenAt} 的清单完全一致。`);
