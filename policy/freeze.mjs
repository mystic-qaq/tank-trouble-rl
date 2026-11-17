/* ===========================================================
   freeze.mjs — 冻结 policy/ 下的 7 份策略
   -----------------------------------------------------------
   对每个策略文件计算 SHA-256，写入 policy/manifest.sha256.json。
   之后任何人改了策略，都能用 verify-frozen.mjs 立刻发现。

   用法：
     node policy/freeze.mjs          # 生成/更新清单
     node policy/freeze.mjs --check  # 只校验，不写入（漂移则退出码 1）
   =========================================================== */

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MANIFEST = path.join(HERE, 'manifest.sha256.json');

/* 7 个参赛策略。rl/ 是空目录（未实现），不参与冻结。 */
export const STRATEGIES = [
  { id: 'deepseek', dir: 'deepseek', entry: 'policy/deepseek/policy.js', label: 'DeepSeek' },
  { id: 'glm', dir: 'glm', entry: 'policy/glm/glm_policy.js', label: 'GLM' },
  { id: 'gpt', dir: 'gpt', entry: 'policy/gpt/ai.js', label: 'GPT Sentinel' },
  { id: 'hy', dir: 'hy', entry: 'policy/hy/policy.js', label: 'HY' },
  { id: 'kimi', dir: 'kimi', entry: 'policy/kimi/policy.mjs', label: 'Kimi' },
  { id: 'minimax', dir: 'minimax', entry: 'policy/minimax/policy.js', label: 'minimax' },
  { id: 'qwen', dir: 'qwen', entry: 'policy/qwen/policy.js', label: 'Qwen' },
];

function walk(dir, base = dir, out = []) {
  for (const name of fs.readdirSync(dir).sort()) {
    const full = path.join(dir, name);
    const st = fs.statSync(full);
    if (st.isDirectory()) walk(full, base, out);
    else out.push(path.relative(base, full).split(path.sep).join('/'));
  }
  return out;
}

function sha256(file) {
  return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

export function buildManifest() {
  const strategies = {};
  for (const s of STRATEGIES) {
    const dir = path.join(HERE, s.dir);
    if (!fs.existsSync(dir)) continue;
    const files = {};
    for (const rel of walk(dir)) {
      const full = path.join(dir, rel);
      const st = fs.statSync(full);
      files[rel] = { sha256: sha256(full), bytes: st.size };
    }
    strategies[s.id] = {
      label: s.label,
      entry: s.entry,
      fileCount: Object.keys(files).length,
      files,
    };
  }
  return {
    frozenAt: new Date().toISOString(),
    note: '7 份参赛策略的 SHA-256 冻结清单。用 node policy/verify-frozen.mjs 校验是否漂移。',
    strategies,
  };
}

export function verifyManifest(manifest) {
  const problems = [];
  for (const [id, s] of Object.entries(manifest.strategies)) {
    const dir = path.join(HERE, id);
    for (const [rel, meta] of Object.entries(s.files)) {
      const full = path.join(dir, rel);
      if (!fs.existsSync(full)) { problems.push(`${id}/${rel}: 文件丢失`); continue; }
      const now = sha256(full);
      if (now !== meta.sha256) problems.push(`${id}/${rel}: 已修改 (${meta.sha256.slice(0, 12)} → ${now.slice(0, 12)})`);
    }
  }
  return problems;
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  const check = process.argv.includes('--check');
  if (check) {
    if (!fs.existsSync(MANIFEST)) { console.error('缺少清单，先运行 node policy/freeze.mjs'); process.exit(1); }
    const manifest = JSON.parse(fs.readFileSync(MANIFEST, 'utf8'));
    const problems = verifyManifest(manifest);
    if (problems.length) {
      console.error('✗ 策略已漂移：');
      for (const p of problems) console.error('  - ' + p);
      process.exit(1);
    }
    console.log('✓ 7 份策略全部与冻结清单一致，未被修改。');
  } else {
    const manifest = buildManifest();
    fs.writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2));
    const total = Object.values(manifest.strategies).reduce((a, s) => a + s.fileCount, 0);
    console.log(`✓ 已冻结 ${Object.keys(manifest.strategies).length} 个策略、共 ${total} 个文件 → policy/manifest.sha256.json`);
    console.log(`  冻结时间：${manifest.frozenAt}`);
  }
}
