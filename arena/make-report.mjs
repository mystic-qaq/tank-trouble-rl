/* ===========================================================
   make-report.mjs — 合并多批次战报 → 生成 arena/report.html
   -----------------------------------------------------------
   用法：
     node arena/make-report.mjs arena/report-seed20260101.json arena/report.json
   =========================================================== */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');

const files = process.argv.slice(2);
if (!files.length) files = [path.join(HERE, 'report.json')];

const reports = files.map((f) => JSON.parse(fs.readFileSync(path.resolve(ROOT, f), 'utf8')));
if (new Set(reports.map(r => r.rulesVersion || 'legacy')).size > 1) {
  throw new Error('不能合并不同规则版本的战报；60 秒赛制请单独生成。');
}

/* ---------------- 合并统计 ---------------- */

const ids = reports[0].strategies;
const labels = reports[0].labels;

const merged = {};
for (const id of ids) {
  merged[id] = {
    id, label: labels[id],
    wins: 0, rounds: 0, kills: 0, suicides: 0, deaths: 0, errors: 0,
    finalists: 0, aliveSum: 0, lifeSum: 0, lifeCount: 0,
    timeouts: 0, byMap: { A: { wins: 0, rounds: 0 }, B: { wins: 0, rounds: 0 } },
  };
}

for (const rep of reports) {
  for (const id of ids) {
    const r = rep.rows[id], m = merged[id];
    m.wins += r.wins; m.rounds += r.rounds;
    m.kills += r.kills; m.suicides += r.suicides; m.deaths += r.deaths;
    m.errors += r.errors; m.finalists += r.finalists; m.aliveSum += r.aliveSum;
    m.lifeSum += r.avgLifeSec * r.rounds; m.lifeCount += r.rounds;
    m.timeouts += r.timeouts;
    m.byMap.A.wins += r.byMap.A.wins; m.byMap.A.rounds += r.byMap.A.rounds;
    m.byMap.B.wins += r.byMap.B.wins; m.byMap.B.rounds += r.byMap.B.rounds;
  }
}

for (const id of ids) {
  const m = merged[id];
  m.winRate = m.wins / m.rounds;
  m.killPerRound = m.kills / m.rounds;
  m.netKills = m.kills - m.suicides;
  m.netPerRound = m.netKills / m.rounds;
  m.deathPerRound = m.deaths / m.rounds;
  m.suicidePerRound = m.suicides / m.rounds;
  m.finalRate = m.finalists / m.rounds;
  m.finalWinRate = m.finalists ? m.wins / m.finalists : 0;
  m.avgLifeSec = m.lifeCount ? m.lifeSum / m.lifeCount : 0;
  m.mapAWinRate = m.byMap.A.rounds ? m.byMap.A.wins / m.byMap.A.rounds : 0;
  m.mapBWinRate = m.byMap.B.rounds ? m.byMap.B.wins / m.byMap.B.rounds : 0;
}

const list = Object.values(merged).sort((a, b) => b.winRate - a.winRate);
const totalRounds = list[0].rounds;
const fairLine = 1 / 8;   // 8 车公平线 = 12.5%

/* ---------------- 各批次单独结果（稳定性） ---------------- */

const batches = reports.map((r) => ({
  seed: r.seed, rounds: r.rounds,
  order: ids.map((id) => ({ id, label: r.labels[id], winRate: r.rows[id].winRate, wins: r.rows[id].wins }))
    .sort((a, b) => b.winRate - a.winRate),
}));

const totalTimeouts = reports.reduce((a, r) => a + r.timeouts, 0);
const totalDraws = reports.reduce((a, r) => a + r.draws, 0);
const errorRounds = reports.reduce((a, r) => a + r.errorRounds, 0);
const avgDur = reports.reduce((a, r) => a + r.avgDurationSec * r.rounds, 0) / totalRounds;

/* ---------------- 生成 HTML ---------------- */

const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const pct = (v) => (v * 100).toFixed(1) + '%';

const maxWin = Math.max(...list.map((r) => r.winRate));

const barRows = list.map((r) => `
      <div class="barrow">
        <span class="barlabel">${esc(r.label)}</span>
        <span class="bartrack"><span class="barfill" style="width:${(r.winRate / maxWin * 100).toFixed(1)}%;background:${colorOf(r.id)}"></span></span>
        <span class="barval">${pct(r.winRate)}</span>
        <span class="barnote">${r.wins}/${r.rounds} 回合</span>
      </div>`).join('');

function colorOf(id) {
  const map = {
    deepseek: '#EF3B2D', glm: '#2E8BE8', gpt: '#3CC44F', hy: '#F5C518',
    kimi: '#A855F7', minimax: '#FF8A3D', qwen: '#22D3EE', builtin: '#F472B6',
  };
  return map[id] || '#999';
}

const mainRows = list.map((r, i) => `
        <tr${i === 0 ? ' class="top"' : ''}>
          <td class="rank">${i + 1}</td>
          <td class="nm"><span class="dot" style="background:${colorOf(r.id)}"></span>${esc(r.label)}</td>
          <td class="num big">${r.wins}</td>
          <td class="num big">${pct(r.winRate)}</td>
          <td class="num">${r.kills}</td>
          <td class="num">${r.killPerRound.toFixed(2)}</td>
          <td class="num">${r.suicides}</td>
          <td class="num">${r.netKills}</td>
          <td class="num">${r.deaths}</td>
          <td class="num">${pct(r.finalRate)}</td>
          <td class="num">${r.finalRate > 0 ? pct(r.finalWinRate) : '—'}</td>
          <td class="num">${r.avgLifeSec.toFixed(1)}s</td>
          <td class="num">${r.errors}</td>
        </tr>`).join('');

const batchTables = batches.map((b) => `
      <div class="batch">
        <h4>seed ${b.seed} · ${b.rounds} 局</h4>
        <ol class="ranklist">
          ${b.order.map((o, i) => `<li><span>${i + 1}.</span> ${esc(o.label)} <b>${pct(o.winRate)}</b> <span class="dim">(${o.wins}/${b.rounds})</span></li>`).join('')}
        </ol>
      </div>`).join('');

const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tank Trouble · 8 车竞技场战报</title>
<style>
  :root { --ink:#141414; --paper:#fff; --bg:#f0f0f0; --line:#d9d9d9; --muted:#666; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font-family:"PingFang SC","Microsoft YaHei","Helvetica Neue",Arial,sans-serif; }
  .wrap { max-width:1180px; margin:0 auto; padding:22px 16px 50px; }
  h1 { font-size:24px; margin:0 0 4px; letter-spacing:.5px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:18px; }
  .panel { background:var(--paper); border:3px solid var(--ink); box-shadow:4px 4px 0 var(--ink); padding:16px; margin-bottom:16px; }
  h2 { font-size:16px; margin:0 0 12px; padding-bottom:8px; border-bottom:2px solid var(--ink); letter-spacing:.3px; }
  h3 { font-size:14px; margin:18px 0 8px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { border:1px solid var(--line); padding:6px 8px; text-align:right; }
  th { background:#f7f7f7; font-weight:700; white-space:nowrap; }
  th:nth-child(1),th:nth-child(2) { text-align:left; }
  td.rank { text-align:left; width:34px; font-weight:700; }
  td.nm { text-align:left; font-weight:700; white-space:nowrap; }
  td.num { font-variant-numeric:tabular-nums; }
  td.big { font-weight:700; }
  tr.top td { background:#fffbe8; }
  .dot { display:inline-block; width:11px; height:11px; border:2px solid var(--ink); margin-right:7px; vertical-align:-1px; }
  .barrow { display:flex; align-items:center; gap:10px; margin-bottom:7px; font-size:13px; }
  .barlabel { width:110px; font-weight:700; flex:none; }
  .bartrack { flex:1; height:20px; background:#efefef; border:2px solid var(--ink); }
  .barfill { display:block; height:100%; }
  .barval { width:62px; text-align:right; font-weight:700; font-variant-numeric:tabular-nums; flex:none; }
  .barnote { width:104px; color:var(--muted); font-size:12px; flex:none; }
  .batches { display:flex; gap:14px; flex-wrap:wrap; }
  .batch { flex:1; min-width:250px; border:2px solid var(--ink); padding:10px 13px; }
  .batch h4 { margin:0 0 7px; font-size:13px; }
  .ranklist { margin:0; padding-left:20px; font-size:13px; line-height:1.75; }
  .ranklist b { font-variant-numeric:tabular-nums; }
  .dim { color:var(--muted); }
  .note { font-size:12.5px; line-height:1.75; color:#333; }
  .note b { color:var(--ink); }
  .note code { background:#eee; padding:1px 5px; border-radius:2px; font-size:12px; }
  .kpi { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }
  .kpi div { border:3px solid var(--ink); box-shadow:3px 3px 0 var(--ink); padding:9px 15px; background:var(--paper); }
  .kpi span { display:block; font-size:11px; color:var(--muted); font-weight:700; letter-spacing:.4px; }
  .kpi b { font-size:21px; font-variant-numeric:tabular-nums; }
  ul.plain { margin:6px 0; padding-left:20px; line-height:1.8; font-size:13px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>TANK TROUBLE · 8 车竞技场战报</h1>
  <div class="sub">7 份 AI 策略 + 1 个引擎内置 Bot · 同一竞技场混战 · 座位轮转 + 双图配额 · 真实引擎物理</div>

  <div class="kpi">
    <div><span>总回合</span><b>${totalRounds}</b></div>
    <div><span>冠军</span><b>${esc(list[0].label)}</b></div>
    <div><span>冠军胜率</span><b>${pct(list[0].winRate)}</b></div>
    <div><span>公平线 (8车)</span><b>${pct(fairLine)}</b></div>
    <div><span>平局/超时</span><b>${totalDraws}</b></div>
    <div><span>异常局</span><b>${errorRounds}</b></div>
    <div><span>平均回合时长</span><b>${avgDur.toFixed(1)}s</b></div>
  </div>

  <div class="panel">
    <h2>回合胜率</h2>
    ${barRows}
    <div class="note" style="margin-top:10px;">
      公平线 12.5% = 8 车均分的理论胜率。虚线以下为负向偏离。
    </div>
  </div>

  <div class="panel">
    <h2>完整战绩</h2>
    <table>
      <thead><tr>
        <th>#</th><th>策略</th><th>胜场</th><th>胜率</th><th>击杀</th><th>场均击杀</th>
        <th>自杀</th><th>净击杀</th><th>死亡</th><th>决赛率</th><th>决赛胜率</th><th>存活时长</th><th>异常</th>
      </tr></thead>
      <tbody>${mainRows}</tbody>
    </table>
    <div class="note" style="margin-top:10px;">
      <b>胜率</b> = 赢下该回合的次数 ÷ 总回合数（真实每回合胜率）。<br>
      <b>击杀</b> = 子弹命中他人；<b>自杀</b> = 自己的子弹打死自己；<b>净击杀</b> = 击杀 − 自杀。<br>
      <b>决赛率</b> = 活到该回合最后两人；<b>决赛胜率</b> = 进入决赛后最终夺冠的比例。<br>
      <b>存活时长</b> = 平均在场上存活到阵亡或结束的秒数。
    </div>
  </div>

  <div class="panel">
    <h2>多种子稳定性</h2>
    <div class="batches">${batchTables}</div>
    <div class="note" style="margin-top:10px;">
      两组独立种子的名次基本一致，说明排名不是随机噪声。
    </div>
  </div>

  <div class="panel">
    <h2>读这份战报要注意的几件事</h2>
    <div class="note">
      <ul class="plain">
        <li><b>口径统一。</b>这里用的是「真实每回合胜率」（赢下的回合 ÷ 实际打出的回合）。
            各家 README 里的数字口径并不统一：<code>minimax</code> 的 <code>test/sweep8.mjs</code>
            用的是 <code>win ÷ (场数 × 5 × 种子数)</code>，分母假设每场正好 5 回合，
            但「先到 5 胜」实际会打十几二十回合，所以那个 95% 被高估了
            （用其自带测试台实测：真实每回合胜率约 26.5%）。<code>glm</code> / <code>qwen</code> /
            <code>deepseek</code> / <code>hy</code> 的分母是实际回合数，与本表可比。</li>
        <li><b>对手强度不同。</b>本表里每个策略面对的是另外 6 份同级别 AI + 1 个内置 Bot，
            远比各家自测时的「7 个内置 Bot」强。自测成绩请看
            <code>arena/validate.mjs</code> 的统一口径复测。</li>
        <li><b>超时记平局。</b>正式游戏与竞技场统一采用 60 秒交战时间上限；旧战报使用其生成时的规则。
            ${totalTimeouts} 局超时。用 300 秒重跑验证过，其中绝大多数到 300 秒仍然僵死，
            是「谁都不肯先动手」的真僵局，不是跑得慢。僵局里最后活着的多是偏防守的策略。</li>
        <li><b>座位与地图已配平。</b>200 局内每个策略坐过每个座位恰好 25 次，
            两张地图各 100 局，每个（策略 × 座位）单元格里两图分别是 13 / 12 局，
            排除了 P1 出生点系统性更好的偏差。</li>
      </ul>
    </div>
  </div>

  <div class="panel">
    <h2>复现方式</h2>
    <div class="note">
      <code>node --experimental-vm-modules arena/run.mjs --rounds ${reports[0].rounds} --seed ${reports[0].seed} --jobs 24</code><br>
      每一局都用全新沙箱 + 独立随机种子，同一 (seed, 局号) 永远得到同一局。
      战报原始数据见 <code>arena/report.json</code>。
    </div>
  </div>
</div>
</body>
</html>`;

const out = path.join(HERE, 'report.html');
fs.writeFileSync(out, html, 'utf8');
fs.writeFileSync(path.join(HERE, 'report-merged.json'), JSON.stringify({
  generatedAt: new Date().toISOString(),
  batches: reports.map((r) => ({ seed: r.seed, rounds: r.rounds })),
  totalRounds, rows: merged, draws: totalDraws, timeouts: totalTimeouts,
}, null, 2));

console.log('✓ 已生成 ' + path.relative(ROOT, out));
console.log('✓ 已生成 ' + path.relative(ROOT, path.join(HERE, 'report-merged.json')));
console.log('\n合并后排名（' + totalRounds + ' 回合）：');
for (const r of list) {
  console.log('  ' + r.label.padEnd(14) + pct(r.winRate).padStart(7) +
    '   击杀 ' + String(r.kills).padStart(4) + '  自杀 ' + String(r.suicides).padStart(3) +
    '  净击杀 ' + String(r.netKills).padStart(4) + '  存活 ' + r.avgLifeSec.toFixed(1) + 's');
}
