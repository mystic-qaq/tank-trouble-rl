/* ===========================================================
   sim_test.js — GLM 策略无头对战测试（Node）
   -----------------------------------------------------------
   用最小 stub 加载游戏逻辑与策略，批量跑对局统计胜率。
   用法：
     node sim_test.js [场景] [场次]
     场景: ffa8 (glm×1 vs bot×7) | team4 (glm×4 vs bot×4) |
           duel (glm×1 vs bot×1) | base8 (内置 8 bot 基线)
   =========================================================== */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

/* ---------------- 浏览器 stub ---------------- */
const root = global;
root.window = root;
root.TT = {};

function makeCtxStub() {
  const chain = { addColorStop() {} };
  return new Proxy({}, {
    get(t, p) {
      if (p === 'canvas') return null;
      if (!(p in t)) t[p] = (...a) => chain;
      return t[p];
    },
    set() { return true; },
  });
}

function makeCanvasStub() {
  return { width: 300, height: 150, style: {}, getContext: () => makeCtxStub(),
           addEventListener() {}, getBoundingClientRect: () => ({ width: 1280, height: 800 }) };
}

root.document = {
  createElement: () => makeCanvasStub(),
  createElementNS: () => makeCanvasStub(),
  getElementById: () => makeCanvasStub(),
  addEventListener() {},
  readyState: 'complete',
  hidden: false,
};
root.addEventListener = () => {};
root.requestAnimationFrame = () => 0;
root.performance = { now: () => Date.now() };
root.localStorage = { getItem: () => null, setItem() {} };
Object.defineProperty(root, 'navigator', { value: { userAgent: 'node' }, configurable: true });

/* ---------------- 加载游戏脚本 ---------------- */
const JS_DIR = path.resolve(__dirname, '../../..', 'js');
const GAME_JS = ['util.js', 'input.js', 'audio.js', 'map.js', 'entities.js', 'ai.js', 'game.js'];
for (const f of GAME_JS) {
  const code = fs.readFileSync(path.join(JS_DIR, f), 'utf8');
  vm.runInThisContext(code, { filename: f });
}
const BUILTIN_BOT = root.TT.updateBot;

/* ---------------- 加载 GLM 策略 ---------------- */
const policyCode = fs.readFileSync(path.resolve(__dirname, '../glm_policy.js'), 'utf8');
vm.runInThisContext(policyCode, { filename: 'glm_policy.js' });
const GLM = root.GLM_POLICY;
if (!GLM) { console.error('策略加载失败'); process.exit(1); }

/* ---------------- 对局分配器 ---------------- */
let glmIndices = new Set();
root.TT.updateBot = function (tank, game, dt) {
  if (glmIndices.has(tank.index)) return GLM.update(tank, game, dt);
  return BUILTIN_BOT(tank, game, dt);
};

/* ---------------- 跑一场对局 ---------------- */
const game = new root.TT.Game(makeCanvasStub());

function runMatch(roundsToWin, maxSeconds = 600, slots) {
  game.startMatch(slots || ['bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot'], roundsToWin);
  const maxFrames = maxSeconds * 60;
  let frames = 0;
  let lastShotLog = 0, shotEvents = [];
  while (game.state !== 'matchover' && frames < maxFrames) {
    game.update(1 / 60);
    frames++;
    if (process.env.DEBUG && frames - lastShotLog > 300) {
      lastShotLog = frames;
      const st = game.tanks.filter(t => t.alive).map(t =>
        `#${t.index}(${t.x | 0},${t.y | 0})a=${t.angle.toFixed(1)}v=${t.speed | 0}`).join(' ');
      shotEvents.push(`${(frames / 60) | 0}s: ${st} bullets=${game.bullets.length}`);
    }
  }
  if (process.env.DEBUG && game.state !== 'matchover') {
    console.log('=== TIMEOUT DIAG at', game.round, 'round ===');
    shotEvents.slice(-15).forEach(l => console.log(l));
  }
  return {
    scores: game.scores.slice(),
    kills: game.kills.slice(),
    rounds: game.round,
    timeout: game.state !== 'matchover',
  };
}

/* ---------------- 场景统计 ---------------- */
function runScenario(name, glmCount, nMatches, roundsToWin, slots) {
  glmIndices = new Set();
  for (let i = 0; i < glmCount; i++) glmIndices.add(i);

  let glmWins = 0, glmRounds = 0, totalRounds = 0, glmKills = 0, glmSelf = 0;
  let botWins = 0, timeouts = 0;
  const t0 = Date.now();

  for (let m = 0; m < nMatches; m++) {
    const r = runMatch(roundsToWin, 600, slots);
    if (r.timeout) timeouts++;
    let gw = 0, bw = 0;
    const nActive = slots ? slots.filter(s => s !== 'off').length : 8;
    for (let i = 0; i < 8; i++) {
      if (glmIndices.has(i)) { gw += r.scores[i]; glmKills += Math.max(0, r.kills[i]); if (r.kills[i] < 0) glmSelf += r.kills[i]; }
      else bw += r.scores[i];
    }
    glmRounds += gw; totalRounds += gw + bw;
    // 比赛冠军
    let champ = 0;
    for (let i = 1; i < 8; i++) if (r.scores[i] > r.scores[champ]) champ = i;
    if (glmIndices.has(champ) && r.scores[champ] > 0) glmWins++;
    else botWins++;
  }

  const dt = ((Date.now() - t0) / 1000).toFixed(1);
  const roundPct = totalRounds ? (glmRounds / totalRounds * 100).toFixed(1) : '0.0';
  console.log(
    `[${name}] glm×${glmCount} ${nMatches} 场 (先到${roundsToWin}胜, ${dt}s)` +
    ` | 冠军率 ${(glmWins / nMatches * 100).toFixed(0)}%` +
    ` | 回合胜率 ${roundPct}%` +
    ` | glm场均击杀 ${(glmKills / nMatches).toFixed(2)} 场均自杀 ${(glmSelf / nMatches).toFixed(2)}` +
    (timeouts ? ` | 超时${timeouts}场` : '')
  );
}

/* ---------------- 主流程 ---------------- */
const scenario = process.argv[2] || 'all';
const n = parseInt(process.argv[3] || '12', 10);

const SLOTS_1V1 = ['bot', 'bot', 'off', 'off', 'off', 'off', 'off', 'off'];

if (scenario === 'all') {
  runScenario('1v1  ', 1, n, 3, SLOTS_1V1);
  runScenario('duel ', 1, n, 3);
  runScenario('team4', 4, n, 3);
  runScenario('ffa8 ', 1, n, 3);
  runScenario('base8', 0, n, 3);   // 基线: 全内置
} else {
  const map = { ffa8: 1, team4: 4, duel: 1, base8: 0, '1v1': 1 };
  runScenario(scenario, map[scenario] !== undefined ? map[scenario] : 1, n, 3,
    scenario === '1v1' ? SLOTS_1V1 : undefined);
}
process.exit(0);
