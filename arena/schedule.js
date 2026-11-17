/* 浏览器与无头竞技场共用：每组固定地图、出生点，逐局轮转全部席位。 */
(function () {
  const ids = ['deepseek', 'glm', 'gpt', 'hy', 'kimi', 'minimax', 'qwen', 'builtin'];
  const allowed = ids.concat(['rl_win', 'rl_frag', 'rl_v7']);
  function roster(selected = ids) {
    if (!Array.isArray(selected) || selected.length < 2 || selected.length > 8 ||
        new Set(selected).size !== selected.length || selected.some(id => !allowed.includes(id))) {
      throw new Error('请选择 2–8 个不同的策略：' + allowed.join(', '));
    }
    return selected.slice();
  }
  function seed(base, index) {
    let h = (base >>> 0) ^ 0x9e3779b9;
    h = Math.imul(h ^ (index + 0x85ebca6b), 0xc2b2ae35);
    h ^= h >>> 13;
    return (h >>> 0) || 1;
  }
  globalThis.TTArenaSchedule = {
    roster,
    seed,
    plan(round, selected = ids, baseSeed = 20260101) {
      const players = roster(selected);
      const block = Math.floor(round / players.length);
      return {
        seats: players.map((_, s) => players[(s + round) % players.length]),
        mapSeed: [20260101, 88888888][block % 2],
        spawnSeed: seed(baseSeed, block),
      };
    },
  };
})();
