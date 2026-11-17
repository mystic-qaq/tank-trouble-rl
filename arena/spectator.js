(function () {
  'use strict';
  const isLearned=id=>id==='rl_win'||id==='rl_frag'||id==='rl_v7';
  let learned;
  async function loadModels(ids){
    if(!ids.some(isLearned))return [];
    if(location.protocol==='file:')throw new Error('本机模型需要本机服务：运行 node arena/serve.mjs 后打开 http://127.0.0.1:8765/arena/?preset=v7');
    learned=learned||await import('./rl-controller.mjs');return learned.loadModels(ids);
  }
  const $ = id => document.getElementById(id);
  const roster = [
    { id:'deepseek', name:'DeepSeek', fn:() => TT.deepseekPolicy },
    { id:'glm', name:'GLM', fn:() => TT.Policies.glm.update },
    { id:'gpt', name:'GPT', fn:() => TT.policies.gpt },
    { id:'hy', name:'HY', fn:() => TT.Policies.hy },
    { id:'kimi', name:'Kimi', fn:() => window.KIMI_TANK.update },
    { id:'minimax', name:'minimax', fn:() => window.TTPolicy.updateBot },
    { id:'qwen', name:'Qwen', fn:() => TT.policies.qwen.update },
    { id:'builtin', name:'内置 Bot', fn:() => window.arenaBuiltinBot },
    { id:'rl_win', name:'RL 胜率', fn:() => tank => trackers.find(t=>t.seat===tank.index)?.action },
    { id:'rl_frag', name:'RL 净击杀', fn:() => tank => trackers.find(t=>t.seat===tank.index)?.action },
    { id:'rl_v7', name:'V7 瞄准蒸馏', fn:() => tank => trackers.find(t=>t.seat===tank.index)?.action },
  ];
  const controllers = {};
  roster.forEach((s, i) => {
    s.palette = { ...TT.COLORS[i%8], name:s.name };
    if(s.id==='rl_win')s.palette={...s.palette,main:'#7651D2',body:'#54399C',tread:'#372661'};
    if(s.id==='rl_frag')s.palette={...s.palette,main:'#119977',body:'#08765A',tread:'#064C3B'};
    if(s.id==='rl_v7')s.palette={...s.palette,main:'#E0533D',body:'#A83A2A',tread:'#6E2419'};
    try { controllers[s.id] = s.fn(); } catch (_) { /* 在开始时提示缺失策略 */ }
  });
  const byId = Object.fromEntries(roster.map(s => [s.id, s]));
  const cv = $('cv'), ctx = cv.getContext('2d');
  const DT = 1 / 60, BASE_SEED = 20260101;
  const parameters=new URLSearchParams(location.search),presetName=parameters.get('preset');
  const preset=presetName==='v2'||presetName==='v7';
  if(presetName==='v2')document.title='Tank Trouble · V2 本机四方对战';
  if(presetName==='v7')document.title='Tank Trouble · V7 瞄准蒸馏 对阵两位教师';
  const fourWay=['rl_win','rl_frag','minimax','builtin'];
  const v7Way=['rl_v7','hy','gpt','builtin'];
  const presetRoster=presetName==='v7'?v7Way:fourWay;
  let selected = preset?presetRoster.slice():roster.slice(0,8).map(s => s.id), stats, game, seats, trackers=[];
  let running = false, speed = preset ? 0.5 : 1, roundIndex = 0, completed = 0;
  let acc = 0, last = null, pending = 0, elapsed = 0, settled = false, timedOut = false;
  let inferenceMs=0,actualSpeed=0,paceStart=performance.now(),paceSim=0,setupSerial=0;
  const strategyRandom=[];
  let frameQueue=Promise.resolve();
  let roundStart;

  $('players').innerHTML = roster.map(s => '<label><input type="checkbox" value="' + s.id + '"'+(selected.includes(s.id)?' checked':'')+'>' + s.name + '</label>').join('');
  function checked() { return Array.from($('players').querySelectorAll('input:checked'), el => el.value); }
  $('players').onchange = () => { $('selectedCount').textContent = checked().length + ' / 8'; };
  $('btnAll').onclick = () => { $('players').querySelectorAll('input').forEach(el => el.checked = !isLearned(el.value)); $('players').onchange(); };
  $('players').onchange();$('speed').value=String(speed);
  $('btnRL').textContent=presetName==='v7'?'V7 对阵两位教师':'V2 四方对战';

  function resetStats() {
    stats = Object.fromEntries(selected.map(id => [id, { wins:0, kills:0, suicides:0, shots:0, errors:0 }]));
    completed = 0; roundIndex = 0;
  }
  function setRunning(value) {
    running = value; last = null; acc = 0;
    paceStart=performance.now();paceSim=0;actualSpeed=0;
    $('btnRun').classList.toggle('on', running);
    $('btnRun').textContent = running ? '⏸ 暂停' : '▶ 继续';
  }
  function newRound(advance = true) {
    if(!advance&&roundStart){stats=structuredClone(roundStart.stats);completed=roundStart.completed;}
    if (advance) roundIndex++;
    roundStart={stats:structuredClone(stats),completed};
    const plan = TTArenaSchedule.plan(roundIndex - 1, selected, BASE_SEED);
    seats = plan.seats;
    strategyRandom.length=0;
    seats.forEach((_,seat)=>strategyRandom.push(TT.util.rng(TTArenaSchedule.seed(plan.spawnSeed,30+seat))));
    game = new TT.Game(cv);
    game.mode = 'match';
    game.slots = Array.from({ length:8 }, (_, i) => i < seats.length ? 'bot' : 'off');
    game.roundsToWin = Infinity;
    game.spawnRandom = TT.util.rng(plan.spawnSeed);
    TT.CFG.MAP_SEEDS = [plan.mapSeed];
    const kill = game.killTank.bind(game);
    game.killTank = (tank, bullet) => {
      if (tank.alive && bullet) {
        const row = stats[seats[bullet.owner]];
        if (row) row[bullet.owner === tank.index ? 'suicides' : 'kills']++;
      }
      kill(tank, bullet);
    };
    TT.updateBot = (tank, g, dt) => {
      const id = seats[tank.index];
      const previousRandom=Math.random;
      try {
        Math.random=strategyRandom[tank.index];
        const out = controllers[id](tank, g, dt) || {};
        return { drive:Math.max(-1, Math.min(1, Number(out.drive) || 0)), steer:Number(out.steer) || 0, shoot:!!out.shoot };
      } catch (_) { stats[id].errors++; return { drive:0, steer:0, shoot:false }; }
      finally{Math.random=previousRandom;}
    };
    game.startRound();
    game.tanks.forEach(t => {
      const s = byId[seats[t.index]];
      t.palette = s.palette; t.color = s.palette.main; t.name = s.name;
      const fire=t.fire.bind(t);
      t.fire=g=>{const n=g.bullets.length;fire(g);stats[s.id].shots+=g.bullets.length-n;};
    });
    trackers=learned?learned.createTrackers(game,seats):[];
    $('inferenceHud').hidden=!trackers.length;
    paceStart=performance.now();paceSim=0;actualSpeed=0;
    acc = 0; pending = 0; elapsed = 0; settled = false; timedOut = false;
    $('hRound').textContent = roundIndex;
    draw(); refresh();
  }
  function settle() {
    if (settled) return;
    settled = true; completed++;
    if (game.roundWinner >= 0) stats[seats[game.roundWinner]].wins++;
    pending = 2.25; acc = 0;
  }
  function step(){
    const expected=game;
    const result=frameQueue.then(()=>expected===game?performFrame():false);
    frameQueue=result.catch(()=>{});return result;
  }
  async function performFrame() {
    if (!game||settled) return false;
    const playing = game.state === 'playing';
    const snapshot=game,currentTrackers=trackers;
    if(playing&&currentTrackers.length){
      const ms=await learned.beforeFrame(currentTrackers);
      if(snapshot!==game)return false;
      inferenceMs=inferenceMs?inferenceMs*.8+ms*.2:ms;
    }
    game.update(DT);
    if(playing&&currentTrackers.length)learned.afterFrame(currentTrackers);
    if (playing) elapsed += DT;
    if(playing)paceSim+=DT;
    elapsed = game.roundTime;
    timedOut = game.roundTimedOut;
    if (game.state === 'over') settle();
    return true;
  }

  function draw() {
    if (!game || !game.map) return;
    const W = game.map.w, H = game.map.h;
    ctx.setTransform(cv.width / W, 0, 0, cv.height / H, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(game.staticLayer, 0, 0, W, H);
    game.fx.drawSmoke(ctx);
    game.tanks.forEach(t => t.draw(ctx));
    game.bullets.filter(b => !b.dead).forEach(b => {
      b.draw(ctx);
      ctx.strokeStyle = b.color; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(b.x, b.y, b.radius + 2, 0, Math.PI * 2); ctx.stroke();
    });
    game.fx.draw(ctx);
    if ($('showNames').checked) {
      const cssScale = cv.clientWidth / W;
      const fontSize = Math.max(16, 12 / cssScale);
      ctx.font = 'bold ' + fontSize + 'px "Microsoft YaHei", sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      game.tanks.filter(t => t.alive).forEach(t => {
        const width = ctx.measureText(t.name).width + 12;
        const x = TT.util.clamp(t.x, width / 2 + 8, W - width / 2 - 8);
        const y = t.y < fontSize + 40 ? t.y + 32 : t.y - 31;
        ctx.fillStyle = 'rgba(255,255,255,.9)'; ctx.fillRect(x - width / 2, y - fontSize / 2 - 3, width, fontSize + 6);
        ctx.fillStyle = '#141414'; ctx.fillText(t.name, x, y);
      });
    }
  }
  function refresh() {
    if(!game)return;
    const winner = settled && game.roundWinner >= 0 ? seats[game.roundWinner] : null;
    const message = settled ? (winner ? byId[winner].name + ' 赢下本回合' : timedOut ? '60 秒超时 · 平局' : '同归于尽 · 平局') : game.state === 'intro' ? '准备中' : '交战中';
    $('hBanner').textContent = message;
    $('result').textContent = message;
    $('hAlive').textContent = game.tanks.filter(t => t.alive).length;
    $('hTime').textContent = elapsed.toFixed(1);
    $('completed').textContent = completed;
    $('hInference').textContent=inferenceMs.toFixed(1);$('hActual').textContent=actualSpeed.toFixed(2);
    $('legend').innerHTML = selected.slice().sort((a,b) => stats[b].wins - stats[a].wins || stats[b].kills - stats[a].kills || selected.indexOf(a) - selected.indexOf(b)).map(id => {
      const s = byId[id], row = stats[id], tank = game.tanks.find(t => seats[t.index] === id);
      return '<tr class="' + (!tank.alive ? 'dead' : '') + (winner === id ? ' win' : '') + '"><td><span class="identity"><span class="sw" style="background:' + s.palette.main + '"></span>' + s.name + '</span><span class="status">' + (tank.alive ? '存活' : '淘汰') + (row.errors ? ' · 策略异常 ' + row.errors : '') + '</span></td><td>' + row.wins + '</td><td>' + row.kills + '</td><td>' + row.suicides + '</td><td>'+row.shots+'</td></tr>';
    }).join('');
  }
  async function loop(ts) {
    const realDt = last === null ? 0 : Math.max(0, Math.min((ts - last) / 1000, 0.25));
    last = ts;
    try{if (running && !document.hidden) {
      if (settled) { pending -= realDt; if (pending <= 0) newRound(); }
      else {
        acc = Math.min(acc+realDt*speed,Math.max(.25,speed*.25));
        let steps = 0;const started=performance.now();
        while (running&&acc >= DT && steps++ < 240 && !settled) {
          acc -= DT;await step();
          if(trackers.length&&performance.now()-started>14)break;
        }
      }
    }}catch(error){setRunning(false);$('selectionError').textContent='模型运行已暂停：'+error.message;console.error(error);}
    const now=performance.now();
    if(game?.state!=='playing'||!running){paceStart=now;paceSim=0;}
    else if(now-paceStart>=500){actualSpeed=paceSim/((now-paceStart)/1000);paceStart=now;paceSim=0;}
    draw(); refresh();
    requestAnimationFrame(loop);
  }
  $('btnRun').onclick = () => setRunning(!running);
  $('btnStep').onclick = async () => {
    setRunning(false);$('btnStep').disabled=true;
    try{if (settled) newRound(); else await step();draw();refresh();}
    catch(error){$('selectionError').textContent=error.message;}
    finally{$('btnStep').disabled=false;}
  };
  $('btnReset').onclick = () => newRound(false);
  $('speed').onchange = () => { speed = Number($('speed').value); acc = 0; };
  $('showNames').onchange = draw;
  async function apply(players){
    const serial=++setupSerial;setRunning(false);
    $('btnApply').disabled=true;$('btnRL').disabled=true;$('btnStep').disabled=true;$('btnRun').disabled=true;$('btnReset').disabled=true;
    try {
      players=TTArenaSchedule.roster(players);
      const missing = players.filter(id => typeof controllers[id] !== 'function');
      if (missing.length) throw new Error('策略未加载：' + missing.join(', '));
      $('modelStatus').textContent=players.some(isLearned)?'正在加载本机模型…':'使用原版固定策略。';
      const models=await loadModels(players);if(serial!==setupSerial)return;
      const describeModel=m=>m.architecture==='aim_distilled_servo_v7'
        ? 'V7 瞄准蒸馏（'+m.architecture+'）· 只完成蒸馏，RL 无收益，未通过冻结验收'
        : 'V2b · BC＋DAgger 初始化模型 · 未进行 PPO';
      $('modelStatus').textContent=models.length?[...new Set(models.map(describeModel))].join('；')+'。模型在本机运行，每秒 60 次决策；本页面不训练。':'使用原版固定策略。';
      $('modelStatus').title=models.map(m=>(m.candidate?m.candidate+' · step '+m.checkpoint_step:m.architecture)
        +' · policy_style '+(m.policy_style||'—')+' · SHA-256 '+m.source_sha256).join('\n');
      selected = players; resetStats(); newRound(); setRunning(true);
      $('selectionError').textContent = '';
    } catch (e) { $('selectionError').textContent = e.message; }
    finally{if(serial===setupSerial){$('btnApply').disabled=false;$('btnRL').disabled=false;$('btnStep').disabled=!game;$('btnRun').disabled=!game;$('btnReset').disabled=!game;}}
  }
  $('btnApply').onclick=()=>apply(checked());
  $('btnRL').onclick=()=>{
    $('players').querySelectorAll('input').forEach(el=>el.checked=presetRoster.includes(el.value));$('players').onchange();
    speed=.5;$('speed').value='0.5';apply(presetRoster.slice());
  };
  document.addEventListener('visibilitychange', () => { last = null; acc = 0; });
  new ResizeObserver(() => { if (game) { game.resize(); draw(); } }).observe(cv);
  window.arenaDebug={get game(){return game;},get trackers(){return trackers;},get selected(){return selected.slice();},
    get stats(){return structuredClone(stats);},get completed(){return completed;},get running(){return running;},
    setRunning,async advance(frames){setRunning(false);for(let i=0;i<frames;i++){if(settled)newRound();await step();}draw();refresh();},restart(){resetStats();newRound();}};
  apply(selected.slice()).then(()=>requestAnimationFrame(loop));
})();
