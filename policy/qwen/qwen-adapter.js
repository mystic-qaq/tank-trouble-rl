/* ===========================================================
   qwen-adapter.js — 可选的「一行接入」适配器（不是策略本体）
   -----------------------------------------------------------
   你的 Arena 装载器如果只想要最省事的接法，加载完游戏 js 和
   policy/qwen/policy.js 之后，再加载这个文件即可：

       <script src="js/ai.js"></script>
       <script src="policy/qwen/policy.js"></script>
       <script src="policy/qwen/qwen-adapter.js"></script>   <!-- 可选 -->

   默认行为：把 Qwen 策略接到「指定的那一个座位」，其余座位继续用
   游戏内置 bot。也可以用 QwenAdopt.all() 接管全部 bot 座位。

   注意：这个文件会改写 TT.updateBot。如果你的 Arena 已经有自己的
   路由逻辑（推荐做法），不要加载它，直接调用
       TT.policies.qwen.update(tank, game, dt)
   并把返回的 {drive, steer, shoot} 交给游戏即可。
   =========================================================== */

(function () {
  'use strict';
  var TT = window.TT;
  var POL = TT && TT.policies && TT.policies.qwen;
  if (!TT || !TT.updateBot || !POL) {
    console.warn('[qwen-adapter] 需要先加载 js/ai.js 与 policy/qwen/policy.js');
    return;
  }

  var builtin = TT.updateBot;
  var seats = null;          // null = 未启用；数组 = 接管的座位号；'all' = 所有 bot 座位
  var installed = false;

  function install() {
    if (installed) return;
    installed = true;
    TT.updateBot = function (tank, game, dt) {
      if (seats === 'all') return POL.update(tank, game, dt);
      if (seats && seats.indexOf(tank.index) >= 0) return POL.update(tank, game, dt);
      if (!seats) return POL.update(tank, game, dt);   // 没指定就全接管
      return builtin(tank, game, dt);
    };
  }

  var API = window.QwenAdopt = {
    /** 只接管某个座位（默认 0 号），其余继续用内置 bot */
    seat: function (i) { install(); seats = [i | 0]; return API; },
    /** 接管若干座位 */
    seats: function (list) { install(); seats = list.slice(); return API; },
    /** 接管所有电脑座位 */
    all: function () { install(); seats = 'all'; return API; },
    /** 交还给游戏内置 bot */
    off: function () { if (installed) { TT.updateBot = builtin; installed = false; } seats = null; return API; },
    policy: POL,
  };

  /* URL 参数快捷开关：?qwen=0 / ?qwen=all / ?qwen=0,3
     手写解析，不依赖 URLSearchParams（有些注入环境里没有） */
  try {
    var search = (typeof location !== 'undefined' && location.search) || '';
    var m = /[?&]qwen=([^&]+)/.exec(search);
    if (m) {
      var val = decodeURIComponent(m[1]);
      if (val === 'all') API.all();
      else API.seats(val.split(',').map(function (s) { return parseInt(s, 10) || 0; }));
    }
  } catch (e) { /* 忽略 */ }
})();
