/* ===========================================================
   main.js — 启动
   =========================================================== */

(function () {
  function boot() {
    TT.Input.init();

    const canvas = document.getElementById('game');
    const game = new TT.Game(canvas);
    TT.UI.init(game);

    game.startDemo();
    game.start();

    let resizeTimer = 0;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => game.resize(), 120);
    });

    const unlock = () => {
      TT.Audio.init();
      TT.Audio.resume();
    };
    window.addEventListener('pointerdown', unlock);
    window.addEventListener('keydown', unlock);

    window.TT.game = game;
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
