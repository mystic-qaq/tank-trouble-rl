import { parentPort, workerData } from 'node:worker_threads';
import { TankEnv } from './env.mjs';
const environments = workerData.indices.map(i => new TankEnv({...workerData.options,seed:workerData.options.seed+i*100003}));
parentPort.on('message', async message => {
  try {
    const observations = [], results = [];
    for (let j=0;j<environments.length;j++) {
      const env = environments[j];
      if (message.command === 'reset') observations.push(await env.reset(message.configs?.[workerData.indices[j]]));
      else {
        const index = workerData.indices[j];
        if (message.paused?.[index]) {
          results.push({reward:0,done:false,active:0,action:[0,0,0],info:null,seed:env.seed});
          observations.push(env.observation());
          continue;
        }
        const a = message.actions?.[index] ?? [0,0,0];
        const result = env.step({drive:a[0],steer:a[1],shoot:a[2]},message.teacher);
        results.push({ reward:result.reward, done:result.terminated, active:result.active,
          seed:env.seed, action:[result.action.drive,result.action.steer,+result.action.shoot], info:result.info });
        observations.push(result.terminated && message.autoreset !== false ? await env.reset() : result.observation);
      }
    }
    const data = new Float32Array(observations.length*926);
    observations.forEach((o,i) => data.set(o,i*926));
    parentPort.postMessage({indices:workerData.indices, data:data.buffer, results},[data.buffer]);
  } catch(e) { parentPort.postMessage({error:e.stack}); }
});
