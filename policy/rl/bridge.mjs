import readline from 'node:readline';
import { Worker } from 'node:worker_threads';
import { SCHEMA } from './observation.mjs';
let workers = [], count = 0;
const send = value => process.stdout.write(JSON.stringify(value)+'\n');
function request(worker,message) {
  return new Promise((resolve,reject) => {
    const cleanup = () => {worker.off('message',onMessage);worker.off('error',onError);worker.off('exit',onExit);};
    const onMessage = response => { cleanup(); response.error ? reject(new Error(response.error)) : resolve(response); };
    const onError = error => {cleanup();reject(error);};
    const onExit = code => {cleanup();reject(new Error('Environment worker exited '+code));};
    worker.once('message',onMessage);worker.once('error',onError);worker.once('exit',onExit);
    worker.postMessage(message);
  });
}
async function handle(message) {
  if (message.command === 'init') {
    count = message.num_envs ?? 16;
    if (!Number.isInteger(count)||count<1||count>256) throw new Error('Invalid num_envs');
    const nworkers = Math.max(1,Math.min(message.workers??count,count));
    workers = Array.from({length:nworkers},(_,i) => new Worker(new URL('./env-worker.mjs',import.meta.url),{
      workerData:{indices:Array.from({length:count},(_,j)=>j).filter(j=>j%nworkers===i),options:message.options},
      execArgv:['--experimental-vm-modules','--disable-warning=ExperimentalWarning'],
    }));
    message = {command:'reset'};
  }
  if (message.command === 'close') {await Promise.all(workers.map(w=>w.terminate()));workers=[];return {closed:true};}
  if (!workers.length) throw new Error('init required');
  const batches = await Promise.all(workers.map(w=>request(w,message)));
  const observations = new Float32Array(count*SCHEMA.size), results = Array(count);
  for (const batch of batches) {
    const data = new Float32Array(batch.data);
    batch.indices.forEach((index,j) => {observations.set(data.subarray(j*SCHEMA.size,(j+1)*SCHEMA.size),index*SCHEMA.size); results[index]=batch.results[j]??null;});
  }
  return {schema:SCHEMA, observations:Buffer.from(observations.buffer).toString('base64'),results};
}
for await (const line of readline.createInterface({input:process.stdin})) {
  try { const message=JSON.parse(line);send(await handle(message)); if(message.command==='close') break; }
  catch(e) {send({error:e.stack});}
}
await Promise.all(workers.map(w=>w.terminate()));
process.stdin.destroy();
