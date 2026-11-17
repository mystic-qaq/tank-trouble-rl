// Local-only static server, including JS modules and the bundled WASM runtime.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const at=process.argv.indexOf('--port'),port=at<0?8765:Number(process.argv[at+1]);
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8',
  '.css':'text/css; charset=utf-8','.json':'application/json; charset=utf-8','.wasm':'application/wasm','.svg':'image/svg+xml',
  '.png':'image/png','.jpg':'image/jpeg'};
const server=http.createServer((req,res)=>{
  let filename;
  try{
    const relative=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    filename=path.resolve(root,'.'+relative);
    if(filename!==root&&!filename.startsWith(root+path.sep))throw new Error('outside root');
    if(fs.statSync(filename).isDirectory())filename=path.join(filename,'index.html');
    if(!fs.statSync(filename).isFile())throw new Error('not file');
  }catch(_){res.writeHead(404);res.end('Not found');return;}
  res.writeHead(200,{'Content-Type':types[path.extname(filename)]||'application/octet-stream','Cache-Control':'no-cache',
    'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'});
  fs.createReadStream(filename).pipe(res);
});
server.listen(port,'127.0.0.1',()=>console.log(`V2 four-way spectator: http://127.0.0.1:${port}/arena/?preset=v2`));
