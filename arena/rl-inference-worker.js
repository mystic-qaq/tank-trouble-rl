/* ONNX Runtime and model execution stay off the rendering thread. */
importScripts('vendor/onnxruntime/ort.wasm.min.js');
ort.env.wasm.numThreads=1;
ort.env.wasm.wasmPaths=new URL('vendor/onnxruntime/',self.location.href).href;
const sessions=new Map();
let queue=Promise.resolve();
async function handle(message){
  if(message.op==='load'){
    for(const model of message.models){
      if(sessions.has(model.id))continue;
      const response=await fetch(model.url);if(!response.ok)throw new Error('模型加载失败：'+model.id);
      const bytes=await response.arrayBuffer();
      const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
      if(digest!==model.sha256)throw new Error('模型校验失败：'+model.id);
      sessions.set(model.id,await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'}));
    }
    return {loaded:Array.from(sessions.keys())};
  }
  if(message.op==='infer'){
    const start=performance.now(),outputs=[];
    for(const input of message.inputs){
      const session=sessions.get(input.id);if(!session)throw new Error('模型未就绪：'+input.id);
      const result=await session.run({obs:new ort.Tensor('float32',input.observation,[1,1966])});
      const controls=Array.from(result.controls.data),logits=Array.from(result.logits.data);
      if(!controls.every(Number.isFinite)||!logits.every(Number.isFinite))throw new Error('模型输出非有限值：'+input.id);
      outputs.push({id:input.id,controls,logits});
    }
    return {outputs,milliseconds:performance.now()-start};
  }
  throw new Error('未知推理请求');
}
self.onmessage=({data})=>{
  queue=queue.then(async()=>{
    try{self.postMessage({request:data.request,result:await handle(data)});}
    catch(error){self.postMessage({request:data.request,error:error.stack||error.message});}
  });
};
