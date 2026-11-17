"""Independent binary workers; callers schedule whichever environment becomes ready."""
import json,os,struct,subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]

class Worker:
    def __init__(self):
        self.process=subprocess.Popen(['node','--experimental-vm-modules','--disable-warning=ExperimentalWarning',str(Path(__file__).with_name('worker.mjs'))],
            cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
    def read(self,n):
        data=bytearray()
        while len(data)<n:
            part=self.process.stdout.read(n-len(data))
            if not part:raise RuntimeError('Environment worker exited')
            data.extend(part)
        return data
    def request(self,**message):
        data=json.dumps(message,separators=(',',':')).encode()
        self.process.stdin.write(struct.pack('<II',len(data),0)+data);self.process.stdin.flush()
        n,m=struct.unpack('<II',self.read(8))
        header=json.loads(self.read(n));array=np.frombuffer(self.read(m),dtype='<f4').copy()
        if not header.get('ok'):raise RuntimeError(header.get('error'))
        return header,array
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
