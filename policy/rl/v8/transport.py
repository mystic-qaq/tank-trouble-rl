import subprocess
from pathlib import Path
from ..v2.transport import Worker as BinaryWorker, ROOT


class Worker(BinaryWorker):
    def __init__(self):
        self.process = subprocess.Popen(
            ['node', '--experimental-vm-modules', '--disable-warning=ExperimentalWarning',
             str(Path(__file__).with_name('worker.mjs'))],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
