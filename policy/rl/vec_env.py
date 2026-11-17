"""Batched persistent Node environments; observations use packed Float32 transport."""
import base64
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

class VecEnv:
    def __init__(self, num_envs=16, workers=None, objective="win", seed=20261106, **options):
        self.num_envs = num_envs
        self.process = subprocess.Popen(
            ["node", "--experimental-vm-modules", "--disable-warning=ExperimentalWarning", str(ROOT / "policy/rl/bridge.mjs")],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self.options = dict(objective=objective, seed=seed, **options)
        response = self.request(dict(command="init", num_envs=num_envs, workers=workers or num_envs, options=self.options))
        self.schema = response["schema"]
        self.observations = self.decode(response)

    def request(self, message):
        self.process.stdin.write(json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Node environment closed: {self.process.poll()}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def decode(self, response):
        return np.frombuffer(base64.b64decode(response["observations"]), dtype="<f4").reshape(self.num_envs, -1).copy()

    def reset(self, configs=None):
        response=self.request(dict(command="reset",configs=configs))
        self.observations=self.decode(response)
        return self.observations

    def step(self, actions=None, teacher=None, autoreset=True, paused=None):
        response = self.request(dict(command="step", actions=None if actions is None else np.asarray(actions).tolist(), teacher=teacher,autoreset=autoreset,paused=paused))
        self.observations = self.decode(response)
        return self.observations, response["results"]

    def close(self):
        if self.process.poll() is None:
            try:
                self.request(dict(command="close"))
                self.process.stdin.close()
                self.process.wait(timeout=10)
            finally:
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
