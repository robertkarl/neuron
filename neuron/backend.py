from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_trace(path):
    """Read version 1 little-endian records; reject partial or invalid output."""
    means, counts, norms, interventions = {}, {}, {}, []
    with open(path, 'rb') as f:
        if f.read(8) != b'NEURON01':
            raise ValueError('missing neuron hook trace magic')
        while header := f.read(16):
            if len(header) != 16:
                raise ValueError('truncated trace header')
            kind, layer, width, count = struct.unpack('<4I', header)
            if layer >= 32 or not 0 < count <= 8192:
                raise ValueError('invalid trace dimensions')
            if (kind, width) not in ((1, 9216), (2, 1), (3, 3)):
                raise ValueError('unknown trace kind or width')
            size = count if kind == 2 else width
            data = f.read(size * 4)
            if len(data) != size * 4:
                raise ValueError('truncated trace payload')
            values = np.frombuffer(data, dtype='<f4').astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError('nonfinite activations')
            if kind == 1:
                means[layer] = means.get(layer, 0) + values * count
                counts[layer] = counts.get(layer, 0) + count
            elif kind == 2:
                norms.setdefault(layer, []).extend(values.tolist())
            else:
                interventions.append(dict(layer=layer, neuron=int(values[0]),
                                          before=float(values[1]), after=float(values[2]), count=count))
    return ({layer: values / counts[layer] for layer, values in means.items()}, norms, interventions)


class Server:
    """Own one server and serialize all requests with a private local side channel."""

    def __init__(self, model, directory, gpu_layers=99):
        self.directory = Path(directory).resolve()
        self.config = self.directory / 'hook.json'
        self.trace = Path(str(self.config) + '.trace')
        self.model = Path(model).resolve()
        self.gpu_layers = gpu_layers
        self.process = None
        self.log = None
        self.request_id = 0

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.config, dict(trace=False, steering=[]))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        self.command = [str(ROOT / 'vendor/llama.cpp/build/bin/llama-server'),
                        '-m', str(self.model), '--host', '127.0.0.1', '--port', str(port),
                        '-np', '1', '-c', '4096', '-b', '512', '-ub', '512',
                        '-ngl', str(self.gpu_layers), '--no-warmup', '--no-webui',
                        '--cache-ram', '0', '--no-context-shift']
        self.log = open(self.directory / 'server.log', 'w')
        try:
            self.process = subprocess.Popen(self.command, stdout=self.log, stderr=self.log,
                                            env={**os.environ, 'NEURON_CONFIG': str(self.config)})
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'llama-server exited: see {self.directory / "server.log"}')
                try:
                    if self.http('/health')['status'] == 'ok':
                        return self
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.25)
            raise TimeoutError('llama-server startup exceeded 180 seconds')
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log:
            self.log.close()

    def http(self, endpoint, body=None):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = urllib.request.Request(self.url + endpoint, data,
                                         {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if endpoint == '/health' and error.code == 503:
                raise urllib.error.URLError('loading') from error
            raise RuntimeError(f'llama-server HTTP {error.code}: {error.read().decode()}') from error

    def complete(self, prompt, *, steering=(), trace=False, seed=42, temperature=0.7, tokens=64):
        for item in steering:
            if not (0 <= item['layer'] < 32 and 0 <= item['neuron'] < 9216
                    and np.isfinite(item['alpha'])):
                raise ValueError('invalid intervention')
        write_json(self.config, dict(trace=trace, steering=list(steering)))
        self.trace.unlink(missing_ok=True)
        # Explicit empty thinking block makes completion sampling comparable across conditions.
        formatted = ('<|im_start|>user\n' + prompt + '<|im_end|>\n'
                     '<|im_start|>assistant\n<think>\n\n</think>\n\n')
        body = dict(prompt=formatted, n_predict=tokens, temperature=temperature,
                    seed=seed, top_k=40, top_p=0.95, min_p=0.05,
                    cache_prompt=False, return_tokens=True, n_probs=5,
                    stop=['<|im_end|>', '<|im_start|>'])
        response = self.http('/completion', body)
        if response.get('truncated'):
            raise RuntimeError('prompt was truncated')
        parsed = read_trace(self.trace)
        if trace and (set(parsed[0]) != set(range(31)) or set(parsed[1]) != set(range(31))):
            raise RuntimeError('incomplete activation trace; instrumentation is not working')
        for item in steering:
            records = [r for r in parsed[2] if r['layer'] == item['layer'] and r['neuron'] == item['neuron']]
            if not records or not all(np.isclose(r['after'] - r['before'], item['alpha'], atol=1e-5, rtol=1e-4) for r in records):
                raise RuntimeError('steering was not applied as requested')
        self.request_id += 1
        record = dict(id=self.request_id, prompt=prompt, steering=list(steering), request=body, response=response)
        with open(self.directory / 'requests.jsonl', 'a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        return response, parsed


def column_norms(model, coordinates):
    from gguf import GGUFReader
    from gguf.quants import dequantize
    reader = GGUFReader(str(model))
    tensors = {t.name: t for t in reader.tensors}
    result = {}
    for layer in sorted({c['layer'] for c in coordinates}):
        tensor = tensors[f'blk.{layer}.ffn_down.weight']
        if list(tensor.shape) != [9216, 2560]:
            raise ValueError('model is not the supported Qwen3.5-4B dense architecture')
        weights = dequantize(tensor.data, tensor.tensor_type).reshape(2560, 9216)
        for c in coordinates:
            if c['layer'] == layer:
                result[layer, c['neuron']] = float(np.linalg.norm(weights[:, c['neuron']].astype(np.float64)))
    return result
