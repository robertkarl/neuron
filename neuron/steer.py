"""Replay one of the measured household-neuron interventions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .backend import ROOT, Server, digest, verify_build, write_json
from .search import alpha_for


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--item', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--registry', type=Path, default=ROOT/'data/household-neurons.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-tokens', type=int, default=96)
    parser.add_argument('--baseline', action='store_true', help='Generate without intervention for comparison')
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32 - 1 or not 1 <= args.max_tokens <= 512:
        parser.error('seed must be a fixed uint32 value; max-tokens must be 1..512')
    registry = json.loads(args.registry.read_text())
    if args.item not in registry['neurons']:
        parser.error('item must be one of: ' + ', '.join(registry['neurons']))
    entry = registry['neurons'][args.item]
    lock = json.loads((ROOT/'models.lock.json').read_text())
    if registry['model'] != lock:
        raise ValueError('registry belongs to a different model')
    build = verify_build()
    if registry['hook_sha256'] != build['hook']:
        raise ValueError('registry was measured with different instrumentation')
    model = ROOT/'models'/lock['filename']
    if digest(model) != lock['sha256']:
        raise ValueError('model checksum mismatch')
    output = ROOT/'runs'/('steer-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    alpha = alpha_for(entry['signed_dose'], entry['residual_median'], entry['column_norm'])
    steering = [] if args.baseline else [dict(layer=entry['layer'], neuron=entry['neuron'], alpha=alpha)]
    with Server(model, output) as server:
        response, _ = server.complete(args.prompt, steering=steering, seed=args.seed, tokens=args.max_tokens)
        write_json(output/'result.json', dict(item=args.item, intervention=steering, registry_sha256=digest(args.registry),
                                             prompt=args.prompt, response=response))
        print(response['content'])
        print(f'Artifacts: {output}', file=sys.stderr)


if __name__ == '__main__':
    main()
