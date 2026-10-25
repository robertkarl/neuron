from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from .backend import ROOT, Server, column_norms, digest, write_json, verify_build
from .topics import topic_data


def rank_candidates(positive, negative, limit):
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError('need at least two examples per discovery class')
    candidates = []
    for layer in sorted(positive[0]):
        pos = np.stack([x[layer] for x in positive])
        neg = np.stack([x[layer] for x in negative])
        delta = pos.mean(0) - neg.mean(0)
        pooled = np.sqrt((pos.var(0, ddof=1) + neg.var(0, ddof=1)) / 2 + 1e-8)
        effect = delta / pooled
        for neuron in np.argsort(-np.abs(effect), kind='stable')[:limit]:
            candidates.append(dict(layer=layer, neuron=int(neuron), contrast=float(effect[neuron]),
                                   delta=float(delta[neuron]), direction=1 if delta[neuron] > 0 else -1))
    return sorted(candidates, key=lambda c: (-abs(c['contrast']), c['layer'], c['neuron']))[:limit]


def alpha_for(dose, residual_median, column_norm):
    if not all(np.isfinite(x) for x in (dose, residual_median, column_norm)) or min(residual_median, column_norm) <= 0:
        raise ValueError('normalization requires finite positive norms')
    return float(dose * residual_median / column_norm)


def lexical_score(text, terms):
    return float(any(re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text, re.I) for term in terms))


def evidence_decision(baseline, positive, negative, random_control, retention):
    # This is a screening rule, not a statistical significance test.
    lift = positive - baseline
    separation = positive - negative
    margin = positive - random_control
    return dict(status='candidate' if min(lift, separation, margin) >= 0.25 and retention >= 0.75
                else 'no_evidence', lift=lift, sign_separation=separation,
                random_control_margin=margin, unrelated_retention=retention,
                interpretation='lexical screening evidence only; not proof of a unique concept neuron')


def validate_dataset(data):
    for field in ('positive', 'negative', 'tune', 'heldout', 'terms'):
        if not isinstance(data.get(field), list) or not data[field] or not all(isinstance(x, str) and x.strip() for x in data[field]):
            raise ValueError(f'dataset {field} must be a nonempty list of strings')
    if min(len(data['positive']), len(data['negative'])) < 2:
        raise ValueError('need at least two discovery prompts per class')
    groups = [set(data[k]) for k in ('positive', 'negative', 'tune', 'heldout')]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError('discovery, tuning, and held-out prompts must be disjoint')
    if not data.get('unrelated'):
        raise ValueError('dataset requires unrelated [prompt, expected_terms] controls')
    for prompt, terms in data['unrelated']:
        if not isinstance(prompt, str) or not terms or not all(isinstance(x, str) and x for x in terms):
            raise ValueError('invalid unrelated control')
        if any(prompt in group for group in groups):
            raise ValueError('unrelated controls must be disjoint')


def parser():
    p = argparse.ArgumentParser(description='Search actual FFN channels using a private instrumented llama-server.')
    p.add_argument('--model', default='qwen-4B', choices=['qwen-4B'])
    p.add_argument('--topic', required=True)
    p.add_argument('--dataset', type=Path, help='JSON with positive, negative, terms, tune, heldout, unrelated')
    p.add_argument('--candidates', type=int, default=6)
    p.add_argument('--doses', type=float, nargs='+', default=[0.25, 0.5, 1.0])
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--max-tokens', type=int, default=64)
    p.add_argument('--gpu-layers', type=int, default=99)
    p.add_argument('--output', type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.candidates <= 128 or not 1 <= args.max_tokens <= 512:
        raise ValueError('candidates must be 1..128 and max-tokens 1..512')
    if not np.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        raise ValueError('temperature must be 0..2')
    if not all(np.isfinite(x) and 0 < x <= 4 for x in args.doses):
        raise ValueError('doses must be finite and in (0, 4]')
    if len(set(args.seeds)) != len(args.seeds) or not all(0 <= x < 2**32 - 1 for x in args.seeds):
        raise ValueError('seeds must be distinct uint32 values excluding the random-seed sentinel')
    data = json.loads(args.dataset.read_text()) if args.dataset else topic_data(args.topic)
    validate_dataset(data)
    lock = json.loads((ROOT / 'models.lock.json').read_text())
    model = ROOT / 'models' / lock['filename']
    if not model.exists():
        raise RuntimeError('model is missing; run scripts/setup first')
    print('Verifying model checksum...', flush=True)
    model_sha = digest(model)
    if model_sha != lock['sha256']:
        raise RuntimeError('model checksum differs from models.lock.json')
    revision = subprocess.check_output(['git', '-C', str(ROOT / 'vendor/llama.cpp'), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != lock['llama_revision']:
        raise RuntimeError('unexpected llama.cpp revision; run scripts/setup')
    build = verify_build()
    output = args.output or ROOT / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', build=build,
                  environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'neuron').glob('*.py'))}, model=lock, model_sha256=model_sha, upstream_revision=revision,
                  hook_sha256=digest(ROOT / 'backend/neuron-hook.h'),
                  patch_sha256=digest(ROOT / 'patches/llama-server-neuron.patch'),
                  binary_sha256=digest(ROOT / 'vendor/llama.cpp/build/bin/llama-server'),
                  options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  dataset=data, calibration='frozen unsteered median over discovery prompts, first token of each excluded',
                  search_layers=list(range(31)), trials=[])
    write_json(output / 'report.json', report)
    print(f'Artifacts: {output}', flush=True)
    try:
        with Server(model, output, args.gpu_layers) as server:
            report['server_command'] = server.command
            positive, negative, residuals = [], [], {}
            for label, target in [('positive', positive), ('negative', negative)]:
                for i, prompt in enumerate(data[label]):
                    print(f'Profile {label} {i + 1}/{len(data[label])}', flush=True)
                    _, (means, norms, _) = server.complete(prompt, trace=True, tokens=1, temperature=0)
                    target.append(means)
                    for layer, values in norms.items():
                        residuals.setdefault(layer, []).extend(values)
            np.savez_compressed(output / 'activations.npz',
                                **{f'{label}_{layer}': np.stack([v[layer] for v in rows])
                                   for label, rows in [('positive', positive), ('negative', negative)]
                                   for layer in range(31)})
            candidates = rank_candidates(positive, negative, args.candidates)
            medians = {layer: float(np.median(values)) for layer, values in residuals.items()}
            rng = np.random.default_rng(args.seeds[0])
            controls = []
            for c in candidates:
                excluded = {x['neuron'] for x in candidates if x['layer'] == c['layer']}
                neuron = int(rng.choice([n for n in range(9216) if n not in excluded]))
                controls.append(dict(layer=c['layer'], neuron=neuron, direction=c['direction']))
            norms = column_norms(model, candidates + controls)
            for c in candidates + controls:
                c['column_norm'] = norms[c['layer'], c['neuron']]
                c['residual_median'] = medians[c['layer']]
            report.update(candidates=candidates, random_controls=controls, residual_medians=medians)
            write_json(output / 'report.json', report)

            def evaluate(prompts, coordinate=None, dose=0, terms=None):
                steering = [] if coordinate is None else [dict(layer=coordinate['layer'], neuron=coordinate['neuron'],
                    alpha=alpha_for(dose * coordinate['direction'], coordinate['residual_median'], coordinate['column_norm']))]
                samples = []
                for prompt in prompts:
                    for seed in args.seeds:
                        response, _ = server.complete(prompt, steering=steering, seed=seed,
                                                       temperature=args.temperature, tokens=args.max_tokens)
                        text = response['content']
                        samples.append(dict(prompt=prompt, seed=seed, text=text,
                                            score=lexical_score(text, terms or data['terms']),
                                            tokens_predicted=response.get('tokens_predicted'),
                                            stop_type=response.get('stop_type')))
                return dict(mean=float(np.mean([s['score'] for s in samples])), samples=samples,
                            steering=steering, dose=dose)

            baseline = evaluate(data['tune'])
            report['tuning_baseline'] = baseline
            for i, c in enumerate(candidates):
                for dose in sorted(set(args.doses)):
                    print(f'Test {i + 1}/{len(candidates)} L{c["layer"]} N{c["neuron"]} dose +/-{dose}', flush=True)
                    plus = evaluate(data['tune'], c, dose)
                    minus = evaluate(data['tune'], c, -dose)
                    trial = dict(candidate_index=i, dose=dose, positive=plus, negative=minus,
                                 selection_score=plus['mean'] - max(baseline['mean'], minus['mean']))
                    report['trials'].append(trial)
                    write_json(output / 'report.json', report)
            best = max(report['trials'], key=lambda t: t['selection_score'])
            index, dose = best['candidate_index'], best['dose']
            chosen, control = candidates[index], controls[index]
            print(f'Held-out evaluation: L{chosen["layer"]} N{chosen["neuron"]}, dose {dose}', flush=True)
            heldout = dict(baseline=evaluate(data['heldout']), positive=evaluate(data['heldout'], chosen, dose),
                           negative=evaluate(data['heldout'], chosen, -dose),
                           random_positive=evaluate(data['heldout'], control, dose),
                           random_negative=evaluate(data['heldout'], control, -dose))
            unrelated = []
            for prompt, terms in data['unrelated']:
                unrelated.append(dict(baseline=evaluate([prompt], terms=terms),
                                      steered=evaluate([prompt], chosen, dose, terms)))
            retention = float(np.mean([x['steered']['mean'] for x in unrelated]))
            decision = evidence_decision(heldout['baseline']['mean'], heldout['positive']['mean'],
                                         heldout['negative']['mean'],
                                         max(heldout['random_positive']['mean'], heldout['random_negative']['mean']), retention)
            report.update(status=decision['status'], decision=decision, selected=chosen, selected_dose=dose,
                          heldout=heldout, unrelated=unrelated)
            write_json(output / 'report.json', report)
            print(json.dumps(decision, indent=2), flush=True)
            print(f'Report: {output / "report.json"}', flush=True)
        return 0
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'error', error=str(error))
        write_json(output / 'report.json', report)
        raise


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f'find_neuron: {error}', file=sys.stderr)
        sys.exit(1)
