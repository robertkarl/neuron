"""Weight-ranked, causally verified household-token neuron search."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .backend import ROOT, Server, digest, verify_build, write_json
from .search import alpha_for

ITEMS = ['chair', 'table', 'lamp', 'plate', 'cup', 'bed', 'door', 'clock', 'mirror', 'fork']
TUNE = [
    ('Choose any ordinary household object. Complete the sentence with a single noun.', 'I picked a'),
    ('Imagine a home being equipped. Name an object someone might bring.', 'They brought a'),
    ('Think of something that could be included in a home inventory.', 'The inventory includes a'),
]
CALIBRATION = [
    'Describe the purpose of a home inventory in one sentence.',
    'What do people consider when furnishing a room?',
    'Explain why people organize their belongings.',
    'Describe a typical quiet evening at home.',
]
CONFIRM = [
    ('Pick an ordinary physical object that someone might own. Finish the sentence.', 'The person owns a'),
    ('Name one thing a visitor could notice inside a house.', 'The visitor noticed a'),
    ('Imagine a photograph of an interior. Choose one object to describe.', 'The photograph shows a'),
    ('Finish this sentence about an item in a furnished apartment.', 'In the apartment there is a'),
    ('Write the name of a commonplace object for a vocabulary exercise.', 'The word is'),
    ('Choose one inanimate thing to include in a drawing of a home.', 'I will draw a'),
    ('Think of a belonging that somebody might replace after moving.', 'They replaced a'),
    ('Give an example of something a person could mention in a home tour.', 'They pointed out a'),
    ('Finish a sentence about a thing found indoors.', 'Indoors I found a'),
    ('Invent one short detail about an ordinary household.', 'The household has a'),
]
FREE = [
    'Name one ordinary household object and explain what people use it for in one sentence.',
    'Describe one object you might notice when visiting a home. Choose the object yourself.',
    'Write a short sentence about a useful physical object, choosing any example you like.',
    'Invent a one-sentence scene taking place inside a home.',
]
OFF_TOPIC = [
    ('What is 7 plus 8? Answer with just the number.', '15'),
    ('What is the capital of France? Answer with just the city.', 'paris'),
    ('Translate the English word cat into Spanish. Answer with one word.', 'gato'),
    ('What is the opposite of hot? Answer with one word.', 'cold'),
]


def weight_candidates(model, token_ids, *, layers=range(20, 32), per_item=24):
    """A direct unembedding projection proposes hypotheses; it is not causal evidence."""
    from gguf import GGUFReader
    from gguf.quants import dequantize
    tensors = {t.name: t for t in GGUFReader(str(model)).tensors}
    embedding = tensors.get('output.weight', tensors['token_embd.weight'])
    rows = np.stack([dequantize(embedding.data[t:t+1], embedding.tensor_type).reshape(-1) for t in token_ids])
    norm = tensors['output_norm.weight']
    output_norm = dequantize(norm.data, norm.tensor_type).reshape(-1)
    directions = (rows - (rows.sum(0) - rows) / (len(rows) - 1)) * output_norm
    result = [[] for _ in token_ids]
    for layer in layers:
        tensor = tensors[f'blk.{layer}.ffn_down.weight']
        weights = dequantize(tensor.data, tensor.tensor_type).reshape(2560, 9216)
        norms = np.linalg.norm(weights.astype(np.float64), axis=0)
        projections = directions @ weights / np.maximum(norms, 1e-12)
        for i in range(len(token_ids)):
            for neuron in np.argsort(-np.abs(projections[i]), kind='stable')[:6]:
                score = float(projections[i, neuron])
                result[i].append(dict(layer=layer, neuron=int(neuron), direction=1 if score > 0 else -1,
                                      projection=score, column_norm=float(norms[neuron])))
    return [sorted(rows, key=lambda x: -abs(x['projection']))[:per_item] for rows in result]


def summarize_effect(baseline, positive, negative, controls, target):
    baseline, positive, negative = map(lambda x: np.asarray(x, dtype=float), (baseline, positive, negative))
    if baseline.shape != positive.shape or baseline.shape != negative.shape or baseline.ndim != 2:
        raise ValueError('probability matrices must have equal prompt-by-item shape')
    lifts = positive - baseline
    target_lifts = lifts[:, target]
    competing = np.delete(lifts, target, axis=1)
    random_lifts = [float(np.median(np.asarray(control)[:, target] - baseline[:, target])) for control in controls]
    statistics = dict(median_logprob_lift=float(np.median(target_lifts)),
                      fraction_lift_at_least_one=float(np.mean(target_lifts >= 1)),
                      median_sign_separation=float(np.median(positive[:, target] - negative[:, target])),
                      specificity_margin=float(np.median(target_lifts - np.median(competing, axis=1))),
                      random_control_lifts=random_lifts,
                      random_control_margin=float(np.median(target_lifts) - max(random_lifts)),
                      baseline_probability=float(np.mean(np.exp(baseline[:, target]))),
                      steered_probability=float(np.mean(np.exp(positive[:, target]))))
    passed = (statistics['median_logprob_lift'] >= 1.5
              and statistics['fraction_lift_at_least_one'] >= 0.8
              and statistics['median_sign_separation'] >= 1.5
              and statistics['specificity_margin'] >= 0.5
              and statistics['random_control_margin'] >= 0.5)
    return dict(status='accepted_token_steering' if passed else 'rejected', **statistics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--doses', type=float, nargs='+', default=[0.25, 0.5, 1.0])
    args = parser.parse_args(argv)
    if not 1 <= args.candidates <= 72 or not all(np.isfinite(x) and 0 < x <= 2 for x in args.doses):
        parser.error('candidates must be 1..72; doses must be finite in (0,2]')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((ROOT / 'models.lock.json').read_text())
    model = ROOT / 'models' / lock['filename']
    build = verify_build()
    if digest(model) != lock['sha256']:
        raise RuntimeError('model checksum mismatch')
    report = dict(status='running', created=datetime.now(timezone.utc).isoformat(), model=lock, build=build,
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT/'neuron').glob('*.py'))},
                  environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
                  items=ITEMS, tuning_prompts=TUNE, confirmation_prompts=CONFIRM, calibration_prompts=CALIBRATION,
                  free_prompts=FREE, off_topic_prompts=OFF_TOPIC, candidates={}, trials={}, results={},
                  interpretation='single-channel household-token steering, not proof of monosemantic concept neurons')
    path = output/'report.json'
    write_json(path, report)
    try:
        with Server(model, output) as server:
            tokenization = {item: {text: server.http('/tokenize', dict(content=text, add_special=False))['tokens']
                                  for text in [item, ' '+item]} for item in ITEMS}
            if any(len(ids) != 1 for variants in tokenization.values() for ids in variants.values()):
                raise RuntimeError('an item is not a single token')
            token_ids = [tokenization[x][' '+x][0] for x in ITEMS]
            bare_ids = [tokenization[x][x][0] for x in ITEMS]
            report.update(tokenization=tokenization, measured_token_ids=token_ids)
            print('Tokenization verified for all ten items.', flush=True)
            medians = {}
            for prompt in CALIBRATION:
                _, (_, norms, _) = server.complete(prompt, trace=True, tokens=1, temperature=0)
                for layer, values in norms.items():
                    medians.setdefault(layer, []).extend(values)
            medians = {layer: float(np.median(v)) for layer, v in medians.items()}
            report['residual_medians'] = medians
            report['calibration'] = 'layers 0-30: unsteered tokens excluding first; layer 31: last-token outgoing residuals due upstream crop'
            print('Ranking weight projections across layers 20-31...', flush=True)
            candidates = weight_candidates(model, token_ids, per_item=args.candidates)
            for i, rows in enumerate(candidates):
                for c in rows:
                    c['residual_median'] = medians[c['layer']]
                report['candidates'][ITEMS[i]] = rows
            write_json(path, report)

            def evaluate(prompts, c=None, dose=0):
                steering = [] if c is None else [dict(layer=c['layer'], neuron=c['neuron'],
                    alpha=alpha_for(dose*c['direction'], c['residual_median'], c['column_norm']))]
                matrix = []
                for prompt, prefix in prompts:
                    response, _ = server.complete(prompt, assistant_prefix=prefix, logit_tokens=token_ids,
                                                  steering=steering, tokens=1, temperature=0)
                    values = response['neuron_logprobs'][-1]
                    matrix.append([values[t] for t in token_ids])
                return matrix

            tune_base = np.asarray(evaluate(TUNE))
            report['tuning_baseline'] = tune_base.tolist()
            selections = []
            for i, item in enumerate(ITEMS):
                report['trials'][item] = []
                for j, c in enumerate(candidates[i]):
                    # Screen with one moderate dose, both signs, before refining the best channels.
                    for direction in (1, -1):
                        dose = min(args.doses) * direction
                        values = np.asarray(evaluate(TUNE, c, dose))
                        lifts = values - tune_base
                        score = float(np.median(lifts[:, i]) - max(0, np.median(np.delete(lifts, i, axis=1))))
                        report['trials'][item].append(dict(candidate=j, dose=dose, values=values.tolist(), score=score))
                    if (j + 1) % 4 == 0:
                        best = max(report['trials'][item], key=lambda x:x['score'])
                        print(f'{item}: screened {j+1}/{len(candidates[i])}; best selective lift {best["score"]:.2f} nats', flush=True)
                        write_json(path, report)
                best = max(report['trials'][item], key=lambda x:x['score'])
                c = candidates[i][best['candidate']]
                for magnitude in sorted(set(args.doses)):
                    dose = magnitude * (1 if best['dose'] > 0 else -1)
                    values = np.asarray(evaluate(TUNE, c, dose))
                    lifts = values - tune_base
                    score = float(np.median(lifts[:, i]) - max(0, np.median(np.delete(lifts, i, axis=1))))
                    report['trials'][item].append(dict(candidate=best['candidate'], dose=dose, values=values.tolist(), score=score))
                best = max(report['trials'][item], key=lambda x:x['score'])
                selections.append(dict(coordinate=candidates[i][best['candidate']], dose=best['dose'], tuning_score=best['score']))
                print(f'{item}: selected {selections[-1]}', flush=True)
                write_json(path, report)
            report['frozen_selections'] = selections
            write_json(output/'frozen-selections.json', selections)
            write_json(path, report)
            print('Selections frozen. Beginning untouched confirmation prompts.', flush=True)
            baseline = evaluate(CONFIRM)
            report['confirmation_baseline'] = baseline
            from .backend import column_norms
            for i, item in enumerate(ITEMS):
                c, dose = selections[i]['coordinate'], selections[i]['dose']
                positive = evaluate(CONFIRM, c, dose)
                negative = evaluate(CONFIRM, c, -dose)
                rng = np.random.default_rng(1000+i)
                excluded = {x['neuron'] for x in candidates[i] if x['layer'] == c['layer']}
                indices = rng.choice([n for n in range(9216) if n not in excluded], 3, replace=False)
                controls = [dict(layer=c['layer'], neuron=int(n), direction=c['direction'], residual_median=c['residual_median']) for n in indices]
                norms = column_norms(model, controls)
                random_values = []
                for ctrl in controls:
                    ctrl['column_norm'] = norms[ctrl['layer'], ctrl['neuron']]
                    random_values.append(evaluate(CONFIRM, ctrl, dose))
                result = summarize_effect(baseline, positive, negative, random_values, i)
                result.update(coordinate=c, dose=dose, alpha=alpha_for(dose*c['direction'],c['residual_median'],c['column_norm']),
                              positive=positive, negative=negative, controls=controls, random_values=random_values)
                steering = [dict(layer=c['layer'], neuron=c['neuron'], alpha=result['alpha'])]
                result['free_generations'] = []
                for prompt in FREE:
                    for seed in [42, 43]:
                        unsteered, _ = server.complete(prompt, seed=seed, tokens=48)
                        steered, _ = server.complete(prompt, seed=seed, tokens=48, steering=steering)
                        result['free_generations'].append(dict(prompt=prompt,seed=seed,baseline=unsteered['content'],steered=steered['content']))
                result['off_topic'] = []
                for prompt, expected in OFF_TOPIC:
                    unsteered, _ = server.complete(prompt, temperature=0, tokens=12)
                    steered, _ = server.complete(prompt, temperature=0, tokens=12, steering=steering)
                    result['off_topic'].append(dict(prompt=prompt, expected=expected, baseline=unsteered['content'], steered=steered['content'],
                                                    correct=steered['content'].strip().strip('.').lower()==expected))
                report['results'][item] = result
                write_json(path, report)
                print(f'{item}: {result["status"]}, lift {result["median_logprob_lift"]:.2f}, specificity {result["specificity_margin"]:.2f}, random margin {result["random_control_margin"]:.2f}', flush=True)
            report['accepted_count'] = sum(x['status']=='accepted_token_steering' for x in report['results'].values())
            report['status'] = 'complete' if report['accepted_count']==10 else 'needs_iteration'
            write_json(path, report)
            print(f'Accepted {report["accepted_count"]}/10. Report: {path}', flush=True)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'error', error=str(error))
        write_json(path, report)
        raise


if __name__ == '__main__':
    main()
