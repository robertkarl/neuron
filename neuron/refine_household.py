"""Refine fixed household neurons to doses that preserve unrelated task answers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .backend import ROOT, Server, column_norms, digest, verify_build, write_json
from .household import ITEMS, TUNE, summarize_effect
from .search import alpha_for

DOSES = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
TUNE_TASKS = [
    ('What is 3 times 4? Answer with just the number.', '12'),
    ('What is the capital of Japan? Answer with just the city name.', 'tokyo'),
    ('Translate dog into French. Answer with just one word.', 'chien'),
]
FRESH_CONFIRM = [
    ('Imagine a real-estate listing photograph. Name one object that could appear.', 'In the picture is a'),
    ('Write a noun for a tangible item someone uses indoors.', 'My noun is'),
    ('Choose an item that could belong to a family living in an apartment.', 'The family owns a'),
    ('Suppose someone is describing their possessions. Supply an ordinary item.', 'One possession is a'),
    ('Finish a short example sentence about an inanimate object.', 'The object was a'),
    ('Invent a small detail for a story set in an ordinary dwelling.', 'Beside them was a'),
    ('Choose one familiar object for an illustrated household glossary.', 'I chose a'),
    ('Name one object without giving a brand or explaining it.', 'It is a'),
    ('Complete a simple sentence about a thing someone spotted inside.', 'She spotted a'),
    ('Picture a room containing everyday belongings. Mention one thing.', 'There was a'),
]
FRESH_FREE = [
    'Choose one common object in a home and write one sentence explaining its use.',
    'Invent a brief scene indoors that mentions one specific physical object.',
    'Give an example of an ordinary household possession and describe it in a sentence.',
    'What is an example of an everyday object? Explain its purpose briefly.',
]
FRESH_TASKS = [
    ('What is 6 times 7? Respond with the number only.', '42'),
    ('Name the capital of Germany. Respond with the city only.', 'berlin'),
    ('Translate red into French. Respond with one word only.', 'rouge'),
    ('Give the opposite of up. Respond with one word only.', 'down'),
]


def answer_correct(text, expected):
    return text.strip().strip('.').lower() == expected


def choose_dose(trials):
    passing = [t for t in trials if t['task_correct'] == len(TUNE_TASKS)
               and t['median_lift'] >= 2 and t['fraction_lift_at_least_one'] == 1
               and t['specificity'] >= .5]
    return max(passing, key=lambda t:t['median_lift'], default=None)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    source=json.loads(args.source.read_text())
    selections=source['frozen_selections']
    if len(selections)!=len(ITEMS) or source['items']!=ITEMS:
        raise ValueError('source must contain frozen selections for all ten items')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    lock=json.loads((ROOT/'models.lock.json').read_text());model=ROOT/'models'/lock['filename']
    if source['model']!=lock or digest(model)!=lock['sha256']:
        raise ValueError('model identity changed')
    token_ids=source['measured_token_ids']
    report=dict(status='running',model=lock,build=verify_build(),source_report=str(args.source.resolve()),source_sha256=digest(args.source),
                source_code_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'neuron').glob('*.py'))},
                environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),
                tokenization=source['tokenization'],items=ITEMS,measured_token_ids=token_ids,
                calibration=source['calibration'],residual_medians=source['residual_medians'],
                doses=DOSES,tune_prompts=TUNE,tune_tasks=TUNE_TASKS,
                confirmation_prompts=FRESH_CONFIRM,free_prompts=FRESH_FREE,off_topic_prompts=FRESH_TASKS,
                trials={},frozen_selections=[],results={},
                interpretation='causal single-channel token steering; concept exclusivity not established')
    path=output/'report.json';write_json(path,report)
    try:
        with Server(model,output) as server:
            def steering(c,dose):
                return [dict(layer=c['layer'],neuron=c['neuron'],alpha=alpha_for(dose*c['direction'],c['residual_median'],c['column_norm']))]
            def evaluate(prompts,c=None,dose=0):
                matrix=[]
                for prompt,prefix in prompts:
                    r,_=server.complete(prompt,assistant_prefix=prefix,logit_tokens=token_ids,tokens=1,temperature=0,
                                        steering=[] if c is None else steering(c,dose))
                    matrix.append([r['neuron_logprobs'][-1][t] for t in token_ids])
                return np.asarray(matrix)
            baseline=evaluate(TUNE);report['tune_baseline']=baseline.tolist()
            for i,item in enumerate(ITEMS):
                c=selections[i]['coordinate'];sign=1 if selections[i]['dose']>0 else -1
                report['trials'][item]=[]
                for magnitude in DOSES:
                    dose=sign*magnitude;values=evaluate(TUNE,c,dose);lifts=values-baseline
                    tasks=[]
                    for prompt,expected in TUNE_TASKS:
                        r,_=server.complete(prompt,steering=steering(c,dose),tokens=12,temperature=0)
                        tasks.append(dict(prompt=prompt,expected=expected,text=r['content'],correct=answer_correct(r['content'],expected)))
                    trial=dict(dose=dose,values=values.tolist(),median_lift=float(np.median(lifts[:,i])),
                               fraction_lift_at_least_one=float(np.mean(lifts[:,i]>=1)),
                               specificity=float(np.median(lifts[:,i]-np.median(np.delete(lifts,i,axis=1),axis=1))),
                               tasks=tasks,task_correct=sum(x['correct'] for x in tasks))
                    report['trials'][item].append(trial)
                selected=choose_dose(report['trials'][item])
                report['frozen_selections'].append(None if selected is None else dict(coordinate=c,dose=selected['dose']))
                print(item,'dose',selected['dose'] if selected else 'NONE','trials',[(t['dose'],round(t['median_lift'],2),t['task_correct']) for t in report['trials'][item]],flush=True)
                write_json(path,report)
            write_json(output/'frozen-selections.json',report['frozen_selections'])
            print('Low-dose selections frozen; starting fresh confirmation.',flush=True)
            base=evaluate(FRESH_CONFIRM);report['confirmation_baseline']=base.tolist()
            baseline_tasks=[]
            for prompt,expected in FRESH_TASKS:
                r,_=server.complete(prompt,tokens=12,temperature=0)
                baseline_tasks.append(dict(prompt=prompt,expected=expected,text=r['content'],correct=answer_correct(r['content'],expected)))
            report['baseline_tasks']=baseline_tasks
            for i,item in enumerate(ITEMS):
                selected=report['frozen_selections'][i]
                if selected is None:
                    report['results'][item]=dict(status='no_preserving_dose');continue
                c,dose=selected['coordinate'],selected['dose']
                plus=evaluate(FRESH_CONFIRM,c,dose);minus=evaluate(FRESH_CONFIRM,c,-dose)
                rng=np.random.default_rng(2000+i)
                indices=rng.choice([n for n in range(9216) if n!=c['neuron']],3,replace=False)
                controls=[dict(layer=c['layer'],neuron=int(n),direction=c['direction'],residual_median=c['residual_median']) for n in indices]
                norms=column_norms(model,controls);control_values=[]
                for ctrl in controls:
                    ctrl['column_norm']=norms[ctrl['layer'],ctrl['neuron']]
                    control_values.append(evaluate(FRESH_CONFIRM,ctrl,dose).tolist())
                result=summarize_effect(base,plus,minus,control_values,i)
                result.update(coordinate=c,dose=dose,alpha=steering(c,dose)[0]['alpha'],positive=plus.tolist(),negative=minus.tolist(),
                              controls=controls,random_values=control_values)
                result['off_topic']=[]
                for prompt,expected in FRESH_TASKS:
                    r,_=server.complete(prompt,tokens=12,temperature=0,steering=steering(c,dose))
                    result['off_topic'].append(dict(prompt=prompt,expected=expected,text=r['content'],correct=answer_correct(r['content'],expected)))
                result['free_generations']=[]
                for prompt in FRESH_FREE:
                    for seed in [51,52]:
                        unsteered,_=server.complete(prompt,seed=seed,tokens=48)
                        steered,_=server.complete(prompt,seed=seed,tokens=48,steering=steering(c,dose))
                        result['free_generations'].append(dict(prompt=prompt,seed=seed,baseline=unsteered['content'],steered=steered['content']))
                result['task_correct']=sum(x['correct'] for x in result['off_topic'])
                if result['task_correct']<3:
                    result['status']='rejected_task_damage'
                report['results'][item]=result;write_json(path,report)
                print(item,result['status'],'lift',round(result['median_logprob_lift'],2),'tasks',result['task_correct'],flush=True)
            report['accepted_count']=sum(x['status']=='accepted_token_steering' for x in report['results'].values())
            report['status']='complete' if report['accepted_count']==10 else 'needs_iteration'
            write_json(path,report);print('Accepted',report['accepted_count'],'/ 10',flush=True)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',error=str(error));write_json(path,report);raise


if __name__=='__main__':
    main()
