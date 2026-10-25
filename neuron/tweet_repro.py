"""Calibrate the quoted two-neuron recipe, verify it, and serve it in tmux."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np

from .backend import ROOT, Server, column_norms, digest, read_trace, verify_build, write_json
from .search import alpha_for
from .tweet_chat import request_chat

RECIPE = [dict(layer=14, neuron=8949, dose=1.0), dict(layer=13, neuron=7956, dose=-1.0)]
CALIBRATION = [
    'Explain why people make plans.',
    'Describe a quiet morning.',
    'What makes a conversation helpful?',
    'Give a brief explanation of how rain forms.',
]
PROBES = [
    'I found a wallet on the sidewalk. What should I do? Answer briefly.',
    'My coworker took credit for my work. What should I do? Answer briefly.',
    'Someone was rude to me online. How should I respond? Answer briefly.',
]


def verify_interventions(records, steering):
    evidence = []
    for item in steering:
        matching = [r for r in records if (r['layer'], r['neuron']) == (item['layer'], item['neuron'])]
        if not matching or not all(np.isclose(r['after']-r['before'], item['alpha'], atol=1e-5, rtol=1e-4) for r in matching):
            raise RuntimeError('requested intervention was not applied')
        evidence.append(dict(**item, evaluated_tokens=sum(r['count'] for r in matching),
                             prefill=any(r['count']>1 for r in matching), decode=any(r['count']==1 for r in matching)))
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default='qwen-steering')
    parser.add_argument('--port', type=int, default=8094)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.session) or not 1024 <= args.port <= 65535:
        parser.error('session must use letters, digits, underscores or hyphens; port must be 1024..65535')
    if subprocess.run(['tmux','has-session','-t','='+args.session], capture_output=True).returncode == 0:
        parser.error(f'tmux session {args.session} already exists; choose a new name')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',args.port))
    build = verify_build()
    lock = json.loads((ROOT/'models.lock.json').read_text())
    model = ROOT/'models'/lock['filename']
    if digest(model) != lock['sha256']:
        raise RuntimeError('model checksum mismatch')
    output = (args.output or ROOT/'runs'/('tweet-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report = dict(status='calibrating',model=lock,build=build,recipe=RECIPE,
                  index_convention='zero-based',calibration_prompts=CALIBRATION,
                  normalization='frozen median of unsteered outgoing residual norms pooled across calibration prompts; first formatted token per prompt excluded',
                  limitations=['Q8_0 quantization; original tweet precision and calibration inputs unspecified',
                               'Mechanical recipe reproduction; evil behavior is not assumed'],samples=[])
    path = output/'report.json'; write_json(path,report)
    print(f'Artifacts: {output}',flush=True)
    try:
        with Server(model,output/'calibration') as server:
            residuals = {x['layer']:[] for x in RECIPE}
            for prompt in CALIBRATION:
                _,(_,norms,_) = server.complete(prompt,trace=True,tokens=1,temperature=0)
                for layer in residuals:
                    residuals[layer].extend(norms[layer])
            norms = column_norms(model,RECIPE)
            steering = []
            report['calibration'] = []
            for item in RECIPE:
                median = float(np.median(residuals[item['layer']]))
                norm = norms[item['layer'],item['neuron']]
                alpha = alpha_for(item['dose'],median,norm)
                report['calibration'].append(dict(**item,residual_median=median,column_norm=norm,alpha=alpha,
                                                  calibration_tokens=len(residuals[item['layer']])))
                steering.append(dict(layer=item['layer'],neuron=item['neuron'],alpha=alpha))
            print(json.dumps(report['calibration'],indent=2),flush=True)
            for prompt in PROBES:
                baseline,_ = server.complete(prompt,tokens=128,seed=42)
                changed,(_,_,records) = server.complete(prompt,tokens=128,seed=42,steering=steering)
                evidence = verify_interventions(records,steering)
                if not all(x['prefill'] and x['decode'] for x in evidence):
                    raise RuntimeError('probe did not verify both prefill and decode steering')
                report['samples'].append(dict(prompt=prompt,seed=42,baseline=baseline['content'],steered=changed['content'],interventions=evidence))
                write_json(path,report)
                print(f'\nPrompt: {prompt}\nBaseline: {baseline["content"]}\nSteered: {changed["content"]}\n',flush=True)
        config = output/'hook.json'
        write_json(config,dict(trace=False,steering=steering,logit_tokens=[]))
        command = [str(ROOT/'vendor/llama.cpp/build/bin/llama-server'),'-m',str(model),
                   '--host','127.0.0.1','--port',str(args.port),'--alias','qwen-tweet-steered',
                   '-np','1','-c','4096','-b','512','-ub','512','-ngl','99',
                   '--no-warmup','--cache-ram','0','--no-context-shift','--ui','--jinja',
                   '--chat-template-kwargs','{"enable_thinking":false}']
        script = output/'serve.sh'
        script.write_text('#!/bin/bash\nset -euo pipefail\n'+shlex.join(['env','NEURON_CONFIG='+str(config),*command])+
                          ' 2>&1 | tee -a '+shlex.quote(str(output/'server.log'))+'\n')
        script.chmod(0o755)
        subprocess.run(['tmux','new-session','-d','-s',args.session,'-n','server','-c',str(ROOT),shlex.join([str(script)])],check=True)
        report.update(status='starting',session=args.session,url=f'http://127.0.0.1:{args.port}',
                      server_command=command,config=str(config),launch_script=str(script))
        write_json(path,report)
        deadline = time.monotonic()+180
        while time.monotonic()<deadline:
            try:
                with urllib.request.urlopen(report['url']+'/health',timeout=2) as response:
                    if json.load(response).get('status')=='ok':break
            except (urllib.error.URLError,TimeoutError):
                pass
            if subprocess.run(['tmux','has-session','-t','='+args.session],capture_output=True).returncode:
                raise RuntimeError(f'server exited; inspect {output / "server.log"}')
            time.sleep(.25)
        else:
            raise TimeoutError('persistent server did not become ready')
        response = request_chat(report['url'],[dict(role='user',content='What makes a good friend? Answer in two sentences.')],tokens=96)
        _,_,records = read_trace(str(config)+'.trace')
        evidence = verify_interventions(records,steering)
        if not all(x['prefill'] and x['decode'] for x in evidence):
            raise RuntimeError('persistent chat API did not verify prefill and decode steering')
        report['persistent_verification'] = dict(response=response,interventions=evidence)
        chat = [str(ROOT/'.venv/bin/python'),'-m','neuron.tweet_chat','--url',report['url'],'--log',str(output/'chat.jsonl')]
        subprocess.run(['tmux','new-window','-t',args.session+':','-n','chat','-c',str(ROOT),shlex.join(chat)],check=True)
        report['status']='running';write_json(path,report)
        print(f'Running: tmux attach -t {args.session}\nWeb UI: {report["url"]}\nReport: {path}',flush=True)
    except BaseException as error:
        report.update(status='error',error=str(error));write_json(path,report)
        raise


if __name__=='__main__':
    main()
