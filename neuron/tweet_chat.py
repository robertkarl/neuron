"""Minimal interactive chat client for the local tweet reproduction server."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request


def request_chat(url, messages, tokens=256):
    body = dict(model='qwen-tweet-steered', messages=messages, max_tokens=tokens,
                temperature=0.7, top_k=40, top_p=0.95, min_p=0.05, seed=42,
                chat_template_kwargs={'enable_thinking': False}, stream=False)
    request = urllib.request.Request(url.rstrip('/')+'/v1/chat/completions',
                                     json.dumps(body).encode(), {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    print('Qwen3.5-4B: L14 N8949 +1; L13 N7956 -1 (normalized, frozen calibration).')
    print('No persona/system prompt. /clear starts a new conversation; /quit exits chat.')
    print(f'Server: {args.url}\n')
    messages = []
    while True:
        try:
            prompt = input('You> ').strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if prompt == '/quit':
            break
        if prompt == '/clear':
            messages.clear(); print('Conversation cleared.'); continue
        if not prompt:
            continue
        pending = messages + [dict(role='user', content=prompt)]
        try:
            response = request_chat(args.url, pending)
            content = response['choices'][0]['message']['content'] or ''
        except (RuntimeError, OSError, KeyError, KeyboardInterrupt) as error:
            print(f'Chat request failed: {error}. Use /clear if the context is full.'); continue
        messages = pending + [dict(role='assistant', content=content)]
        with args.log.open('a') as log:
            log.write(json.dumps(dict(messages=messages, response=response))+'\n')
        print(f'\nQwen> {content}\n')


if __name__ == '__main__':
    main()
