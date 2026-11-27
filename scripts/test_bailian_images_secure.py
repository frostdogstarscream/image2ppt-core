"""Bounded manual Bailian experiment; credential stays in process memory."""
import argparse
import base64
import getpass
import json
import re
import time
from pathlib import Path
from urllib import request, error, parse


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    key = getpass.getpass('Bailian API key (hidden): ')
    opener = request.build_opener(NoRedirect)
    index = 0

    def safe(value):
        if isinstance(value, str):
            return re.sub(r'https?://[^\s"<>]+', '[URL]', value.replace(key, '[REDACTED]'))
        if isinstance(value, list):
            return [safe(v) for v in value]
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        return value

    print('READY: JSON commands on stdin; quit ends and discards credential.', flush=True)
    try:
        while True:
            line = input()
            if line == 'quit':
                break
            command = json.loads(line)
            base = command.get('base', 'https://dashscope.aliyuncs.com')
            host = parse.urlsplit(base)
            if host.scheme != 'https' or host.netloc not in (
                'dashscope.aliyuncs.com', 'dashscope-intl.aliyuncs.com',
                'dashscope-us.aliyuncs.com'):
                raise ValueError('Only documented official inference hosts are allowed')
            path = command.get('path', '/compatible-mode/v1/models')
            if not path.startswith('/') or path.startswith('//') or '?' in path:
                raise ValueError('Invalid API path')
            body = command.get('body')
            public_body = json.loads(json.dumps(body)) if body else None
            if body and command.get('input_image'):
                source = Path(command['input_image']).resolve()
                encoded = 'data:image/png;base64,' + base64.b64encode(source.read_bytes()).decode()
                content = body['input']['messages'][0]['content']
                for item in content:
                    if item.get('image') == '$INPUT_IMAGE':
                        item['image'] = encoded
            headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
            req = request.Request(base + path, data=json.dumps(body).encode() if body else None, headers=headers)
            start = time.monotonic()
            index += 1
            record = {'base': base, 'path': path, 'request': public_body,
                      'input_image': command.get('input_image')}
            try:
                with opener.open(req, timeout=300) as response:
                    record['http_status'] = response.status
                    data = json.load(response)
            except error.HTTPError as exc:
                record['http_status'] = exc.code
                try:
                    data = json.loads(exc.read(50000))
                except Exception:
                    data = {'error': 'NonJSONHTTPError'}
            except Exception as exc:
                data = {'error': type(exc).__name__}
                record['http_status'] = None
            record['elapsed_s'] = round(time.monotonic() - start, 3)
            record['response'] = safe(data)
            images = []
            for choice in data.get('output', {}).get('choices', []):
                for item in choice.get('message', {}).get('content', []):
                    if isinstance(item.get('image'), str):
                        images.append(item['image'])
            record['downloaded_images'] = []
            for n, url in enumerate(images):
                parsed = parse.urlsplit(url)
                if parsed.scheme != 'https' or not parsed.hostname.endswith('.aliyuncs.com'):
                    raise ValueError('Unexpected image result host')
                # The inference credential is never sent with artifact downloads.
                with opener.open(url, timeout=90) as response:
                    raw = response.read(30 * 1024 * 1024)
                filename = f'{index:02d}-image-{n:02d}.png'
                (output / filename).write_bytes(raw)
                record['downloaded_images'].append(filename)
            (output / f'{index:02d}-response.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
            if isinstance(data.get('data'), list):
                display = {'http_status': record['http_status'], 'elapsed_s': record['elapsed_s'],
                           'model_ids': [m.get('id') for m in data['data'] if isinstance(m, dict)]}
            else:
                display = record
            print(json.dumps(display, ensure_ascii=False), flush=True)
    finally:
        key = None


if __name__ == '__main__':
    main()
