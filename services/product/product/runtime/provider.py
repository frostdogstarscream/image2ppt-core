"""Host-selected Agent and independent reviewer inference credentials and endpoint configuration."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

import httpx

GO_BASE_URL = 'https://opencode.ai/inference/go/openai/v1'
DEFAULT_GO_MODEL = 'hy4-flash-free'
GO_MODEL_ALIASES = {
    'hunyuan4': 'hy4',
    'hunyuan-4': 'hy4',
    'opencode-go/hy4': 'hy4',
    'opencode-go/hunyuan4': 'hy4',
    'hunyuan4-flash-free': 'hy4-flash-free',
    'hunyuan-4-flash-free': 'hy4-flash-free',
    'opencode-go/hy4-flash-free': 'hy4-flash-free',
    'opencode-go/hunyuan4-flash-free': 'hy4-flash-free',
}

DEEPSEEK_BASE_URL = 'https://api.deepseek.com/v1'
DEFAULT_DEEPSEEK_MODEL = 'deepseek-flash'
DEEPSEEK_MODEL_ALIASES = {
    'deepseek-v4.1-flash': 'deepseek-flash',
    'deepseek-v4.1-flash-1017': 'deepseek-flash',
    'deepseek/deepseek-v4.1-flash': 'deepseek-flash',
    'deepseek/deepseek-flash': 'deepseek-flash',
    'deepseek-v4-flash': 'deepseek-flash',
    'deepseek-chat': 'deepseek-flash',
    'deepseek/deepseek-v4-pro': 'deepseek-v4-pro',
}

MOMA_BASE_URL = 'https://moma.cmecloud.cn/v1'
DEFAULT_MOMA_MODEL = 'qwen/qwen3.8-max'
SUPPORTED_PROVIDERS = ('opencode_go', 'deepseek', 'moma')


def _dedupe_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        val = (raw or '').strip()
        if val and val not in seen:
            seen.add(val)
            ordered.append(val)
    return ordered


def moma_api_keys() -> list[str]:
    """Collect configured MoMA keys in priority order without leaking secrets."""
    raw: list[str] = []
    for name in ('MOMA_API_KEY',):
        v = os.environ.get(name, '').strip()
        if v:
            raw.append(v)
    multi = os.environ.get('MOMA_API_KEYS', '')
    if multi:
        raw.extend(part.strip() for part in re.split(r'[,;\r\n]+', multi) if part.strip())
    for name in ('MOMA_API_KEY_BACKUP', 'MOMA_API_KEY_2'):
        v = os.environ.get(name, '').strip()
        if v:
            raw.append(v)
    planner_url = os.environ.get('PLANNER_BASE_URL', '').strip()
    if not planner_url or 'moma.cmecloud.cn' in planner_url:
        pk = os.environ.get('PLANNER_API_KEY', '').strip()
        if pk:
            raw.append(pk)
    return _dedupe_nonempty(raw)


def deepseek_api_key() -> str:
    """Return configured DeepSeek API key, if present."""
    ds = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if ds:
        return ds
    if os.environ.get('IMAGE2PPT_AGENT_PROVIDER', '').strip() == 'deepseek':
        ak = os.environ.get('IMAGE2PPT_AGENT_API_KEY', '').strip()
        if ak:
            return ak
    planner_url = os.environ.get('PLANNER_BASE_URL', '').strip()
    if planner_url.startswith('https://api.deepseek.com'):
        pk = os.environ.get('PLANNER_API_KEY', '').strip()
        if pk:
            return pk
    return ''


def all_secret_env_values() -> list[str]:
    """Return all active API keys / service tokens for log and transcript redaction."""
    candidates: list[str] = []
    for name in (
        'MOMA_API_KEY', 'MOMA_API_KEY_BACKUP', 'MOMA_API_KEY_2',
        'DEEPSEEK_API_KEY', 'PLANNER_API_KEY', 'IMAGE2PPT_AGENT_API_KEY',
        'OPENCODE_GO_API_KEY', 'IMAGE2PPT_OPENCODE_PASSWORD',
        'IMAGE2PPT_PRODUCT_TOKEN', 'IMAGE2PPT_OFFICE_WORKER_TOKEN',
    ):
        val = os.environ.get(name, '').strip()
        if val:
            candidates.append(val)
    multi = os.environ.get('MOMA_API_KEYS', '')
    if multi:
        candidates.extend(part.strip() for part in re.split(r'[,;\r\n]+', multi) if part.strip())
    return _dedupe_nonempty(candidates)


def _read_request(job=None) -> dict:
    if job is not None:
        req_file = Path(job) / 'request.json'
        if req_file.is_file():
            return json.loads(req_file.read_text('utf8'))
    return {}


def inspect_agent_spec(job=None) -> tuple[str, str, list[str], str]:
    """Resolve (provider, base_url, api_keys, model_id) without mutating os.environ."""
    request = _read_request(job)
    provider = request.get('agent_inference_provider') or os.environ.get('IMAGE2PPT_AGENT_PROVIDER', 'moma')
    if provider == 'opencode_go':
        key = os.environ.get('IMAGE2PPT_AGENT_API_KEY') or os.environ.get('OPENCODE_GO_API_KEY') or ''
        raw_model = request.get('agent_model') or os.environ.get('IMAGE2PPT_AGENT_MODEL') or DEFAULT_GO_MODEL
        model = GO_MODEL_ALIASES.get(raw_model, raw_model)
        base_url = os.environ.get('OPENCODE_GO_BASE_URL') or GO_BASE_URL
        return provider, base_url, ([key] if key else []), model
    if provider == 'deepseek':
        key = deepseek_api_key()
        raw_model = request.get('agent_model') or os.environ.get('IMAGE2PPT_AGENT_MODEL') or DEFAULT_DEEPSEEK_MODEL
        model = DEEPSEEK_MODEL_ALIASES.get(raw_model, raw_model)
        base_url = os.environ.get('DEEPSEEK_BASE_URL') or DEEPSEEK_BASE_URL
        return provider, base_url, ([key] if key else []), model
    if provider == 'moma':
        keys = moma_api_keys()
        prev_base = os.environ.get('PLANNER_BASE_URL', '').strip()
        planner_model = os.environ.get('PLANNER_MODEL', '').strip() if (not prev_base or 'moma.cmecloud.cn' in prev_base) else ''
        model = (
            request.get('agent_model')
            or os.environ.get('IMAGE2PPT_AGENT_MODEL')
            or os.environ.get('MOMA_MODEL')
            or planner_model
            or DEFAULT_MOMA_MODEL
        )
        base_url = os.environ.get('MOMA_BASE_URL') or MOMA_BASE_URL
        return provider, base_url, keys, model
    raise ValueError('unsupported Agent inference provider')


def configure_model(job=None):
    provider, base_url, keys, model = inspect_agent_spec(job)
    if provider == 'opencode_go':
        if not keys:
            raise RuntimeError('OpenCode Go Agent credential absent')
        os.environ.update(PLANNER_API_KEY=keys[0], PLANNER_BASE_URL=base_url, PLANNER_MODEL=model)
    elif provider == 'deepseek':
        if not keys:
            raise RuntimeError('DeepSeek Agent credential absent')
        os.environ.update(PLANNER_API_KEY=keys[0], PLANNER_BASE_URL=base_url, PLANNER_MODEL=model)
    else:
        prev_base = os.environ.get('PLANNER_BASE_URL', '').strip()
        if keys:
            os.environ['PLANNER_API_KEY'] = keys[0]
        elif prev_base and 'moma.cmecloud.cn' not in prev_base:
            os.environ.pop('PLANNER_API_KEY', None)
        os.environ['PLANNER_BASE_URL'] = base_url
        os.environ['PLANNER_MODEL'] = model
    from ..pipeline import settings
    return provider, settings()


def provider_id(provider):
    if provider == 'opencode_go':
        return 'image2ppt-go'
    if provider == 'deepseek':
        return 'image2ppt-deepseek'
    return 'image2ppt-moma'


class OpenAICompatReviewer:
    """OpenAI-compatible multimodal client with multi-key failover and credential redaction."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_keys: list[str],
        model: str,
        session: str,
        thinking_mode: str = 'enable_thinking',
    ):
        self.provider = provider
        self.base_url = base_url.rstrip('/')
        self.keys = _dedupe_nonempty(list(api_keys))
        if not self.keys:
            raise RuntimeError(f'{provider} credentials absent')
        self.key = self.keys[0]
        self.model = model
        self.session = str(session)
        self.thinking_mode = thinking_mode
        self.last_usage = None
        self.reported_model = None

    def _redact(self, text: str) -> str:
        return 'credential_redacted' if any(k in text for k in self.keys) else text

    def _post(self, body: dict) -> dict:
        from ..pipeline import PipelineError
        last_error: Exception | None = None
        headers_base = {
            'User-Agent': 'image2ppt-product/1.0',
            'x-opencode-session': self.session,
        }
        for idx, api_key in enumerate(self.keys):
            has_next = idx + 1 < len(self.keys)
            try:
                with httpx.Client(trust_env=False, follow_redirects=False, timeout=httpx.Timeout(300, connect=20)) as client:
                    response = client.post(
                        self.base_url + '/chat/completions',
                        json=body,
                        headers={'Authorization': 'Bearer ' + api_key, **headers_base},
                    )
                if response.status_code != 200:
                    err = PipelineError(f'{self.provider}_http_{response.status_code}')
                    if has_next and (response.status_code in (401, 402, 403, 429) or response.status_code >= 500):
                        last_error = err
                        continue
                    raise err
                data = response.json()
                choice = data['choices'][0]
                content = choice['message']['content']
                if isinstance(content, list):
                    content = ''.join(p.get('text', '') for p in content if isinstance(p, dict))
                if any(k in content for k in self.keys):
                    raise PipelineError('credential_echo_rejected')
                if choice.get('finish_reason') not in ('stop', None):
                    raise ValueError(f'{self.provider} response truncated')
                self.last_usage = data.get('usage')
                self.reported_model = data.get('model')
                return {'data': data, 'content': content, 'finish_reason': choice.get('finish_reason')}
            except httpx.HTTPError as exc:
                err = PipelineError(f'{self.provider}_{type(exc).__name__}')
                err.transport_detail = self._redact(str(exc))[:500]
                last_error = err
                if has_next:
                    continue
                raise err from None
        raise last_error or PipelineError(f'{self.provider}_request_failed')

    def _build_body(self, system: str, context: dict, images: list, contract) -> dict:
        from ..pipeline import image_part
        text = json.dumps({**context, 'output_schema': contract.model_json_schema()}, ensure_ascii=False)
        sys_prompt = (
            system
            if 'json' in system.lower()
            else (system.rstrip() + ' Respond in valid JSON matching output_schema.')
        )
        content = [{'type': 'text', 'text': text}, *[image_part(p) for p in images]] if images else text
        body = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': content},
            ],
            'temperature': 0,
            'stream': False,
            'max_tokens': 16000,
            'response_format': {'type': 'json_object'},
        }
        if self.thinking_mode == 'thinking':
            body['thinking'] = {'type': 'disabled'}
        elif self.thinking_mode == 'enable_thinking':
            body['enable_thinking'] = False
        return body

    def ask(self, system, context, images, contract):
        from image2ppt_planner.client import strip_code_fence
        res = self._post(self._build_body(system, context, images, contract))
        return json.loads(strip_code_fence(res['content']))

    def ask_receipted(self, system, context, images, contract, receipt):
        from ..pipeline import PipelineError
        from ..parallel_provider import parse_reply, save_reply
        receipt_path = Path(receipt)
        if receipt_path.exists():
            return parse_reply(json.loads(receipt_path.read_text(encoding='utf-8')))
        res = self._post(self._build_body(system, context, images, contract))
        raw = res['data']
        data = {
            'content': res['content'],
            'finish_reason': res['finish_reason'],
            'usage': self.last_usage,
            'raw_response': raw,
            'model': raw.get('model', self.model),
        }
        encoded = json.dumps(data, ensure_ascii=False)
        if any(k in encoded for k in self.keys):
            raise PipelineError('credential_echo_rejected')
        save_reply(receipt_path, data, self.keys[0])
        return parse_reply(data)


def resolve_review_config(job=None, *, mutate_env: bool = False) -> tuple[str, str, list[str], str]:
    """Select independent reviewer (provider, base_url, api_keys, model) distinct from the builder Agent."""
    if job and (Path(job)/'request.json').is_file():
        request=json.loads((Path(job)/'request.json').read_text('utf8'))
        if request.get('orchestrator')=='editable_reconstruction':
            provider,base_url,keys,model=inspect_agent_spec(job)
            if provider!='moma' or not keys:
                raise ValueError('direct visual review requires configured MoMA')
            return provider,base_url,keys,model
    if mutate_env:
        agent_provider, cfg = configure_model(job)
        agent_model = cfg.model
    else:
        agent_provider, _, _, agent_model = inspect_agent_spec(job)

    explicit_prov = os.environ.get('IMAGE2PPT_REVIEW_PROVIDER', '').strip()
    explicit_model = os.environ.get('IMAGE2PPT_REVIEW_MODEL', '').strip()

    if explicit_prov:
        if explicit_prov not in SUPPORTED_PROVIDERS:
            raise ValueError('unsupported independent review provider')
        rev_prov = explicit_prov
        if rev_prov == 'moma':
            rev_keys = moma_api_keys()
            rev_base = os.environ.get('MOMA_BASE_URL') or MOMA_BASE_URL
            rev_model = explicit_model or DEFAULT_MOMA_MODEL
        elif rev_prov == 'deepseek':
            k = deepseek_api_key()
            rev_keys = [k] if k else []
            rev_base = os.environ.get('DEEPSEEK_BASE_URL') or DEEPSEEK_BASE_URL
            raw_m = explicit_model or DEFAULT_DEEPSEEK_MODEL
            rev_model = DEEPSEEK_MODEL_ALIASES.get(raw_m, raw_m)
        else:
            k = os.environ.get('OPENCODE_GO_API_KEY') or (os.environ.get('IMAGE2PPT_AGENT_API_KEY') if agent_provider == 'opencode_go' else '')
            rev_keys = [k] if k else []
            rev_base = os.environ.get('OPENCODE_GO_BASE_URL') or GO_BASE_URL
            raw_m = explicit_model or 'hy4'
            rev_model = GO_MODEL_ALIASES.get(raw_m, raw_m)
    else:
        ds_k = deepseek_api_key()
        mm_keys = moma_api_keys()
        go_k = os.environ.get('OPENCODE_GO_API_KEY') or (os.environ.get('IMAGE2PPT_AGENT_API_KEY') if agent_provider == 'opencode_go' else '')
        if agent_provider == 'deepseek' and mm_keys:
            rev_prov = 'moma'
            rev_base = os.environ.get('MOMA_BASE_URL') or MOMA_BASE_URL
            rev_keys = mm_keys
            rev_model = explicit_model or DEFAULT_MOMA_MODEL
        elif agent_provider == 'moma' and ds_k:
            rev_prov = 'deepseek'
            rev_base = os.environ.get('DEEPSEEK_BASE_URL') or DEEPSEEK_BASE_URL
            rev_keys = [ds_k]
            raw_m = explicit_model or DEFAULT_DEEPSEEK_MODEL
            rev_model = DEEPSEEK_MODEL_ALIASES.get(raw_m, raw_m)
        elif agent_provider == 'opencode_go' and go_k:
            rev_prov = 'opencode_go'
            rev_base = os.environ.get('OPENCODE_GO_BASE_URL') or GO_BASE_URL
            rev_keys = [go_k]
            raw_m = explicit_model or 'hy4'
            rev_model = GO_MODEL_ALIASES.get(raw_m, raw_m)
        elif agent_provider == 'moma' and mm_keys:
            rev_prov = 'moma'
            rev_base = os.environ.get('MOMA_BASE_URL') or MOMA_BASE_URL
            rev_keys = mm_keys
            rev_model = explicit_model or (DEFAULT_MOMA_MODEL if agent_model != DEFAULT_MOMA_MODEL else 'deepseek-v4.1-flash')
        else:
            raise ValueError('independent reviewer credentials unavailable')

    if not rev_keys:
        raise ValueError('independent reviewer credentials unavailable')
    if rev_prov == agent_provider and rev_model == agent_model:
        raise ValueError('independent reviewer model must differ from builder agent')
    return rev_prov, rev_base, rev_keys, rev_model


def review_actor(job, session: str):
    prov, base_url, keys, model = resolve_review_config(job, mutate_env=False)
    thinking_mode = 'thinking' if prov in ('deepseek', 'opencode_go') else 'enable_thinking'
    return prov, model, OpenAICompatReviewer(
        provider=prov,
        base_url=base_url,
        api_keys=keys,
        model=model,
        session=session,
        thinking_mode=thinking_mode,
    )


def planning_actor(job):
    from ..parallel_provider import Planner, Reviewer
    from ..go_provider import GoProvider
    selected, cfg = configure_model(job)
    if selected == 'opencode_go':
        class GoAgentPlanner:
            # Reuse the receipted Go transport for planning under this Agent's job.
            ask = GoProvider.ask
            ask_receipted = Reviewer.ask_receipted

            def __init__(self):
                self.key = cfg.api_key
                self.model = cfg.model
                self.base_url = GO_BASE_URL
                self.session = Path(job).name

        return GoAgentPlanner()
    if selected == 'deepseek':
        return OpenAICompatReviewer(
            provider='deepseek',
            base_url=cfg.base_url,
            api_keys=[cfg.api_key],
            model=cfg.model,
            session=Path(job).name,
            thinking_mode='thinking',
        )
    keys = moma_api_keys()
    if len(keys) > 1:
        return OpenAICompatReviewer(
            provider='moma',
            base_url=cfg.base_url,
            api_keys=keys,
            model=cfg.model,
            session=Path(job).name,
            thinking_mode='enable_thinking',
        )
    return Planner(cfg)
