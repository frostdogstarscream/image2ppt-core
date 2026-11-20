"""Host-selected Agent inference credentials and endpoint configuration."""
import os


DEFAULT_DEEPSEEK_MODEL='deepseek-flash'
GO_BASE_URL='https://opencode.ai/inference/go/openai/v1'
DEFAULT_GO_MODEL='hy4-flash-free'
GO_MODEL_ALIASES={
    'hunyuan4':'hy4',
    'hunyuan-4':'hy4',
    'opencode-go/hy4':'hy4',
    'opencode-go/hunyuan4':'hy4',
    'hunyuan4-flash-free':'hy4-flash-free',
    'hunyuan-4-flash-free':'hy4-flash-free',
    'opencode-go/hy4-flash-free':'hy4-flash-free',
    'opencode-go/hunyuan4-flash-free':'hy4-flash-free',
}


def agent_identity(request):
    from ..pipeline import Settings
    provider = request.get('agent_inference_provider') or os.environ.get('IMAGE2PPT_AGENT_PROVIDER', 'moma')
    model = request.get('agent_model') or os.environ.get('IMAGE2PPT_AGENT_MODEL')
    if provider == 'deepseek':
        return provider, model or DEFAULT_DEEPSEEK_MODEL
    if provider == 'opencode_go':
        model = model or DEFAULT_GO_MODEL
        return provider, GO_MODEL_ALIASES.get(model, model)
    model = model or next((os.environ[k].strip() for k in ('PLANNER_MODEL', 'MOMA_MODEL', 'VISION_LLM_MODEL')
                           if os.environ.get(k, '').strip()), Settings().model)
    return provider, model


def model_identity(model):
    """Compare catalog and wire IDs without confusing vendor prefixes with model names."""
    name = (model or '').strip().lower().rsplit('/', 1)[-1]
    if name in ('deepseek-v4-flash', 'deepseek-v4-flash-vision-exp',
                'deepseek-v4.1-flash', 'deepseek-v4.1-flash-1017'):
        return DEFAULT_DEEPSEEK_MODEL
    return GO_MODEL_ALIASES.get(name, name)


def credential_environment(provider):
    """Names only. A dual-slot host exposes only the Agent key to OpenCode's model adapter."""
    host = os.environ.get('IMAGE2PPT_AGENT_PROVIDER')
    agent_key = os.environ.get('IMAGE2PPT_AGENT_API_KEY')
    if 'deepseek' in (provider, host):
        if provider != host:
            raise RuntimeError('DeepSeek Agent credential must be bound to its official provider')
        if not agent_key:
            raise RuntimeError('DeepSeek Agent credential absent')
        if agent_key == os.environ.get('MOMA_API_KEY'):
            raise RuntimeError('DeepSeek execution and MoMA review require distinct credentials')
        return ['IMAGE2PPT_AGENT_API_KEY']
    if agent_key:
        if (provider == 'moma' and os.environ.get('IMAGE2PPT_REVIEW_PROVIDER') == 'moma'
                and agent_key == os.environ.get('MOMA_API_KEY')):
            raise RuntimeError('MoMA execution and review require distinct credentials')
        return ['IMAGE2PPT_AGENT_API_KEY']
    if provider == 'opencode_go':
        return ['OPENCODE_GO_API_KEY']
    if os.environ.get('IMAGE2PPT_REVIEW_PROVIDER') == 'moma':
        raise RuntimeError('separate MoMA Agent credential absent')
    return ['PLANNER_API_KEY', 'MOMA_API_KEY', 'VISION_LLM_API_KEY']


def configure_model(job=None):
    import json
    from dataclasses import replace
    from pathlib import Path
    from ..pipeline import Settings, settings
    from image2ppt_planner.config import DEEPSEEK_BASE_URL, DEFAULT_BASE_URL

    request = {}
    if job is not None:
        file = Path(job)/'request.json'
        if file.is_file():
            request = json.loads(file.read_text('utf8'))
    provider, model = agent_identity(request)
    if provider not in ('moma', 'opencode_go', 'deepseek'):
        raise ValueError('unsupported Agent inference provider')
    names = credential_environment(provider)
    key = next((os.environ[k] for k in names if os.environ.get(k)), None)
    if provider == 'opencode_go' and not key:
        raise RuntimeError('OpenCode Go Agent credential absent')
    if names[0] != 'IMAGE2PPT_AGENT_API_KEY' and provider == 'moma':
        # Legacy MoMA hosts may use planner-specific endpoint settings or the
        # existing OS credential store. Keep all of those settings read-only.
        return provider, replace(settings(), model=model)
    if provider == 'deepseek':
        # The OpenCode adapter and the direct tools share the host's pinned origin and thinking mode.
        return provider, replace(Settings.from_env(), api_key=key, model=model, base_url=DEEPSEEK_BASE_URL,
                                 disable_thinking=not request.get('agent_thinking', False),
                                 max_attempts=1, transport_attempts=1)
    base_url = GO_BASE_URL if provider == 'opencode_go' else os.environ.get('MOMA_BASE_URL') or DEFAULT_BASE_URL
    if not base_url.startswith('https://'):
        raise ValueError('model endpoint requires HTTPS')
    return provider, replace(Settings.from_env(), api_key=key, model=model, base_url=base_url,
                             max_attempts=1, transport_attempts=1)


def provider_id(provider):
    return {'opencode_go': 'image2ppt-go', 'moma': 'image2ppt-moma', 'deepseek': 'image2ppt-deepseek'}[provider]


def planning_actor(job):
    from ..parallel_provider import Planner,Reviewer
    from ..go_provider import GoProvider
    from pathlib import Path
    selected,cfg=configure_model(job)
    if selected!='opencode_go':
        return Planner(cfg)
    class GoAgentPlanner:
        # Reuse the receipted Go transport for planning under this Agent's job.
        ask=GoProvider.ask
        ask_receipted=Reviewer.ask_receipted
        def __init__(self):
            self.key=cfg.api_key
            self.model=cfg.model
            self.base_url=GO_BASE_URL
            self.session=Path(job).name
    return GoAgentPlanner()
