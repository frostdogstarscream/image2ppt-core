"""Host-selected Agent inference credentials and endpoint configuration."""
import os


GO_BASE_URL='https://opencode.ai/inference/go/openai/v1'


def configure_model():
    provider=os.environ.get('IMAGE2PPT_AGENT_PROVIDER','moma')
    if provider=='opencode_go':
        key=os.environ.get('IMAGE2PPT_AGENT_API_KEY') or os.environ.get('OPENCODE_GO_API_KEY')
        if not key:
            raise RuntimeError('OpenCode Go Agent credential absent')
        model=os.environ.get('IMAGE2PPT_AGENT_MODEL','qwen3.8-flash')
        os.environ.update(PLANNER_API_KEY=key,PLANNER_BASE_URL=GO_BASE_URL,PLANNER_MODEL=model)
    elif provider!='moma':
        raise ValueError('unsupported Agent inference provider')
    from ..pipeline import settings
    return provider,settings()


def provider_id(provider):
    return 'image2ppt-go' if provider=='opencode_go' else 'image2ppt-moma'


def planning_actor(job):
    from ..parallel_provider import Planner,Reviewer
    from ..go_provider import GoProvider
    from pathlib import Path
    selected,cfg=configure_model()
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
