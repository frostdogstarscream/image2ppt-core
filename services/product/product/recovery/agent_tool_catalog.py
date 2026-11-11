"""Host registration for concrete regional repair adapters."""
from .agent_decision import RegisteredTool


TOOLS = {
    'retry_text_mask_completion': {
        'kind': 'text',
        'description': 'Retry source-bound TextMaskCompletion using changed source/scene/layer/code evidence, '
                       'or switch a proved incomplete original/bounded mask to the existing semantic completion strategy. '
                       'Use for MASK_INCOMPLETE or SOURCE_GLYPH_UNCERTAIN. Unchanged failed inputs reuse the failure receipt. '
                       'The adapter preserves protected neighbors and only returns a candidate after existing mask checks pass.',
    },
    'switch_background_candidate': {
        'kind': 'text',
        'description': 'Read a source/scene/candidate-bound BACKGROUND_REPAIR_FAILED rejection and request the next '
                       'legal background method. Use after an actual background defect, not a minor accepted visual difference. '
                       'The rejected candidate is not repeated and unsafe masks stay blocked.',
    },
    'retry_asset_extraction': {
        'kind': 'asset',
        'description': 'Retry a complete independent transparent PNG from verified LayerD/source evidence. '
                       'Use for actual ASSET_EXTRACTION_UNCERTAIN, incomplete alpha or source-owner closure failure. '
                       'The host rechecks original semantic object coverage, protected peers, backing, source recomposition '
                       'and the full owner mask. A clipped object or unresolved owner remains blocked.',
    },
}


def register_repair_tools(object_kind, callbacks, implementation_bindings):
    """Register supplied executable callbacks only; absent adapters stay absent.

    Callbacks close over host-selected graph/source/layer/receipt paths and take
    an isolated output directory. The model receives tool IDs and descriptions.
    """
    if object_kind not in ('text', 'asset'):
        raise ValueError('unsupported repair object kind')
    if set(callbacks)-TOOLS.keys() or set(callbacks)!=set(implementation_bindings):
        raise ValueError('tool registration or implementation binding mismatch')
    result=[]
    for name,callback in callbacks.items():
        if not callable(callback):raise ValueError('registered repair adapter must be executable')
        if TOOLS[name]['kind']!=object_kind:continue
        binding=implementation_bindings[name]
        if not isinstance(binding,str) or len(binding)!=64 or any(c not in '0123456789abcdef' for c in binding):
            raise ValueError('repair implementation hash is invalid')
        result.append(RegisteredTool(name,TOOLS[name]['description'],binding,callback))
    return result
