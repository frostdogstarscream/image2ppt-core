"""Connect verified LayerD stacks to the r12 background precheck prototype."""
from dataclasses import replace

from .background_adapter import RecoveryBlocked, _validate, generate_background_candidates
from .layerd_adapter import lower_backgrounds


def generate_with_layerd(source, write_mask, protected, layer_directory, *, inpaint=None):
    """Try lower prefixes before one optional inpaint attempt.

    This returns diagnostic trials. It grants no source ownership and makes no
    candidate-stage, Office, Go, or publication decision.
    """
    _validate(source, write_mask, protected, None)
    issues = []
    try:
        lower = lower_backgrounds(source, layer_directory, write_mask)
    except (ValueError, OSError) as error:
        lower = []
        issues.append({'method': 'layerd_lower',
                       'code': getattr(error, 'code', 'INVALID_LAYER_PACK')})
    for background in lower:
        results, rejected = generate_background_candidates(
            source, write_mask, protected, lower_rgb=background.image,
            lower_provenance_verified=True)
        issues.extend({**issue, 'provenance': background.provenance}
                      for issue in rejected if issue['code'] != 'NO_VALID_REPAIR_CANDIDATE')
        if results:
            candidate = results[0]
            return [replace(candidate, diagnostics={**candidate.diagnostics,
                            'layerd_provenance': background.provenance})], issues
    if not lower and not issues:
        issues.append({'method': 'layerd_lower', 'code': 'NO_OPAQUE_LOWER_STACK'})
    if inpaint is not None:
        results, rejected = generate_background_candidates(source, write_mask, protected, inpaint=inpaint)
        return results, issues + rejected
    issues.append({'method': 'source_fallback', 'code': 'NO_VALID_REPAIR_CANDIDATE'})
    return [], issues
