"""Host-selected policy files only; pages/model outputs cannot select a skill."""
from dataclasses import dataclass
from pathlib import Path
import re
from ..materialize import digest
from ..parallel_store import key

FILES = ('SKILL.md', 'references/architecture.md', 'references/tool-catalog.md',
    'references/region-reconstruction.md', 'references/quality-and-editability.md',
    'references/runtime-security.md', 'references/troubleshooting.md',
    'schemas/region-plan.schema.json', 'schemas/tool-action.schema.json',
    'schemas/quality-decision.schema.json', 'schemas/agent-session.schema.json',
    'schemas/page-plan.schema.json', 'schemas/quality-opinion.schema.json',
    'schemas/execution-action.schema.json','schemas/scene-proposal.schema.json','schemas/execution-budget.schema.json')


@dataclass(frozen=True)
class Policy:
    version: str
    system: str
    files: dict


def load_skill(directory):
    root = Path(directory).resolve(strict=True)
    hashes = {}; texts = []
    for name in FILES:
        file = (root / name).resolve(strict=True)
        if not file.is_relative_to(root):
            raise ValueError('policy path escape')
        if file.stat().st_size > 100_000:
            raise ValueError('policy file too large')
        hashes[name] = digest(file)
        text = file.read_text(encoding='utf-8-sig')
        if name.endswith('.md'):
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)', text):
                if '://' in link:
                    raise ValueError('policy references must be bundled')
                target = (file.parent / link).resolve(strict=True)
                if not target.is_relative_to(root):
                    raise ValueError('policy reference escapes package')
            texts.append('### ' + name + '\n' + text)
    return Policy(key(hashes), '\n\n'.join(texts), hashes)
