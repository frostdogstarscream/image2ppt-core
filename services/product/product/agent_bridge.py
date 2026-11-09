"""Experimental planner transport for an explicitly delegated Codex sub-agent.

This is not a standalone Terra API. The active Codex agent answers file requests;
all validation, compilation and OpenCode Go review still use the product pipeline.
"""
import json
from pathlib import Path
import time
from .materialize import write


class FilePlanner:
    model='gpt-6.1-terra'
    provider_name='codex_terra_bridge'
    codex_required_at_runtime=True

    def __init__(self,directory,timeout=7200):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.number=0;self.timeout=timeout

    def ask(self,system,context,images,contract):
        self.number+=1
        response=self.directory/f'response-{self.number:03d}.json'
        request=self.directory/f'request-{self.number:03d}.json'
        write(request,{'planner':self.model,'system':system,'context':context,
            'images':[str(Path(p).resolve()) for p in images],
            'output_schema':contract.model_json_schema(),'response_path':str(response.resolve())})
        deadline=time.monotonic()+self.timeout
        while time.monotonic()<deadline:
            if response.is_file():
                value=json.loads(response.read_text(encoding='utf-8-sig'))
                write(self.directory/f'consumed-{self.number:03d}.json',{'response':response.name})
                return value
            if (self.directory/'cancel.requested').exists():raise RuntimeError('agent_bridge_cancelled')
            time.sleep(.5)
        raise RuntimeError('agent_bridge_timeout')
