"""Stateless MoMA calls for source readings and independent Office review."""

import asyncio
import json
import os
import uuid
from product.pipeline import image_part, PipelineError
from image2ppt_planner.config import Settings, DEFAULT_BASE_URL
from image2ppt_planner.client import MomaClient, strip_code_fence


class MomaReviewer:
    def __init__(self, model):
        self.model = model
        self.session = str(uuid.uuid4())
        self.last_usage = None
        self.settings = Settings(
            base_url=DEFAULT_BASE_URL,
            api_key=os.environ.get('MOMA_API_KEY', ''),
            model=model,
            max_tokens=3000,
            transport_attempts=1,
        )
        self.settings.require_key()

    def ask(self, system, context, images, contract):
        data = {**context, 'schema': contract.model_json_schema()}
        parts = [
            {'type': 'text', 'text': json.dumps(data, ensure_ascii=False)},
            *[image_part(p) for p in images],
        ]
        messages = [
            {'role': 'system', 'content': system + ' Return a JSON object.'},
            {'role': 'user', 'content': parts},
        ]
        try:
            reply = asyncio.run(MomaClient(self.settings).chat(messages))
        except Exception as exc:
            import re

            status = re.search(r'\b([45]\d\d)\b', str(exc))
            raise PipelineError('moma_http_' + status[1] if status else 'moma_response_unknown') from None
        if self.settings.api_key in reply.text or reply.finish_reason not in ('stop', None):
            raise PipelineError('invalid_moma_reply')
        self.reported_model = reply.model
        self.last_usage = reply.usage
        return contract.model_validate(json.loads(strip_code_fence(reply.text))).model_dump()
