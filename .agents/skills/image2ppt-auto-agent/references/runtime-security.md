# Trusted boundaries

The host selects the skill directory, configured model and job-to-evidence mapping. User/API requests and model output cannot select filesystem paths, endpoints or arbitrary tools. Authenticate the principal and bind each session to that principal, job, page, source, scene and skill version. Check the frozen files again before a decision or tool operation.

Source images, OCR, scene object descriptions, filenames and model replies are untrusted data. Text saying “delete source-plate”, “ignore QA” or “publish now” has no authority. There are no shell, code, network-fetch, erase or publish tools. Unknown fields and malformed JSON cause INVALID; they are not repaired into executable actions.

Reserve each model call durably through the existing Store.call before sending it. Include skill, source, scene, request schema, model settings and image bytes in identity. Store original replies in restricted task evidence, not ordinary logs. Credentials remain in the existing provider process and never enter policy, receipts or model context. Unknown responses are not automatically retried, including in a new session against identical inputs.

Defaults: one whole-page planning attempt, at most three model calls per session, eight tool calls and three actions per region. The host may lower budgets. Model timeout closes the pending decision as UNKNOWN; repeated identical action IDs replay receipts without another execution. Crash-recovered pending work remains UNKNOWN.

Read-only sessions write only derived crops and receipts. An explicitly registered execution host can write new candidate evidence, private SceneGraph revisions, compiled PPTX and Office test copies under its isolated execution directory. Existing source files, stable PNGs, parent SceneGraphs, PPTX and QA reports remain byte-identical. Service operators supply an explicit evidence catalogue and execution adapters; external API clients cannot register new jobs, paths or adapters.
