"""Export only project-owned sources and a Claude Code skill entry."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OWN_SOURCES = [
    "image2ppt.ps1", "image2ppt.sh",
    "scripts/runtime.mjs", "scripts/ir.mjs", "scripts/compile.mjs",
    "scripts/main.mjs", "scripts/prepare.py", "scripts/pptx_xml.py",
    "scripts/check_demo.mjs", "scripts/check_pixels.py",
    "tests/ir.test.mjs",
    "skills/image2ppt/SKILL.md", "skills/image2ppt/agents/openai.yaml",
    "skills/image2ppt/references/slideir.md",
    "skills/image2ppt-claude/SKILL.md",
    "skills/image2ppt-claude/references/slideir.md",
    "examples/synthetic/slideir.json", "examples/synthetic/source/page.png",
    "examples/synthetic/assets/visual.png",
]

# Include the actual local compiler/core/review dependency closure. The former
# hand-picked subset omitted slide.mjs, core helpers and skill references.
# Only source trees are visited: no outputs, credentials or private fixtures.
OWN_SOURCES = sorted(set(OWN_SOURCES + [
    'requirements-core.lock', 'requirements-review.lock', 'requirements-auto.lock', 'pyproject.toml',
    'AGENTS.md', 'docs/automatic-workflow.md', 'docs/automatic-region-routing-v2-verification.md', 'services/vision/tools/auto_remote.py',
    'services/vision/pyproject.toml', 'configs/auto-client.example.json', 'configs/auto-remote.example.json',
    '.github/workflows/ci.yml', 'docs/candidate-workflow.md', 'docs/candidate-workflow-verification.md',
    'services/vision/REGION_LLM_PLANNING.md',
    'docs/layered-v3-verification.md', 'deploy/layered-worker/compose.yaml',
] + [p.relative_to(ROOT).as_posix()
     for directory, suffixes in [('scripts',{'.py','.mjs','.json'}),('src',{'.py'}),
                                ('services/vision/src',{'.py','.json','.html'}),('tests',{'.py','.mjs'}),
                                ('skills/image2ppt',{'.md','.yaml'}),('skills/image2ppt-claude',{'.md'})]
     for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in suffixes
     and '__pycache__' not in p.parts]
    + ['services/vision/tests/'+name for name in ['test_region_plan.py','test_region_candidates.py',
                                                'test_supervised_elements.py','test_candidate_workflow.py',
                                                'test_auto_workflow.py','test_auto_remote.py','test_element_service.py',
                                                'test_auto_regions.py','test_layered.py']]
    + [p.relative_to(ROOT).as_posix() for directory in ['docs/region-shadow-llm','docs/region-candidates']
       for p in (ROOT/directory).glob('*') if p.suffix in {'.json','.md'}]))

README = r"""# image2PPT 技能源码包与 Claude Code 入口

本包导出 image2PPT v0.1 的项目自有源码，并提供 Claude Code 的项目级技能入口。
它是源码及接入包；没有包含 PPT 编译、渲染引擎，不能当作独立安装包。

## 包含的内容

- `skills/image2ppt/`：Codex Skill 源码及原有 UI 元数据。
- `skills/image2ppt-claude/`：Claude Code Skill 的**维护源**（bash 调用形式、多页 `compose` 说明）。
- `.claude/skills/image2ppt/`：由上述维护源写入的项目级 Claude Code 入口，改用当前具备视觉能力的模型分析图片。
- `scripts/`、`image2ppt.ps1`、`image2ppt.sh`：准备图片、校验 SlideIR、编译、回渲染、编辑扰动检查与多页合成。
- `tests/`、`examples/synthetic/`：协议测试及人工生成的合成输入。
- `manifest.json`：包内文件的 SHA256 和字节数。

没有打包用户业务图片、历史 PPT、runs、outputs、缓存、实际配置、凭据，
也没有打包 `@oai/artifact-tool` 或 Codex Presentations 的第三方运行文件。
本机所读取的 Artifact Tool 许可证限制再分发，因此本包仅引用这些依赖。

## 在 Claude Code 中加载

解压后，进入包含本说明与 `image2ppt.ps1` 的 `image2ppt` 目录启动 Claude Code：

```powershell
claude
```

然后在会话中输入（替换图片路径）：

```text
/image2ppt 把 D:\slides\source.png 还原为可编辑 PPT，保留原文和布局，并检查实际导出的文件。
```

Claude Code 的项目技能路径是 `.claude/skills/<name>/SKILL.md`。
个人技能路径是 `~/.claude/skills/<name>/SKILL.md`。
本 Skill 还需要项目根目录下的脚本；只复制 SKILL.md 不能获得完整工具链。
仅移动技能文件夹时，必须另行告诉模型工具链根目录的位置。
同时安装项目级与个人级会让技能列表里出现两条同名条目，二者内容一致，只留一份即可。

### 两个必须成立的前提

- `image2ppt.ps1` 必须存成 **带 UTF-8 BOM**。Windows PowerShell 5.1 按系统 ANSI 码页解码无 BOM 文件，
  会在脚本的中文字符串处报 `ParserError`，导致**所有**命令失效。用 `head -c 3 image2ppt.ps1 | xxd`
  确认以 `efbb bf` 开头。
- 渲染引擎必须可解析：`@oai/artifact-tool` 与 Codex Presentations 的 `container_tools` 校验器；
  `./image2ppt.sh doctor` 会报告这两项。

### 多页合成

`compile` 一次只产出一页。把多个已还原页合成一个 deck（传入顺序即页序）：

```bash
./image2ppt.sh compose -InputPath 'runs/<run02>/slideir.json,runs/<run03>/slideir.json' -OutputPath outputs/deck
```

官方格式说明：https://code.claude.com/docs/en/skills

## 当前运行依赖

现有编译器继续使用当前原型的 `@oai/artifact-tool` 和 Codex Presentations 辅助工具。
依赖定位规则见 `scripts/runtime.mjs`；PowerShell 执行器规则见 `image2ppt.ps1`。

- `IMAGE2PPT_NODE`：Node 可执行文件的绝对路径。未设置时使用 Codex 本机 Node 路径。
- `IMAGE2PPT_PYTHON`：Python 可执行文件的绝对路径。未设置时使用 Codex 本机 Python 路径。
- `IMAGE2PPT_NODE_MODULES`：已有 `@oai/artifact-tool` 所在 node_modules 的绝对路径。
- `runtime.local.json`：可配置 `node_modules`、`python`、`presentations_skill`；示例见同名 example 文件。

`runtime.local.json` 只配置本机路径，不安装依赖，也不覆盖上述环境变量。
包装脚本的 Node/Python 可执行文件路径需要通过对应环境变量设置。
Python 的图片准备功能需要 Pillow；当前回渲染及几何检查还依赖上述引擎与辅助工具。
本包没有提供下载或复制专有运行库的脚本。

先检查已有环境。Claude Code 的 shell 是 bash，而 `.\image2ppt.ps1` 在 bash 下是语法错误
（`syntax error near unexpected token`），因此请使用转发脚本：

```bash
./image2ppt.sh doctor
```

在 PowerShell 会话里两者等价：

```powershell
.\image2ppt.ps1 doctor
```

缺少引擎时，该检查会失败，编译与实际 PPTX 回渲染无法完成。
协议测试本身不依赖 PPT 引擎，可用一个支持 Node test runner 的 Node 运行：

```powershell
node --test .\tests\ir.test.mjs
```

若要在没有 Codex 运行依赖的另一台电脑上完整执行，仍需将编译与回渲染部分
适配到可独立安装、可分发的后端，并对实际导出的 PPTX 重新验证。

## 在 Claude Code 中接入其他模型

模型选择由 Claude Code 和模型服务配置承担，Skill 本身不调用模型 API。
通过自建或供应商网关使用非 Claude 模型属于需要逐项验证的集成方案；
本包没有验证任何特定第三方模型，也没有更改用户的 Claude 配置。

使用 `ANTHROPIC_BASE_URL` 时，网关应实现 Claude Code 所用的 Anthropic Messages 协议。
只有 OpenAI Chat Completions/Responses 格式的服务地址不能直接等同于这个接口。
协议转换时还应正确处理图片内容、工具调用/工具结果及流式事件。
目标模型必须具备图片输入能力，才能直接执行当前的图片分析流程。
纯文本模型需要增加独立的视觉/OCR 环节，当前源码没有这个自动化功能。

以下仅为 PowerShell 临时会话的配置示例，请按网关的实际认证说明填写：

```powershell
$env:ANTHROPIC_BASE_URL = "https://your-gateway.example.com"
$env:ANTHROPIC_AUTH_TOKEN = "your-gateway-token"
claude --model "your-provider-model-id"
```

`ANTHROPIC_BASE_URL` 只改变请求地址；模型 ID 还需要通过 `--model` 或 Claude Code 的
相应模型配置选择。不要把真实凭据写入 Skill 或提交到仓库。

官方接入说明：https://code.claude.com/docs/en/llm-gateway-connect
官方协议说明：https://code.claude.com/docs/en/llm-gateway-protocol
官方模型配置：https://code.claude.com/docs/en/model-config

## 验证边界

本项目已在本机 Codex 环境生成过可编辑 PPT 并检查实际文件的回渲染。
这不证明解压到新机器后可以运行，也不证明 Claude Code 或其他模型可以完成同样任务。
本次导出检查覆盖文件清单、SHA256、ZIP CRC、Skill 格式及源码包中的协议测试。
未运行 Claude Code + 第三方模型的完整图片到 PPT 流程。
"""


def main():
    revision = "image2ppt-source-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    destination = ROOT / "exports" / revision
    package = destination / "image2ppt"
    package.mkdir(parents=True, exist_ok=False)
    for name in OWN_SOURCES:
        source = ROOT / name
        source.resolve(strict=True).relative_to(ROOT.resolve())
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())

    # The Claude Code entry has its own maintained source. Do not derive it from
    # the Codex skill by string substitution: that drifts silently whenever the
    # Codex wording moves, and it cannot express Claude-Code-specific guidance.
    claude_source = ROOT / "skills/image2ppt-claude"
    claude = package / ".claude/skills/image2ppt"
    (claude / "references").mkdir(parents=True)
    (claude / "SKILL.md").write_bytes((claude_source / "SKILL.md").read_bytes())
    for item in (claude_source/'references').glob('*.md'):
        (claude/'references'/item.name).write_bytes(item.read_bytes())
    # Project-local Codex mirror, generated from its own maintained source.
    for item in (ROOT/'skills/image2ppt').rglob('*'):
        if item.is_file():
            target=package/'.agents/skills/image2ppt'/item.relative_to(ROOT/'skills/image2ppt')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(item.read_bytes())
    (package / "README.md").write_text(README, encoding="utf-8", newline="\n")
    example = {"node_modules": "C:/path/to/existing/node_modules",
               "python": "C:/path/to/python.exe",
               "presentations_skill": "C:/path/to/existing/presentations/skills/presentations"}
    (package / "runtime.local.example.json").write_text(json.dumps(example, indent=2) + "\n", encoding="utf-8")
    (package / ".gitignore").write_text(".runtime/\n.build/\nruns/\noutputs/\n__pycache__/\nruntime.local.json\nexamples/**/outputs/\n", encoding="utf-8")

    payload = sorted(p for p in package.rglob("*") if p.is_file())
    manifest = {"package_kind": "source-only-with-claude-code-entry",
                "engine_included": False, "model_credentials_included": False,
                "files": [{"path": p.relative_to(package).as_posix(), "bytes": p.stat().st_size,
                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in payload]}
    (package / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = ROOT / "exports" / (revision + ".zip")
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as bundle:
        for item in sorted(p for p in package.rglob("*") if p.is_file()):
            bundle.write(item, item.relative_to(destination).as_posix())
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None, "ZIP CRC check failed"
        expected = {"image2ppt/" + entry["path"] for entry in manifest["files"]}
        expected.add("image2ppt/manifest.json")
        assert set(bundle.namelist()) == expected, "Unexpected files in archive"
        for entry in manifest["files"]:
            assert hashlib.sha256(bundle.read("image2ppt/" + entry["path"])).hexdigest() == entry["sha256"]
    print(json.dumps({"archive": str(archive), "package_root": str(package),
                      "files": len(manifest["files"]) + 1, "bytes": archive.stat().st_size,
                      "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                      "crc_and_manifest_passed": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
