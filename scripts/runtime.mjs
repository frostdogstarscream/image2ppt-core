import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export async function runtime() {
  const profile = process.env.USERPROFILE || process.env.HOME;
  let config = {};
  try { config = JSON.parse(await fs.readFile(path.join(ROOT, 'runtime.local.json'), 'utf8')); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const primary = path.join(profile || '', '.cache/codex-runtimes/codex-primary-runtime/dependencies');
  const candidates = [process.env.IMAGE2PPT_NODE_MODULES, config.node_modules,
    path.join(primary, 'node/node_modules'),
    path.join(ROOT, '.runtime/bundle/codex-primary-runtime/dependencies/node/node_modules')].filter(Boolean);
  let selected;
  for (const candidate of candidates) {
    try { await fs.access(path.join(candidate, '@oai/artifact-tool/package.json')); selected = candidate; break; }
    catch {}
  }
  if (!selected) throw new Error('缺少 @oai/artifact-tool。运行 doctor 查看环境；不要把网页预览当作 PPTX 回渲染。');
  const require = createRequire(path.join(selected, '__image2ppt__.cjs'));
  const metadata = JSON.parse(await fs.readFile(path.join(selected, '@oai/artifact-tool/package.json'), 'utf8'));
  process.env.RUNTIME_NODE_MODULES = selected;
  const api = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
  const python = process.env.IMAGE2PPT_PYTHON || config.python || path.join(primary, 'python/python.exe');
  let skill = config.presentations_skill;
  if (!skill) {
    const parent = path.join(profile || '', '.codex/plugins/cache/openai-primary-runtime/presentations');
    const versions = (await fs.readdir(parent)).sort().reverse();
    for (const version of versions) {
      const candidate = path.join(parent, version, 'skills/presentations');
      try { await fs.access(path.join(candidate, 'container_tools/artifact_tool_utils.mjs')); skill = candidate; break; }
      catch {}
    }
  }
  if (!skill) throw new Error('找不到 Presentations 技能的校验工具。');
  return { api, require, node_modules: selected, python, skill, artifact_version: metadata.version };
}

export async function writeJson(file, value) {
  await fs.writeFile(file, JSON.stringify(value, null, 2) + '\n', {encoding:'utf8', flag:'wx'});
}

export async function saveBlob(file, blob) {
  await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer()), {flag:'wx'});
}

