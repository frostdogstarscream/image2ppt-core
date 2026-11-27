import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';
import { runtime, ROOT } from './runtime.mjs';
import { assertAuthoring, loadIR } from './ir.mjs';
import { renderPptx } from './render.mjs';
import { buildSlide } from './slide.mjs';
import { pythonJson } from './compile.mjs';

const file = process.argv[2];
const log = message => console.error(new Date().toISOString(), message);

await assertAuthoring(file);
const data = await loadIR(file), { ir } = data;
log('loaded');
const rt = await runtime();
const { Presentation, PresentationFile } = rt.api;
const stamp = new Date().toISOString().replace(/[:.]/g, '-') + '-' + randomUUID().slice(0, 8);
const buildDir = path.join(ROOT, '.build', stamp);
await fs.mkdir(buildDir, { recursive: true });
const [tw, th] = ir.canvas.target_px;
const presentation = Presentation.create({ slideSize: { width: tw, height: th } });
await buildSlide(presentation, { ir, info: data.info });
log('built slide');
const raw = path.join(buildDir, 'raw.pptx'), candidate = path.join(buildDir, 'candidate.pptx'), final = path.join(buildDir, 'result.pptx');
await (await PresentationFile.exportPptx(presentation)).save(raw);
log('exported raw');
pythonJson(rt, [path.join(ROOT, 'scripts/pptx_xml.py'), 'adapt', raw, '--target', candidate, '--ir', data.file]);
log('adapted');
const { finalizePresentation } = await import(pathToFileURL(path.join(rt.skill, 'container_tools/artifact_tool_utils.mjs')).href);
const declaredFonts = [...new Set(ir.nodes.filter(n => ['text', 'table', 'chart'].includes(n.type)).flatMap(n => [n.font.latin, n.font.ea]))];
await finalizePresentation({ workspaceDir: ROOT, candidatePath: candidate, finalPath: final, pythonExecutable: rt.python,
  explicitTotalSlideCount: 1, requiredNativeTableOwnerSlides: [], requiredNativeChartOwnerSlides: [],
  materializeLiteralChartWorkbooks: true,
  integrityValidatorPath: path.join(rt.skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(rt.skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', `${Math.round(tw * 9525)},${Math.round(th * 9525)}`],
  fontPolicy: { basis: 'design', families: declaredFonts.length ? declaredFonts : ['Arial'] },
  verifyArtifactToolImport: true, receiptPath: path.join(buildDir, 'validation.json') });
log('finalized');
const rendered = await renderPptx(final, path.join(buildDir, 'render'));
log('rendered ' + rendered.pages[0].path);
