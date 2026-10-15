import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { pathToFileURL } from 'node:url';
import { runtime, ROOT, writeJson } from './runtime.mjs';
import { assertAuthoring, loadIR, sha256 } from './ir.mjs';
import { renderPptx } from './render.mjs';
import { buildSlide } from './slide.mjs';
import { pythonJson } from './compile.mjs';

const pageNumber = ir => {
  const match = /(\d{2,})\.png/.exec(ir.review_notes?.[0] ?? '');
  return match ? match[1] : null;
};

// Build one deck whose slides are the given IRs, in the order supplied.
// Each page still goes through the same authoring, OOXML adapt, finalize and
// re-import render chain as the single-page `compile` path.
export async function compose(files, outputRoot) {
  if (!files || !files.length) throw new Error('compose 需要至少一个 IR。');
  const loaded = [];
  for (const file of files) { await assertAuthoring(file); loaded.push(await loadIR(file)); }

  const [sw, sh] = loaded[0].ir.canvas.source_px, [tw, th] = loaded[0].ir.canvas.target_px;
  for (const {ir, file} of loaded) {
    if (ir.unresolved_evidence_ids?.length) throw new Error(`${file}: 还有未处理源内容；先用图片回退保留内容，再导出。`);
    const [s2, s3] = ir.canvas.source_px, [t2, t3] = ir.canvas.target_px;
    if (t2 !== tw || t3 !== th) throw new Error(`${file}: 目标画布与第 1 页不一致；一个 deck 的所有页必须同尺寸。`);
    if (s2 !== sw || s3 !== sh) throw new Error(`${file}: 源画布与第 1 页不一致。`);
  }

  const rt = await runtime();
  const { Presentation, PresentationFile } = rt.api;
  const stamp = new Date().toISOString().replace(/[:.]/g,'-') + '-' + randomUUID().slice(0,8);
  const buildDir = path.join(ROOT, '.build', 'compose-' + stamp);
  const output = path.join(path.resolve(outputRoot ?? path.join(ROOT, 'outputs')), stamp);
  await fs.mkdir(buildDir, {recursive:true});
  await fs.mkdir(output, {recursive:true});

  const presentation = Presentation.create({slideSize:{width:tw,height:th}});
  for (let index = 0; index < loaded.length; index++) {
    await buildSlide(presentation, loaded[index], `slide ${index+1}/${loaded.length}`);
  }

  const raw = path.join(buildDir,'raw.pptx'), candidate = path.join(buildDir,'candidate.pptx'), final = path.join(output,'deck.pptx');
  await (await PresentationFile.exportPptx(presentation)).save(raw);

  const adaptArgs = [path.join(ROOT,'scripts/pptx_xml.py'),'adapt',raw,'--target',candidate];
  for (const {file} of loaded) adaptArgs.push('--ir', file);
  const packageInfo = pythonJson(rt, adaptArgs);

  const { finalizePresentation } = await import(pathToFileURL(path.join(rt.skill,'container_tools/artifact_tool_utils.mjs')).href);
  // Slide numbers, not IR paths: the finalizer keys on the delivered deck.
  const tableOwners = loaded.flatMap(({ir}, index) => ir.nodes.some(n=>n.type==='table') ? [index+1] : []);
  const chartOwners = loaded.flatMap(({ir}, index) => ir.nodes.some(n=>n.type==='chart') ? [index+1] : []);
  await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:final,pythonExecutable:rt.python,
    explicitTotalSlideCount:loaded.length,requiredNativeTableOwnerSlides:tableOwners,
    requiredNativeChartOwnerSlides:chartOwners,materializeLiteralChartWorkbooks:true,
    integrityValidatorPath:path.join(rt.skill,'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath:path.join(rt.skill,'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs:['--expected-slide-size-emu',`${Math.round(tw*9525)},${Math.round(th*9525)}`,
      ...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
    fontPolicy:{basis:'design',families:[...new Set(loaded.flatMap(({ir})=>ir.nodes.filter(n=>n.type==='text'||n.type==='table').flatMap(n=>[n.font.latin,n.font.ea])))]},
    verifyArtifactToolImport:true,receiptPath:path.join(buildDir,'validation.json')});

  const rendered = await renderPptx(final, path.join(buildDir,'render'));
  if (rendered.pages.length !== loaded.length) throw new Error(`渲染出 ${rendered.pages.length} 页，期望 ${loaded.length} 页。`);
  const finalInfo = pythonJson(rt, [path.join(ROOT,'scripts/pptx_xml.py'),'inspect',final]);
  if (finalInfo.slide_count !== loaded.length) throw new Error(`最终 deck 有 ${finalInfo.slide_count} 页，期望 ${loaded.length} 页。`);
  for (let index = 0; index < loaded.length; index++) {
    if (finalInfo.slides[index].groups !== loaded[index].ir.groups.length) {
      throw new Error(`第 ${index+1} 页真实分组数 ${finalInfo.slides[index].groups} 与 IR 的 ${loaded[index].ir.groups.length} 不一致。`);
    }
  }

  const previewDir = path.join(output,'preview');
  await fs.mkdir(previewDir, {recursive:true});
  const pages = [], compareCases = [];
  for (let index = 0; index < loaded.length; index++) {
    const name = `page-${String(index+1).padStart(3,'0')}.png`;
    const dest = path.join(previewDir, name);
    await fs.copyFile(rendered.pages[index].path, dest);
    pages.push({slide:index+1, page:pageNumber(loaded[index].ir), ir:path.relative(ROOT,loaded[index].file),
      ir_sha256:loaded[index].ir_sha256, source_sha256:loaded[index].ir.source_sha256, preview:`preview/${name}`,
      nodes:loaded[index].ir.nodes.length, groups:loaded[index].ir.groups.length,
      images:loaded[index].ir.nodes.filter(n=>n.type==='image').length});
    compareCases.push({slide:index+1, page:pageNumber(loaded[index].ir),
      source:loaded[index].source, preview:dest});
  }

  const compareDir = path.join(output,'compare');
  const casesFile = path.join(buildDir,'compare-cases.json');
  await writeJson(casesFile, compareCases);
  const compare = pythonJson(rt, [path.join(ROOT,'scripts/compare_pages.py'), casesFile, compareDir]);

  const manifest = {schema_version:'0.1', kind:'composed_deck', slide_count:loaded.length,
    canvas:{source_px:[sw,sh],target_px:[tw,th]}, pptx:'deck.pptx', pptx_sha256:sha256(await fs.readFile(final)),
    engine:{name:'@oai/artifact-tool',version:rt.artifact_version,node_modules:rt.node_modules},
    status:'REVIEW_REQUIRED', preview:{origin:'reimported_final_pptx',renderer:'Artifact Tool',dir:'preview'},
    pages, compare, package_checks:finalInfo,
    warnings:[...new Set(loaded.flatMap(({info})=>info.warnings))],
    font_identification_verified:false,powerPoint_checked:false,wps_checked:false,human_acceptance:false,
    validation_receipt:path.relative(ROOT,path.join(buildDir,'validation.json'))};
  await writeJson(path.join(output,'quality.json'), manifest);
  return {output_dir:output, pptx:final, preview:previewDir, compare:compareDir,
    quality:path.join(output,'quality.json'), slide_count:loaded.length, status:manifest.status};
}

async function main() {
  // IR paths are positional, in slide order: compose.mjs <ir1> <ir2> ... --output <root>
  const {values,positionals} = parseArgs({options:{output:{type:'string'}},allowPositionals:true});
  const result = await compose(positionals, values.output);
  console.log(JSON.stringify(result,null,2));
}

if (process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  main().catch(error=>{console.error(error.stack||error.message);process.exitCode=1;});
}
