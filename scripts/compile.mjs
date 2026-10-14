import fs from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';
import { runtime, ROOT, writeJson, saveBlob } from './runtime.mjs';
import { loadIR, sha256 } from './ir.mjs';
import { renderPptx } from './render.mjs';
import { buildSlide } from './slide.mjs';

export function pythonJson(rt, args) {
  const result=spawnSync(rt.python,['-X','utf8',...args],{encoding:'utf8',windowsHide:true,maxBuffer:4*1024*1024});
  if (result.error || result.status!==0) throw new Error(result.error?.message || result.stderr || result.stdout);
  return JSON.parse(result.stdout);
}

export async function compile(file) {
  const data=await loadIR(file), {ir,info}=data;
  if (ir.unresolved_evidence_ids?.length) throw new Error('还有未处理源内容；先用图片回退保留内容，再导出。');
  const rt=await runtime();
  const {Presentation,PresentationFile,FileBlob}=rt.api;
  const stamp=new Date().toISOString().replace(/[:.]/g,'-')+'-'+randomUUID().slice(0,8);
  const buildDir=path.join(ROOT,'.build',stamp), output=path.join(data.base,'outputs',stamp);
  await fs.mkdir(buildDir,{recursive:true});
  await fs.mkdir(output,{recursive:true});
  const [tw,th]=ir.canvas.target_px;
  const presentation=Presentation.create({slideSize:{width:tw,height:th}});
  await buildSlide(presentation,{ir,info});
  const raw=path.join(buildDir,'raw.pptx'), candidate=path.join(buildDir,'candidate.pptx'), final=path.join(output,'result.pptx');
  await (await PresentationFile.exportPptx(presentation)).save(raw);
  const packageInfo=pythonJson(rt,[path.join(ROOT,'scripts/pptx_xml.py'),'adapt',raw,'--target',candidate,'--ir',data.file]);
  const {finalizePresentation}=await import(pathToFileURL(path.join(rt.skill,'container_tools/artifact_tool_utils.mjs')).href);
  await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:final,pythonExecutable:rt.python,
    explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
    integrityValidatorPath:path.join(rt.skill,'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath:path.join(rt.skill,'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs:['--expected-slide-size-emu',`${Math.round(tw*9525)},${Math.round(th*9525)}`],
    fontPolicy:{basis:'design',families:[...new Set(ir.nodes.filter(n=>n.type==='text').flatMap(n=>[n.font.latin,n.font.ea]))]},
    verifyArtifactToolImport:true,receiptPath:path.join(buildDir,'validation.json')});
  const rendered=await renderPptx(final,path.join(buildDir,'render'));
  await fs.copyFile(rendered.pages[0].path,path.join(output,'preview.png'));
  const finalInfo=pythonJson(rt,[path.join(ROOT,'scripts/pptx_xml.py'),'inspect',final]);
  if (finalInfo.slides[0].groups!==ir.groups.length) throw new Error('最终 PPTX 的真实分组数量与 IR 不一致。');
  const immutableIR=structuredClone(ir);
  await fs.mkdir(path.join(output,'source'));
  await fs.mkdir(path.join(output,'assets'));
  await fs.copyFile(data.source,path.join(output,'source/page.png'));
  immutableIR.source_image='source/page.png';
  for (const asset of immutableIR.assets) {
    const relative=`assets/${asset.id}${path.extname(asset._file).toLowerCase()}`;
    await fs.copyFile(asset._file,path.join(output,relative));
    asset.path=relative;
    delete asset._file;
  }
  await writeJson(path.join(output,'slideir.snapshot.json'),immutableIR);
  await loadIR(path.join(output,'slideir.snapshot.json'));
  const report={schema_version:'0.1',status:ir.review_status,source_sha256:ir.source_sha256,ir_sha256:data.ir_sha256,
    pptx_sha256:sha256(await fs.readFile(final)),engine:{name:'@oai/artifact-tool',version:rt.artifact_version,node_modules:rt.node_modules},
    preview:{origin:'reimported_final_pptx',renderer:'Artifact Tool',file:'preview.png'},
    package_checks:finalInfo,native_nodes:ir.nodes.filter(n=>n.type!=='image').map(n=>({id:n.id,type:n.type})),
    image_fallbacks:ir.nodes.filter(n=>n.type==='image').map(n=>({id:n.id,asset_id:n.asset_id,editable:'move_resize_only'})),
    warnings:info.warnings,font_identification_verified:false,
    powerPoint_checked:false,wps_checked:false,human_acceptance:false,
    validation_receipt:path.relative(ROOT,path.join(buildDir,'validation.json')),intermediate_package_checks:packageInfo};
  await writeJson(path.join(output,'quality.json'),report);
  return {output_dir:output,pptx:final,preview:path.join(output,'preview.png'),quality:path.join(output,'quality.json'),status:report.status};
}

