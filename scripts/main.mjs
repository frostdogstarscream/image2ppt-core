import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { runtime, ROOT, writeJson, saveBlob } from './runtime.mjs';
import { loadIR, sha256 } from './ir.mjs';
import { compile, pythonJson } from './compile.mjs';

export async function demo() {
  const rt=await runtime();
  const run=path.join(ROOT,'runs','synthetic-'+new Date().toISOString().replace(/[:.]/g,'-')+'-'+randomUUID().slice(0,8));
  await fs.mkdir(path.join(run,'source'),{recursive:true}); await fs.mkdir(path.join(run,'assets'));
  // A small deterministic fixture, not a performance or font-identification sample.
  const p=rt.api.Presentation.create({slideSize:{width:1280,height:720}}), s=p.slides.add();
  s.background.fill='#FFFFFF';
  const nodes=[
    {id:'title',type:'text',bbox:[72,52,1100,68],z:10,text:'2026 年业务规划',font:{latin:'Arial',ea:'Microsoft YaHei',size_pt:32},bold:true,color:'172B4D'},
    {id:'subtitle',type:'text',bbox:[74,137,1050,40],z:20,text:'收入增长 18% · 成本下降 7% · Q4 交付',font:{latin:'Arial',ea:'Microsoft YaHei',size_pt:20},color:'536479'},
    {id:'card-bg',type:'shape',geometry:'rect',bbox:[72,230,520,298],z:30,fill:'EAF2FA'},
    {id:'card-title',type:'text',bbox:[100,263,455,50],z:40,text:'交付目标',font:{latin:'Arial',ea:'Microsoft YaHei',size_pt:24},bold:true,color:'172B4D'},
    {id:'card-body',type:'text',bbox:[100,342,455,112],z:50,text:'主要文字可以修改\n简单模块可以移动\n修改之后保留完整内容',font:{latin:'Arial',ea:'Microsoft YaHei',size_pt:20},color:'26384B'},
    {id:'visual-bg',type:'shape',geometry:'rect',bbox:[672,230,520,298],z:60,fill:'172B4D'},
    {id:'visual-text',type:'text',bbox:[730,340,400,76],z:70,text:'复杂视觉模块',font:{latin:'Arial',ea:'Microsoft YaHei',size_pt:26},bold:true,color:'FFFFFF'},
    {id:'footer',type:'text',bbox:[74,628,1100,38],z:80,text:'合成测试页：仅用于验证工具链，不代表真实图片还原效果。',font:{latin:'Arial',ea:'Microsoft YaHei',size_pt:16},color:'536479'}
  ];
  for (const n of nodes) {
    const shape=s.shapes.add({name:n.id,geometry:n.type==='text'?'textbox':n.geometry,
      position:{left:n.bbox[0],top:n.bbox[1],width:n.bbox[2],height:n.bbox[3]},fill:n.type==='text'?'none':'#'+n.fill,line:{fill:'none',width:0}});
    if (n.type==='text') {shape.text=n.text;shape.text.style={typeface:n.font.ea,fontSize:n.font.size_pt*96/72,bold:n.bold??false,color:'#'+n.color,wrap:'none',autoFit:'none',insets:0,verticalAlignment:'top'};}
  }
  const source=path.join(run,'source/page.png');
  await saveBlob(source,await p.export({slide:s,format:'png',scale:1}));
  // Use a source crop as a movable image fallback, with its text owned by that image.
  const crop=pythonJson(rt,[path.join(ROOT,'scripts/prepare.py'),'crop',source,'--bbox','672','230','520','298','--output',path.join(run,'assets/visual.png')]);
  const evidence=nodes.map(n=>({id:'ev-'+n.id,kind:n.type,bbox:n.bbox,...(n.text?{text:n.text}:{})}));
  const outNodes=nodes.filter(n=>!['visual-bg','visual-text'].includes(n.id)).map(n=>({...n,evidence_ids:['ev-'+n.id]}));
  outNodes.push({id:'visual-module',type:'image',bbox:[672,230,520,298],z:60,asset_id:'visual',evidence_ids:[]});
  const ir={schema_version:'0.1',source_image:'source/page.png',source_sha256:sha256(await fs.readFile(source)),
    canvas:{source_px:[1280,720],target_px:[1280,720]},background:{color:'FFFFFF'},review_status:'needs_review',
    review_notes:['合成测试页，结构来自测试夹具，未进行真实图片识别评测。','字体为测试指定字体；尚未在 PowerPoint/WPS 中人工编辑。'],
    evidence,assets:[{id:'visual',path:'assets/visual.png',purpose:'module',sha256:crop.sha256,contains_evidence_ids:['ev-visual-bg','ev-visual-text']}],
    groups:[{id:'delivery-card',member_ids:['card-bg','card-title','card-body']}],nodes:outNodes};
  const file=path.join(run,'slideir.json'); await writeJson(file,ir);
  const result=await compile(file);
  await writeJson(path.join(run,'demo.json'),{synthetic:true,ir:file,...result});
  return {synthetic:true,ir:file,source,...result};
}

export async function perturb(file,nodeId,action,value) {
  const data=await loadIR(file),ir=structuredClone(data.ir);
  for (const a of ir.assets) delete a._file;
  const group=ir.groups.find(g=>g.id===nodeId), node=ir.nodes.find(n=>n.id===nodeId);
  if (!group && !node) throw new Error('找不到扰动目标。');
  const ids=new Set(group?group.member_ids:[nodeId]);
  if (action==='move') {
    const [dx,dy]=(value||'24,16').split(',').map(Number);
    if (![dx,dy].every(Number.isFinite)) throw new Error('移动 Value 为 dx,dy。');
    for (const n of ir.nodes.filter(n=>ids.has(n.id))) {n.bbox[0]+=dx;n.bbox[1]+=dy;}
  } else if (action==='replace') {
    if (!node || node.type!=='text' || !value) throw new Error('replace 需要文本节点和新文字。');
    node.text=value; delete node.runs; node.text_override_reason='编辑扰动测试：用户替换文字。';
  } else if (action==='delete') {
    const removed=new Set();
    for (const n of ir.nodes.filter(n=>ids.has(n.id))) {
      for (const e of n.evidence_ids) removed.add(e);
      if (n.type==='image') for (const e of ir.assets.find(a=>a.id===n.asset_id).contains_evidence_ids) removed.add(e);
    }
    const usedAssetIds=new Set(ir.nodes.filter(n=>!ids.has(n.id)).map(n=>n.asset_id).filter(Boolean));
    if (ir.background.asset_id) usedAssetIds.add(ir.background.asset_id);
    ir.nodes=ir.nodes.filter(n=>!ids.has(n.id)); ir.evidence=ir.evidence.filter(e=>!removed.has(e.id));
    ir.assets=ir.assets.filter(a=>usedAssetIds.has(a.id));
    ir.groups=ir.groups.map(g=>({...g,member_ids:g.member_ids.filter(id=>!ids.has(id))})).filter(g=>g.member_ids.length>=2);
  } else throw new Error('Action 只支持 delete/move/replace。');
  ir.review_status='needs_review'; ir.review_notes.push(`编辑扰动测试: ${action} ${nodeId}`);
  const target=path.join(data.base,`slideir.${action}.${randomUUID().slice(0,8)}.json`);
  await writeJson(target,ir);
  return {ir:target,...await compile(target)};
}

async function main() {
  const {values,positionals}=parseArgs({options:{input:{type:'string'},node:{type:'string'},action:{type:'string'},value:{type:'string'}},allowPositionals:true});
  const cmd=positionals[0];
  let result;
  if (cmd==='doctor') {
    const rt=await runtime();
    const metadata=JSON.parse(await fs.readFile(path.join(rt.node_modules,'@oai/artifact-tool/package.json'),'utf8'));
    result={engine_available:true,artifact_version:metadata.version,node_modules:rt.node_modules,python:rt.python,
      source:'Codex bundled runtime; project-local recovery if primary packages are absent',model_api_required:false,
      supported:['single_page_image','native_text','simple_shapes','image_modules','real_groups','final_pptx_reimport_preview'],
      not_implemented:['automatic_ocr','pdf_ingestion','automatic_inpainting','native_tables','native_charts']};
  } else if (cmd==='demo') result=await demo();
  else if (cmd==='compile') result=await compile(values.input);
  else if (cmd==='validate') {const d=await loadIR(values.input);result={valid:true,nodes:d.ir.nodes.length,warnings:d.info.warnings};}
  else if (cmd==='inspect') {const rt=await runtime();result=pythonJson(rt,[path.join(ROOT,'scripts/pptx_xml.py'),'inspect',values.input]);}
  else if (cmd==='perturb') result=await perturb(values.input,values.node,values.action,values.value);
  else throw new Error('命令: doctor/demo/compile/validate/inspect/perturb');
  console.log(JSON.stringify(result,null,2));
}

if (process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  main().catch(error=>{console.error(error.stack||error.message);process.exitCode=1;});
}

