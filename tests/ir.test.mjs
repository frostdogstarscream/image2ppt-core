import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { assertAuthoring, loadIR, localFile, sha256, validateIR } from '../scripts/ir.mjs';
import { COMPILER, IR_NODE_TYPES, IR_SHAPE_GEOMETRIES, capabilities } from '../scripts/capabilities.mjs';

function fixture() {
  return {schema_version:'0.1',canvas:{source_px:[1280,720],target_px:[1280,720]},background:{color:'FFFFFF'},
    review_status:'needs_review',review_notes:[],assets:[],groups:[],
    evidence:[{id:'ev-title',kind:'text',bbox:[20,20,500,80],text:'收入增长 18%'}],
    nodes:[{id:'title',type:'text',bbox:[20,20,500,80],z:1,evidence_ids:['ev-title'],text:'收入增长 18%',font:{latin:'Arial',ea:'Microsoft YaHei',size_pt:24},color:'222222'}]};
}

test('rejects duplicate node IDs with the exact business diagnostic',()=>{const x=fixture();x.nodes.push({...structuredClone(x.nodes[0]),z:2});assert.throws(()=>validateIR(x),/节点 ID 重复/);});
test('requires boolean node and run bold',()=>{
  for(const value of ['false',0,null]) {
    const x=fixture();x.nodes[0].bold=value;assert.throws(()=>validateIR(x),/bold 必须为 boolean/);
    delete x.nodes[0].bold;x.nodes[0].runs=[{text:x.nodes[0].text,bold:value}];assert.throws(()=>validateIR(x),/run.bold 必须为 boolean/);
  }
  const x=fixture();x.nodes[0].bold=false;x.nodes[0].runs=[{text:x.nodes[0].text,bold:true}];assert.doesNotThrow(()=>validateIR(x));
});
test('rejects unknown semantic fields at each boundary',()=>{
  for(const change of [x=>x.rotation=45,x=>x.canvas.dpi=96,x=>x.background.opacity=0.5,x=>x.nodes[0].rotation=45,x=>x.nodes[0].font.weight=700,
    x=>x.nodes[0].runs=[{text:x.nodes[0].text,italic:true}],x=>x.evidence[0].confidence=0.5]) {
    const x=fixture();change(x);assert.throws(()=>validateIR(x),/未支持字段/);
  }
});
test('rejects null or empty runs instead of silently ignoring them',()=>{
  for(const value of [null,[],false]) {const x=fixture();x.nodes[0].runs=value;assert.throws(()=>validateIR(x),/runs 必须为非空数组/);}
});

test('keeps source text and evidence ownership',()=>assert.equal(validateIR(fixture()).owners.get('ev-title'),'title'));
test('rejects silent number rewriting',()=>{const x=fixture();x.nodes[0].text='收入增长 81%';assert.throws(()=>validateIR(x),/改写源文字/);});
test('records intentional text changes',()=>{const x=fixture();x.nodes[0].text='收入增长 20%';x.nodes[0].text_override_reason='编辑测试';assert.match(validateIR(x).warnings[0],/编辑测试/);});
test('rejects source text repeated in a preserved image',()=>{const x=fixture();x.assets=[{id:'a',path:'a.png',sha256:'0'.repeat(64),purpose:'module',contains_evidence_ids:['ev-title']}];x.nodes.push({id:'image',type:'image',bbox:[20,20,500,80],z:2,evidence_ids:[],asset_id:'a'});assert.throws(()=>validateIR(x),/重复输出/);});
test('rejects omitted source content',()=>{const x=fixture();x.evidence.push({id:'forgotten',kind:'text',bbox:[20,200,400,50],text:'负号 -7%'});assert.throws(()=>validateIR(x),/未分配输出/);});
test('rejects arbitrary types and nonfinite coordinates',()=>{const x=fixture();x.nodes[0].type='execute';assert.throws(()=>validateIR(x),/不支持/);x.nodes[0].type='text';x.nodes[0].bbox[0]=Infinity;assert.throws(()=>validateIR(x),/bbox 无效/);});
test('accepts text insets and vertical anchor with strict bounds',()=>{
  const x=fixture();x.nodes[0].insets=[4,2,4,2];x.nodes[0].vertical_anchor='middle';assert.doesNotThrow(()=>validateIR(x));
  for(const value of [[1,2,3],[-1,0,0,0],['4',0,0,0],4]) {const y=fixture();y.nodes[0].insets=value;assert.throws(()=>validateIR(y),/insets 必须是四个非负像素值/);}
  for(const value of ['center','ctr','']) {const y=fixture();y.nodes[0].vertical_anchor=value;assert.throws(()=>validateIR(y),/垂直锚点无效/);}
});
test('rejects canvas stretching and out of bounds',()=>{const x=fixture();x.canvas.target_px=[1280,800];assert.throws(()=>validateIR(x),/禁止静默拉伸/);x.canvas.target_px=[1280,720];x.nodes[0].bbox[0]=1200;assert.throws(()=>validateIR(x),/超出源画布/);});
test('rejects duplicate z and unresolved content declared checked',()=>{const x=fixture();x.nodes.push({...structuredClone(x.nodes[0]),id:'other',evidence_ids:[]});assert.throws(()=>validateIR(x),/z 必须/);const y=fixture();y.evidence.push({id:'uncertain',kind:'text',bbox:[20,200,400,50],text:'?'});y.unresolved_evidence_ids=['uncertain'];y.review_status='source_checked';assert.throws(()=>validateIR(y),/needs_review/);});
test('rejects noncontiguous and multiply owned group members',()=>{const x=fixture();for(let i=2;i<=3;i++){x.evidence.push({id:'e'+i,kind:'shape',bbox:[20,200,40,40]});x.nodes.push({id:'n'+i,type:'shape',bbox:[20,200,40,40],z:i,evidence_ids:['e'+i],geometry:'rect',fill:'FFFFFF'});}x.groups=[{id:'g',member_ids:['title','n3']}];assert.throws(()=>validateIR(x),/不连续/);x.groups=[{id:'g',member_ids:['title','n2']},{id:'g2',member_ids:['n2','n3']}];assert.throws(()=>validateIR(x),/多个分组/);});
test('blocks local traversal and remote asset paths',async()=>{const temp=await fs.mkdtemp(path.join(os.tmpdir(),'image2ppt-test-'));await fs.mkdir(path.join(temp,'run'));await fs.writeFile(path.join(temp,'outside.png'),'fixture');await assert.rejects(localFile(path.join(temp,'run'),'../outside.png'),/逃逸/);await assert.rejects(localFile(path.join(temp,'run'),'https://example.com/a.png'),/本地路径/);});
test('loads the preserved synthetic input and its asset hashes',async()=>{const d=await loadIR(fileURLToPath(new URL('../examples/synthetic/slideir.json',import.meta.url)));assert.equal(d.ir.nodes.length,7);assert.equal(d.ir.groups.length,1);});
test('rejects wrong source dimensions and stale source hashes',async()=>{const original=new URL('../examples/synthetic/',import.meta.url);const temp=await fs.mkdtemp(path.join(os.tmpdir(),'image2ppt-input-'));await fs.cp(original,temp,{recursive:true});const file=path.join(temp,'slideir.json'),ir=JSON.parse(await fs.readFile(file,'utf8'));ir.canvas.source_px=[2560,1440];await fs.writeFile(file,JSON.stringify(ir));await assert.rejects(loadIR(file),/实际 PNG/);ir.canvas.source_px=[1280,720];ir.source_sha256='0'.repeat(64);await fs.writeFile(file,JSON.stringify(ir));await assert.rejects(loadIR(file),/SHA256/);});
test('accepts editable rounded gradients and rejects malformed stops',()=>{const x=fixture();x.nodes[0]={id:'panel',type:'shape',bbox:[20,20,500,80],z:1,evidence_ids:['ev-title'],geometry:'roundRect',fill:'FFF8FA',corner_radius_px:7,fill_gradient:{angle_deg:90,stops:[{offset:0,color:'FFF8FA'},{offset:100000,color:'FFF0F4'}]}};assert.doesNotThrow(()=>validateIR(x));x.nodes[0].fill_gradient.stops[1].offset=0;assert.throws(()=>validateIR(x),/色标顺序/);x.nodes[0].corner_radius_px=100;assert.throws(()=>validateIR(x),/圆角/);});

test('authoring sidecar is optional but must describe the exact IR bytes',async()=>{
  const temp=await fs.mkdtemp(path.join(os.tmpdir(),'image2ppt-authoring-'));
  const ir=path.join(temp,'slideir.json'),record=path.join(temp,'authoring.json');
  await fs.writeFile(ir,'{"schema_version":"0.1"}');
  const stamp=async value=>fs.writeFile(record,JSON.stringify({ir_sha256:value}));
  assert.equal(await assertAuthoring(ir),null,'a missing sidecar predates the record');
  const real=sha256(await fs.readFile(ir));
  await stamp(real); assert.equal((await assertAuthoring(ir)).ir_sha256,real);
  await stamp('0'.repeat(64));
  await assert.rejects(assertAuthoring(ir),/与当前 IR 不符/);
  await stamp('not-a-hash');
  await assert.rejects(assertAuthoring(ir),/有效的 ir_sha256/);
  await fs.writeFile(record,'{oops');
  await assert.rejects(assertAuthoring(ir),/不是合法 JSON/);
});

// The two skill flavors have separate maintained sources, but two mirrors must
// never drift: the installed Codex copy, and the protocol reference, which is
// flavor-independent. Nothing else enforced this, so an edit to one side could
// ship a stale skill silently.
test('skill copies stay byte-identical to their maintained sources',async()=>{
  const pairs=[
    ['../skills/image2ppt/SKILL.md','../.agents/skills/image2ppt/SKILL.md'],
    ['../skills/image2ppt/references/slideir.md','../.agents/skills/image2ppt/references/slideir.md'],
    ['../skills/image2ppt/references/core.md','../.agents/skills/image2ppt/references/core.md'],
    ['../skills/image2ppt/agents/openai.yaml','../.agents/skills/image2ppt/agents/openai.yaml'],
    ['../skills/image2ppt/references/slideir.md','../skills/image2ppt-claude/references/slideir.md'],
  ];
  for(const [left,right] of pairs){
    const a=await fs.readFile(fileURLToPath(new URL(left,import.meta.url)));
    const b=await fs.readFile(fileURLToPath(new URL(right,import.meta.url)));
    assert.ok(a.equals(b),`${left} 与 ${right} 不再逐字节一致；改完维护源后请同步镜像`);
  }
});

// A table node. The bounds below mirror native_table_arithmetic_gate.py, so an
// IR that validates here cannot produce a table the finalizer later refuses.
function tableFixture(overrides={}) {
  const x=fixture();
  x.evidence.push({id:'ev-tbl',kind:'shape',bbox:[20,120,500,120]});
  x.nodes.push({id:'tbl',type:'table',bbox:[20,120,500,120],z:2,evidence_ids:['ev-tbl'],
    rows:[['指标','本期'],['收入','1240']],
    font:{latin:'Arial',ea:'Microsoft YaHei',size_pt:14},...overrides});
  return x;
}

test('a table node validates and keeps its own evidence',()=>{
  assert.equal(validateIR(tableFixture()).owners.get('ev-tbl'),'tbl');
});

test('a table node no longer falls through to the image-asset path',()=>{
  // The branch chain used to end in `else useAsset(node.asset_id,...)`, which
  // dereferenced an undefined asset_id for any other node type.
  assert.doesNotThrow(()=>validateIR(tableFixture()));
});

test('table bounds mirror the finalizer gate',()=>{
  const long=tableFixture(); long.nodes.at(-1).rows[0][0]='x'.repeat(2049);
  assert.throws(()=>validateIR(long),/单元格/);
  const cols=tableFixture(); cols.nodes.at(-1).rows=[Array.from({length:101},()=>'c')];
  assert.throws(()=>validateIR(cols),/列数/);
  const ragged=tableFixture(); ragged.nodes.at(-1).rows=[['a','b'],['c']];
  assert.throws(()=>validateIR(ragged),/各行长度/);
  const empty=tableFixture(); empty.nodes.at(-1).rows=[];
  assert.throws(()=>validateIR(empty),/行数/);
});

test('a table must declare its font and well-formed column tracks',()=>{
  const noFont=tableFixture(); delete noFont.nodes.at(-1).font;
  assert.throws(()=>validateIR(noFont),/缺少字体/);
  const badTrack=tableFixture(); badTrack.nodes.at(-1).column_tracks=[{mode:'pct',value:1},{mode:'fr',value:1}];
  assert.throws(()=>validateIR(badTrack),/column_track/);
  const short=tableFixture(); short.nodes.at(-1).column_tracks=[{mode:'fr',value:1}];
  assert.throws(()=>validateIR(short),/column_tracks 长度/);
});

test('a table node rejects unknown fields like every other node type',()=>{
  const x=tableFixture(); x.nodes.at(-1).merge=true;
  assert.throws(()=>validateIR(x),/未支持字段/);
});

test('table merges preserve cell text and reject overlap or lost content',()=>{
  const x=tableFixture({rows:[['合并标题',''],['收入','1240']],row_heights:[40,80],merges:[{row:0,column:0,row_span:1,column_span:2}],cell_styles:[{row:0,column:0,fill:'F0F0F0',color:'000000',bold:true}]});
  assert.doesNotThrow(()=>validateIR(x));
  x.nodes.at(-1).rows[0][1]='不能覆盖';assert.throws(()=>validateIR(x),/必须为空/);
  x.nodes.at(-1).rows[0][1]='';x.nodes.at(-1).merges.push({row:0,column:0,row_span:2,column_span:1});assert.throws(()=>validateIR(x),/重叠/);
});

test('table row heights and cell styles have strict bounds',()=>{
  for(const values of [[60,'60'],[60,61],[0,120],[60]]) assert.throws(()=>validateIR(tableFixture({row_heights:values})),/row_heights/);
  for(const s of [{row:2,column:0},{row:0,column:0,bold:1},{row:0,column:0,fill:'bad'}]) assert.throws(()=>validateIR(tableFixture({cell_styles:[s]})),/cell/);
});

function chartFixture(overrides={}) {
  const x=fixture();
  x.evidence.push({id:'ev-ch',kind:'shape',bbox:[20,120,500,200]});
  x.nodes.push({id:'ch',type:'chart',bbox:[20,120,500,200],z:2,evidence_ids:['ev-ch'],
    chart_type:'bar',title:'各季度营收',categories:['春','夏'],series:[{name:'营收',values:[1,2]}],
    font:{latin:'Arial',ea:'Microsoft YaHei',size_pt:14},...overrides});
  return x;
}

test('a chart node validates and keeps its own evidence',()=>{
  assert.equal(validateIR(chartFixture()).owners.get('ev-ch'),'ch');
});

test('a chart title must be real when present',()=>{
  // The delivered package fails an empty title or the engine's placeholder.
  const empty=chartFixture(); empty.nodes.at(-1).title='   ';
  assert.throws(()=>validateIR(empty),/title 不得为空/);
  const placeholder=chartFixture(); placeholder.nodes.at(-1).title='Chart Title';
  assert.throws(()=>validateIR(placeholder),/占位符/);
  // Absent is allowed: the engine then emits no title element at all.
  const absent=chartFixture(); delete absent.nodes.at(-1).title;
  assert.doesNotThrow(()=>validateIR(absent));
});

test('chart data must be internally consistent',()=>{
  const mismatch=chartFixture(); mismatch.nodes.at(-1).series[0].values=[1,2,3];
  assert.throws(()=>validateIR(mismatch),/长度必须等于/);
  const noName=chartFixture(); noName.nodes.at(-1).series[0].name='';
  assert.throws(()=>validateIR(noName),/series 缺少名称/);
  const badType=chartFixture(); badType.nodes.at(-1).chart_type='treemap';
  assert.throws(()=>validateIR(badType),/图表类型不支持/);
  const noFont=chartFixture(); delete noFont.nodes.at(-1).font;
  assert.throws(()=>validateIR(noFont),/缺少字体/);
});

test('a chart node rejects unknown fields',()=>{
  const x=chartFixture(); x.nodes.at(-1).stacked=true;
  assert.throws(()=>validateIR(x),/未支持字段/);
});

// A capability list is only honest if it is derived from the code rather than
// maintained beside it. These three hold the derivation together: the IR schema
// follows the flags, the flags follow the builder, and every native flag lands
// in exactly one doctor list. The dangerous drift is a capability that exists in
// code but not in the list — the skill then rasterises something it could have
// rebuilt, silently.
test('compiler capability flags agree with the code that implements them',async()=>{
  const builder=await fs.readFile(fileURLToPath(new URL('../scripts/slide.mjs',import.meta.url)),'utf8');
  for(const [flag,marker] of [['native_tables','tables.add'],['native_charts','charts.add']]){
    assert.equal(COMPILER[flag],builder.includes(marker),
      `${flag} 声明为 ${COMPILER[flag]}，但 slide.mjs ${builder.includes(marker)?'含':'不含'} ${marker}；能力表必须与实现同步`);
  }
});

// The shape catalogue is a contract between the validator and the protocol
// reference. Widening one without the other either rejects a shape the author
// was told to use, or advertises one the compiler will refuse.
test('the documented shape geometries match the ones the IR accepts',async()=>{
  const doc=await fs.readFile(fileURLToPath(new URL('../skills/image2ppt/references/slideir.md',import.meta.url)),'utf8');
  for(const geometry of IR_SHAPE_GEOMETRIES){
    assert.match(doc,new RegExp('`'+geometry+'`'),`slideir.md 未记录几何 ${geometry}`);
  }
});

// Dropping these silently would put the flow-diagram class back out of reach,
// which is the regression this catalogue was widened to fix.
test('the flow-diagram geometries stay in the catalogue',()=>{
  for(const geometry of ['chevron','pentagon','rightArrow','leftArrow']){
    assert.ok(IR_SHAPE_GEOMETRIES.includes(geometry),`${geometry} 必须保留在形状白名单里`);
  }
});

test('the IR node types derive from the capability flags',()=>{
  assert.ok(IR_NODE_TYPES.includes('text'));
  assert.equal(IR_NODE_TYPES.includes('table'),COMPILER.native_tables);
  assert.equal(IR_NODE_TYPES.includes('chart'),COMPILER.native_charts);
});

test('every native capability lands in exactly one doctor list',()=>{
  const {supported,not_implemented}=capabilities();
  for(const flag of ['native_tables','native_charts']){
    assert.notEqual(supported.includes(flag),not_implemented.includes(flag),`${flag} 必须恰好出现在一张表里`);
  }
  assert.equal(new Set(supported).size,supported.length,'supported 不应重复');
  assert.equal(new Set(not_implemented).size,not_implemented.length,'not_implemented 不应重复');
});

test('only the layered pipeline claims automatic OCR and inpainting',()=>{
  const replay=capabilities();
  const layered=capabilities({pipeline:'layered'});
  for(const item of ['automatic_ocr','automatic_inpainting']){
    assert.ok(replay.not_implemented.includes(item)&&!replay.supported.includes(item),`${item} 在默认流程中仍未实现`);
    assert.ok(layered.supported.includes(item)&&!layered.not_implemented.includes(item),`${item} 属于 layered 流程`);
  }
  assert.ok(layered.not_implemented.includes('pdf_ingestion'));
  assert.throws(()=>capabilities({pipeline:'unknown'}),/unknown pipeline/);
});

// The policy must not silently revert to cropping whole complex regions. Two
// wordings have caused that, and both must stay gone: the original
// unconditioned instruction, and the escape hatch that let one resisting
// sub-region collapse an entire region back into a single crop.
test('the skill asks for per-sub-region decomposition, not whole-region crops',async()=>{
  const files=['../skills/image2ppt/SKILL.md','../skills/image2ppt-claude/SKILL.md'];
  for(const file of files){
    const text=await fs.readFile(fileURLToPath(new URL(file,import.meta.url)),'utf8');
    assert.doesNotMatch(text,/Keep complex visual modules as cropped images/,'整块裁图的无条件规则不得回归');
    assert.doesNotMatch(text,/Only when the background itself cannot be rebuilt does the whole region/,'整块保留的逃生舱不得回归');
    assert.match(text,/[Dd]ecompos/,`${file} 应保留下拆解要求`);
    assert.match(text,/sub-region/,`${file} 应按子区域判定`);
    assert.match(text,/part native and part crop/,`${file} 应说明半原生半裁剪是正常结果`);
    // Require source/actual-PPTX evidence and an explicit human goal. A fixed
    // MAE cutoff must not silently overturn a human's editability decision.
    assert.match(text,/candidate_workflow/,`${file} 应接入候选复检流程`);
    assert.match(text,/human.*(?:fidelity|scope)|fidelity.*human/i,`${file} 应遵循人工保真和编辑范围`);
    assert.doesNotMatch(text,/above that it goes back to a crop/,`${file} 不得按单一像素阈值自动回退`);
  }
});

