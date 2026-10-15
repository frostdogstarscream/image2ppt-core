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
  }
});

