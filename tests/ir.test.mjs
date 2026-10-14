import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { validateIR, localFile, loadIR } from '../scripts/ir.mjs';

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

