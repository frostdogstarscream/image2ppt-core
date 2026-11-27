import fs from 'node:fs/promises';
import {perturb} from './main.mjs';
import {sha256} from './ir.mjs';
const root='outputs/supervised-elements-20261019';
const cases=[];
for(const page of ['02','03','05']) {
  const file=`${root}/calibrated-${page}/slideir.json`;const before=await fs.readFile(file);const ir=JSON.parse(before);
  const card=ir.nodes.find(n=>n.type==='shape'&&n.geometry==='roundRect'&&n.fill!=='FFFFFF'&&n.bbox[2]*n.bbox[3]>10000&&n.bbox[2]*n.bbox[3]<100000);
  const icon=ir.nodes.find(n=>n.type==='image'&&n.bbox[0]+n.bbox[2]+24<ir.canvas.source_px[0]);
  const text=ir.nodes.find(n=>n.type==='text'&&n.text.length>12);
  for(const [node,action,value] of [[card,'delete',null],[icon,'move','24,0'],[text,'replace',`原生文字编辑验证-${page}`]]) {
    const result=await perturb(file,node.id,action,value,{exportBatchSize:80});
    const quality=JSON.parse(await fs.readFile(result.quality,'utf8'));const actual=quality.package_checks.slides[0];
    if(action==='delete'&&actual.names.includes(node.id))throw new Error('deleted object remains');
    if(action==='replace'&&!actual.text.includes(value))throw new Error('native text replacement missing');
    cases.push({page,action,node_id:node.id,original_bbox:node.bbox,value,...result});
  }
  if(sha256(before)!==sha256(await fs.readFile(file)))throw new Error('original IR changed');
}
await fs.writeFile(`${root}/edit-probes.json`,JSON.stringify({originals_preserved:true,cases,human_acceptance:false},null,2));
console.log('Nine actual-file edit probes compiled and inspected.');
