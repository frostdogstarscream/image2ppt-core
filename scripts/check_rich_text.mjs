// Real-file regression. Requires the local Artifact runtime; never runs in CI.
import fs from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { ROOT, runtime, writeJson } from './runtime.mjs';
import { compile, pythonJson } from './compile.mjs';

const run=path.join(ROOT,'runs','rich-text-'+randomUUID());
await fs.mkdir(run,{recursive:true});
const fixture=path.join(ROOT,'examples/synthetic');
for(const dir of ['source','assets']) await fs.cp(path.join(fixture,dir),path.join(run,dir),{recursive:true});
const ir=JSON.parse(await fs.readFile(path.join(fixture,'slideir.json'),'utf8'));
const node=ir.nodes.find(n=>n.type==='text');
node.bold=true; node.color='172B4D';
node.runs=[{text:'红色 ',color:'FF0000',bold:true},{text:'Mixed ',color:'008000',bold:false},
           {text:'正常 ',bold:false},{text:'继承 Default'}];
node.text=node.runs.map(r=>r.text).join('');
for(const e of ir.evidence) if(node.evidence_ids.includes(e.id)) e.text=node.text;
ir.review_notes=['Synthetic style regression only; not a fidelity evaluation.'];
await writeJson(path.join(run,'slideir.json'),ir);
const result=await compile(path.join(run,'slideir.json'));
const rt=await runtime();
const check=pythonJson(rt,[path.join(ROOT,'scripts/check_rich_text_xml.py'),result.pptx,node.id]);
await writeJson(path.join(run,'verification.json'),{synthetic:true,...result,...check,human_acceptance:false});
console.log(JSON.stringify({run,...result,...check},null,2));
