import fs from 'node:fs/promises';
import path from 'node:path';
import {parseArgs} from 'node:util';
import {compose} from './compose.mjs';
import {runtime, ROOT, writeJson} from './runtime.mjs';
import {pythonJson} from './compile.mjs';

const {values,positionals}=parseArgs({options:{output:{type:'string'},'batch-size':{type:'string',default:'80'}},allowPositionals:true});
if(!values.output || !positionals.length) throw new Error('Supply page folders and --output new output root.');
const batchSize=Number(values['batch-size']);
if(!Number.isInteger(batchSize)||batchSize<1||batchSize>100) throw new Error('Invalid batch size.');
const rt=await runtime(),guard=path.join(ROOT,'scripts/text_ledger.py');
const before=positionals.map(folder=>pythonJson(rt,[guard,'verify','--folder',folder]));
const result=await compose(positionals.map(folder=>path.join(folder,'slideir.json')),values.output,{exportBatchSize:batchSize});
const after=positionals.map((folder,index)=>pythonJson(rt,[guard,'verify','--folder',folder,'--pptx',result.pptx,'--slide',String(index+1)]));
await writeJson(path.join(result.output_dir,'text-ledger-verification.json'),{status:'PASS',preflight:before,actual_pptx:after,human_acceptance:false});
console.log(JSON.stringify(result,null,2));
