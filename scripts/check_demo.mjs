import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { runtime, ROOT, writeJson } from './runtime.mjs';
import { perturb } from './main.mjs';
import { pythonJson } from './compile.mjs';

const input=path.resolve(process.argv[2]);
const run=path.dirname(input),rt=await runtime();
const manifest=JSON.parse(await fs.readFile(path.join(run,'demo.json'),'utf8'));
const before=await fs.readFile(manifest.pptx),cases=[];
for (const [node,action,value] of [['title','delete',null],['title','replace','2027 年业务规划'],['delivery-card','move','30,20'],['visual-module','move','24,16']]) {
  const result=await perturb(input,node,action,value);
  const quality=JSON.parse(await fs.readFile(result.quality,'utf8'));
  const slide=quality.package_checks.slides[0];
  if (action==='delete') {assert(!slide.names.includes('title'));assert(!slide.text.includes('2026 年业务规划'));}
  if (action==='replace') {assert(slide.text.includes('2027 年业务规划'));assert(!slide.text.includes('2026 年业务规划'));}
  if (node==='delivery-card') assert.equal(slide.groups,1);
  cases.push({node,action,...result});
}
assert.equal(Buffer.compare(before,await fs.readFile(manifest.pptx)),0,'原始结果被改动');
const list=path.join(run,'perturbation-cases.json');await writeJson(list,cases);
const checks=pythonJson(rt,[path.join(ROOT,'scripts/check_pixels.py'),list,manifest.source||path.join(run,'source/page.png'),manifest.preview]);
await writeJson(path.join(run,'checks.json'),{synthetic:true,original_preserved:true,pixel_checks:checks,cases,
  limitations:['非盲测；IR 来自合成夹具。','扰动通过 IR 重编译；PowerPoint/WPS 人工编辑未验证。']});
console.log(JSON.stringify({passed:true,report:path.join(run,'checks.json'),checks},null,2));

