import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {ROOT,writeJson} from './runtime.mjs';
import {buildSlide} from './slide.mjs';

// Explicit recovery path for a reproducible upstream export stall. Every visual
// object is still authored by Artifact Tool, with unchanged names and styling.
export async function exportBatched(rt,loaded,raw,size) {
  if(!Number.isInteger(size)||size<1||size>100) throw new Error('Native batch size must be 1..100.');
  for(const d of loaded) if(d.ir.background.asset_id||d.ir.nodes.some(n=>!['text','shape','image'].includes(n.type))) throw new Error('Native batching supports text/shape/image nodes without background assets only.');
  const [width,height]=loaded[0].ir.canvas.target_px;
  const presentation=rt.api.Presentation.create({slideSize:{width,height}});
  const part=(d,start)=>({...d,info:{...d.info,ordered:d.info.ordered.slice(start,start+size)}});
  for(const data of loaded) await buildSlide(presentation,part(data,0));
  const base=path.join(path.dirname(raw),'batch-base.pptx');
  await (await rt.api.PresentationFile.exportPptx(presentation)).save(base);
  const batches=[];
  for(let index=0;index<loaded.length;index++) {
    const data=loaded[index];
    for(let start=size;start<data.info.ordered.length;start+=size) {
      const fragment=rt.api.Presentation.create({slideSize:{width,height}});
      await buildSlide(fragment,part(data,start));
      const file=path.join(path.dirname(raw),`batch-${index+1}-${start}.pptx`);
      await (await rt.api.PresentationFile.exportPptx(fragment)).save(file);
      batches.push({slide:index+1,path:file});
    }
  }
  const manifest=path.join(path.dirname(raw),'batch-manifest.json');
  await writeJson(manifest,{base,batches,irs:loaded.map(d=>d.file)});
  const result=spawnSync(rt.python,['-X','utf8',path.join(ROOT,'scripts/merge_native_batches.py'),manifest,raw],{encoding:'utf8',windowsHide:true,maxBuffer:4*1024*1024});
  if(result.error||result.status!==0) throw new Error(result.error?.message||result.stderr||result.stdout);
  const receipt=JSON.parse(result.stdout);receipt.batch_size=size;
  await writeJson(raw+'.batch-export.json',receipt);
  return receipt;
}
