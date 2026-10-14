import path from 'node:path';
import fs from 'node:fs/promises';
import { runtime, saveBlob } from './runtime.mjs';
import { sha256 } from './ir.mjs';

export async function renderPptx(file, directory) {
  const {api:{PresentationFile,FileBlob}}=await runtime();
  const before=sha256(await fs.readFile(file));
  const imported=await PresentationFile.importPptx(await FileBlob.load(file));
  await fs.mkdir(directory,{recursive:false});
  const pages=[];
  for(let i=0;i<imported.slides.items.length;i++) {
    const target=path.join(directory,`page-${String(i+1).padStart(3,'0')}.png`);
    await saveBlob(target,await imported.export({slide:imported.slides.items[i],format:'png',scale:1}));
    pages.push({slide:i+1,path:target,sha256:sha256(await fs.readFile(target))});
  }
  if(!pages.length || before!==sha256(await fs.readFile(file))) throw new Error('Empty presentation or input changed during render');
  return {renderer:'Artifact Tool',origin:'reimported_final_pptx',pptx_sha256:before,pages};
}
