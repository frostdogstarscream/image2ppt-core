// Diagnostic re-import at a declared scale; never changes the supplied PPTX.
import fs from 'node:fs/promises';
import path from 'node:path';
import { runtime, saveBlob } from './runtime.mjs';
import { sha256 } from './ir.mjs';

const [file, directory, scaleText] = process.argv.slice(2);
const scale = Number(scaleText);
if (!file || !directory || !Number.isFinite(scale) || scale <= 0 || scale > 4) {
  throw new Error('PPTX, new output directory and scale in (0,4] are required');
}
const before = sha256(await fs.readFile(file));
const {api:{PresentationFile,FileBlob}} = await runtime();
const imported = await PresentationFile.importPptx(await FileBlob.load(file));
await fs.mkdir(directory, {recursive:false});
const pages=[];
for (let i=0;i<imported.slides.items.length;i++) {
  const target=path.join(directory,`page-${String(i+1).padStart(3,'0')}.png`);
  await saveBlob(target,await imported.export({slide:imported.slides.items[i],format:'png',scale}));
  pages.push({slide:i+1,path:target,sha256:sha256(await fs.readFile(target))});
}
if (!pages.length || before!==sha256(await fs.readFile(file))) throw new Error('Input changed or empty presentation');
console.log(JSON.stringify({origin:'reimported_final_pptx',renderer:'Artifact Tool',scale,pptx_sha256:before,pages}));
