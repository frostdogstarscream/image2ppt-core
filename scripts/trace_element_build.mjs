// Diagnostic only: log object IDs, never source text; do not change the compiler.
import {runtime} from './runtime.mjs';
import {loadIR} from './ir.mjs';
import {buildSlide} from './slide.mjs';
import fs from 'node:fs/promises';
const data=await loadIR(process.argv[2]);
if(process.argv[3] && process.argv[3]!=='all') data.info.ordered=data.info.ordered.filter(n=>process.argv[3].split(',').includes(n.type));
if(Number(process.argv[5])>0) data.info.ordered=data.info.ordered.slice(0,Number(process.argv[5]));
if(process.argv[6]==='drop-first') data.info.ordered=data.info.ordered.slice(1);
if(process.argv[6]==='rename') data.info.ordered=data.info.ordered.map((n,i)=>({...n,id:'probe-'+i}));
if(process.argv[6]==='short-last') data.info.ordered=data.info.ordered.map((n,i,a)=>i===a.length-1?{...n,text:'Test'}:n);
if(process.argv[6]==='chinese-font') data.info.ordered=data.info.ordered.map(n=>n.type==='text'?{...n,font:{...n.font,ea:'微软雅黑',latin:'微软雅黑'}}:n);
if(process.argv[6]==='last-first') data.info.ordered=[data.info.ordered.at(-1),...data.info.ordered.slice(0,-1)];
if(process.argv[6]==='last-height') data.info.ordered=data.info.ordered.map((n,i,a)=>i===a.length-1?{...n,bbox:[...n.bbox.slice(0,3),n.bbox[3]+1]}:n);
if(process.argv[6]==='last-font') data.info.ordered=data.info.ordered.map((n,i,a)=>i===a.length-1?{...n,font:{...n.font,size_pt:10}}:n);
if(process.argv[7]) { const selected=new Set(JSON.parse(await fs.readFile(process.argv[7],'utf8')));data.info.ordered=data.info.ordered.filter(n=>selected.has(n.id)); }
if(!data.info.ordered.length) throw new Error('diagnostic selection is empty');
console.log('selected-nodes',data.info.ordered.length);
const rt=await runtime();
const p=rt.api.Presentation.create({slideSize:{width:data.ir.canvas.target_px[0],height:data.ir.canvas.target_px[1]}});
console.log('runtime-ready');
const wrapper={slides:{add(){
  const s=p.slides.add();
  return {background:s.background,speakerNotes:s.speakerNotes,
    images:{add(options){console.log('image',options.name);return s.images.add(options);}},
    shapes:{add(options){console.log('shape',options.name,options.geometry);return s.shapes.add(options);}},
    tables:s.tables,charts:s.charts};
}}};
await buildSlide(wrapper,data);
console.log('build-complete');
if(process.argv[4]) {
  const inspect=p.inspect.bind(p);
  p.inspect=async options=>{console.log('inspect-start',options.kind);const value=await inspect(options);console.log('inspect-complete');return value;};
  console.log('export-start');
  await (await rt.api.PresentationFile.exportPptx(p)).save(process.argv[4]);
  console.log('export-complete');
}
