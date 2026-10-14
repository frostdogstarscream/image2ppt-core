import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';

export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const expect = (ok, message) => { if (!ok) throw new Error(message); };
const color = value => typeof value === 'string' && /^[0-9a-fA-F]{6}$/.test(value);
const array = value => Array.isArray(value);
const fields = (value, allowed, label) => {
  expect(value && typeof value==='object' && !array(value), `${label} 必须为对象。`);
  const unknown=Object.keys(value).filter(k=>!allowed.split(' ').includes(k));
  expect(!unknown.length, `${label} 未支持字段: ${unknown.join(', ')}。`);
};

export function validateIR(ir) {
  fields(ir,'schema_version source_image source_sha256 canvas background review_status review_notes nodes evidence assets groups unresolved_evidence_ids','SlideIR');
  fields(ir.canvas,'source_px target_px','canvas');
  fields(ir.background,'color asset_id','background');
  if(ir.source_image!==undefined) expect(typeof ir.source_image==='string' && ir.source_image.length>0,'source_image 无效。');
  if(ir.source_sha256!==undefined) expect(typeof ir.source_sha256==='string' && /^[a-f0-9]{64}$/.test(ir.source_sha256),'source_sha256 无效。');
  expect(ir?.schema_version === '0.1', '只支持 SlideIR 0.1。');
  expect(array(ir.canvas?.source_px) && ir.canvas.source_px.length===2 && array(ir.canvas?.target_px) && ir.canvas.target_px.length===2, '缺少有效的源画布与目标画布。');
  const [w,h] = ir.canvas.source_px, [tw,th] = ir.canvas.target_px;
  expect([w,h,tw,th].every(x => Number.isFinite(x) && x >= 1 && x <= 40000), '画布尺寸无效。');
  expect(Math.abs(tw*h/w-th) <= 1, '源图与目标画布比例不同；禁止静默拉伸。');
  expect(tw <= 8192 && th <= 8192, '目标画布超过 8192px。');
  expect(color(ir.background?.color), 'background.color 必须为六位 RGB。');
  expect(['needs_review','source_checked'].includes(ir.review_status), '缺少有效 review_status。');
  expect(array(ir.review_notes) && ir.review_notes.every(x=>typeof x==='string'), 'review_notes 必须为字符串数组。');
  for (const key of ['nodes','evidence','assets','groups']) expect(array(ir[key]), `${key} 必须为数组。`);
  for(const e of ir.evidence) fields(e,'id kind bbox text description','evidence');
  for(const a of ir.assets) {
    fields(a,'id path purpose sha256 contains_evidence_ids description','asset');
    expect(typeof a.path==='string' && a.path.length>0,'asset.path 无效。');
  }
  for(const g of ir.groups) fields(g,'id member_ids','group');
  for(const entry of [...ir.evidence,...ir.assets]) if(entry.description!==undefined) expect(typeof entry.description==='string','description 必须为字符串。');
  for(const node of ir.nodes) {
    expect(['text','shape','image'].includes(node?.type), `${node?.id} 不支持的节点类型 ${node?.type}。`);
    const common='id type bbox z evidence_ids';
    const byType={text:'text font bold color align wrap line_spacing_pt runs text_override_reason source_text_bbox',
      shape:'geometry fill line corner_radius_px fill_gradient',image:'asset_id'};
    fields(node,common+' '+(byType[node?.type]||''),'node');
    if(node.type==='text') {
      fields(node.font,'latin ea size_pt','font');
      if(node.bold!==undefined) expect(typeof node.bold==='boolean',`${node.id} bold 必须为 boolean。`);
      if(node.text_override_reason!==undefined) expect(typeof node.text_override_reason==='string' && node.text_override_reason.trim(),`${node.id} text_override_reason 无效。`);
      if(node.runs!==undefined) {
        expect(array(node.runs) && node.runs.length>0,`${node.id} runs 必须为非空数组。`);
        for(const run of node.runs) {
          fields(run,'text color bold','run');
          if(run.bold!==undefined) expect(typeof run.bold==='boolean',`${node.id} run.bold 必须为 boolean。`);
        }
      }
    }
    if(node.line!==undefined) fields(node.line,'color width_pt','line');
    if(node.fill_gradient!==undefined) {
      fields(node.fill_gradient,'angle_deg stops','fill_gradient');
      expect(array(node.fill_gradient.stops),'gradient.stops 必须为数组。');
      for(const stop of node.fill_gradient.stops) fields(stop,'offset color','gradient.stop');
    }
  }
  expect(ir.nodes.length > 0 && ir.nodes.length <= 500, '页面需要 1—500 个节点；准备模板不能直接交付。');
  expect(ir.evidence.length > 0, '必须建立源内容清单。');
  const idCheck = (items, label) => {
    const ids = items.map(x=>x.id);
    expect(ids.every(x=>typeof x==='string' && /^[A-Za-z0-9_-]{1,80}$/.test(x)), `${label} ID 无效。`);
    expect(new Set(ids).size===ids.length, `${label} ID 重复。`);
  };
  idCheck(ir.nodes, '节点'); idCheck(ir.evidence, '证据'); idCheck(ir.assets, '资产'); idCheck(ir.groups, '分组');
  expect(new Set([...ir.nodes,...ir.groups].map(x=>x.id)).size===ir.nodes.length+ir.groups.length, '分组与节点 ID 不能冲突。');
  const boxCheck = (box, id) => {
    expect(array(box) && box.length===4 && box.every(Number.isFinite), `${id} bbox 无效。`);
    const [x,y,bw,bh] = box;
    expect(x>=0 && y>=0 && bw>0 && bh>0 && x+bw<=w+.01 && y+bh<=h+.01, `${id} 超出源画布或尺寸非正。`);
  };
  const evidence = new Map(ir.evidence.map(e=>[e.id,e]));
  for (const e of ir.evidence) {
    boxCheck(e.bbox,e.id);
    expect(['text','shape','image'].includes(e.kind), `${e.id} 证据类型无效。`);
    if (e.kind==='text') expect(typeof e.text==='string', `${e.id} 缺少源文字。`);
  }
  const owners = new Map(), usedAssets = new Set();
  const claim = (ids, owner) => {
    expect(array(ids), `${owner} 的 evidence_ids 必须为数组。`);
    for (const id of ids) {
      expect(evidence.has(id), `${owner} 引用了不存在的证据 ${id}。`);
      expect(!owners.has(id), `${id} 重复输出：${owners.get(id)} 与 ${owner}。`);
      owners.set(id,owner);
    }
  };
  const assets = new Map(ir.assets.map(a=>[a.id,a]));
  const useAsset = (id,owner) => {
    expect(assets.has(id), `缺少资产 ${id}。`);
    expect(!usedAssets.has(id), `资产 ${id} 被重复使用。`);
    usedAssets.add(id);
    const a=assets.get(id);
    expect(['module','cleaned_background'].includes(a.purpose), `${id} 资产用途无效。`);
    expect(typeof a.sha256==='string' && /^[a-f0-9]{64}$/.test(a.sha256), `${id} 缺少 SHA256。`);
    claim(a.contains_evidence_ids, owner);
  };
  if (ir.background.asset_id) {
    useAsset(ir.background.asset_id,'background');
    expect(assets.get(ir.background.asset_id).purpose==='cleaned_background', '背景资产必须声明 cleaned_background。');
  }
  const zs = new Set();
  for (const node of ir.nodes) {
    expect(['text','shape','image'].includes(node.type), `${node.id} 不支持的节点类型 ${node.type}。`);
    boxCheck(node.bbox,node.id);
    if(node.source_text_bbox!==undefined) boxCheck(node.source_text_bbox,node.id+' source_text_bbox');
    expect(Number.isInteger(node.z) && !zs.has(node.z), `${node.id} z 必须是唯一整数。`); zs.add(node.z);
    claim(node.evidence_ids,node.id);
    if (node.type==='text') {
      expect(typeof node.text==='string' && node.text.length>0, `${node.id} 缺少文字。`);
      expect(node.font && ['latin','ea'].every(k=>typeof node.font[k]==='string' && node.font[k].trim()), `${node.id} 缺少字体。`);
      expect(Number.isFinite(node.font.size_pt) && node.font.size_pt>0 && node.font.size_pt<=400, `${node.id} 字号无效。`);
      expect(color(node.color), `${node.id} 文字颜色无效。`);
      expect(node.align===undefined || ['left','center','right'].includes(node.align), `${node.id} 对齐无效。`);
      expect(node.wrap===undefined || typeof node.wrap==='boolean', `${node.id} wrap 无效。`);
      if (node.line_spacing_pt!==undefined) expect(Number.isFinite(node.line_spacing_pt) && node.line_spacing_pt>0, `${node.id} 行距无效。`);
      const texts=node.evidence_ids.map(id=>evidence.get(id)).filter(e=>e.kind==='text').map(e=>e.text);
      expect(texts.length>0, `${node.id} 没有源文字证据。`);
      if (texts.join('\n')!==node.text) expect(typeof node.text_override_reason==='string' && node.text_override_reason.trim(), `${node.id} 改写源文字但未说明原因。`);
      if (node.runs) {
        expect(!node.text.includes('\n') && array(node.runs) && node.runs.map(r=>r.text).join('')===node.text, `${node.id} runs 必须是与 text 一致的单行。`);
        expect(node.runs.every(r=>typeof r.text==='string' && (r.color===undefined || color(r.color))), `${node.id} run 无效。`);
      }
    } else if (node.type==='shape') {
      expect(['rect','roundRect','ellipse','line'].includes(node.geometry), `${node.id} 形状不支持。`);
      expect(node.fill==='none' || color(node.fill), `${node.id} 填充无效。`);
      if (node.line) expect(color(node.line.color) && Number.isFinite(node.line.width_pt) && node.line.width_pt>=0, `${node.id} 线条无效。`);
      if (node.corner_radius_px!==undefined) expect(Number.isFinite(node.corner_radius_px) && node.corner_radius_px>=0 && node.corner_radius_px<=Math.min(node.bbox[2],node.bbox[3])/2, `${node.id} 圆角半径无效。`);
      if (node.fill_gradient) {
        const g=node.fill_gradient;
        expect(Number.isFinite(g.angle_deg) && array(g.stops) && g.stops.length>=2 && g.stops.length<=8, `${node.id} 渐变无效。`);
        expect(g.stops.every(s=>Number.isFinite(s.offset) && s.offset>=0 && s.offset<=100000 && color(s.color)), `${node.id} 渐变色标无效。`);
        expect(g.stops[0].offset===0 && g.stops.at(-1).offset===100000 && g.stops.every((s,i)=>!i || s.offset>g.stops[i-1].offset), `${node.id} 渐变色标顺序无效。`);
      }
    } else useAsset(node.asset_id,node.id);
  }
  expect(usedAssets.size===assets.size, '存在未使用资产；请移除旧资产声明。');
  const unresolved = ir.unresolved_evidence_ids || [];
  claim(unresolved,'unresolved');
  expect(!unresolved.length || ir.review_status==='needs_review', '有未处理证据时必须 needs_review。');
  expect(owners.size===evidence.size, '存在未分配输出的源内容。');
  const ordered=[...ir.nodes].sort((a,b)=>a.z-b.z), grouped=new Set();
  for (const group of ir.groups) {
    expect(array(group.member_ids) && group.member_ids.length>=2, `${group.id} 至少需要两个成员。`);
    const indexes=group.member_ids.map(id=>ordered.findIndex(n=>n.id===id));
    expect(indexes.every(i=>i>=0), `${group.id} 成员不存在。`);
    expect(new Set(group.member_ids).size===group.member_ids.length, `${group.id} 成员重复。`);
    const sorted=[...indexes].sort((a,b)=>a-b);
    expect(sorted.at(-1)-sorted[0]+1===sorted.length, `${group.id} 成员在 z 顺序中不连续，不能保持原遮挡顺序。`);
    for (const id of group.member_ids) { expect(!grouped.has(id), `${id} 属于多个分组。`); grouped.add(id); }
  }
  return {ordered, assets, owners, warnings: [
    ...ir.review_notes,
    ...ir.nodes.filter(n=>n.text_override_reason).map(n=>`${n.id}: ${n.text_override_reason}`),
    ...(unresolved.length ? [`未处理源内容: ${unresolved.join(', ')}`] : [])
  ]};
}

export async function localFile(base, relative) {
  expect(typeof relative==='string' && relative.length>0 && !path.isAbsolute(relative) && !/^[A-Za-z][A-Za-z0-9+.-]*:/.test(relative), '资产路径必须为相对本地路径。');
  const root=await fs.realpath(base), file=await fs.realpath(path.resolve(base,relative));
  const rel=path.relative(root,file);
  expect(rel && !rel.startsWith('..') && !path.isAbsolute(rel), '资产路径逃逸运行目录。');
  return file;
}

export async function loadIR(file) {
  file=path.resolve(file);
  const bytes=await fs.readFile(file), ir=JSON.parse(bytes.toString('utf8').replace(/^\uFEFF/,''));
  const info=validateIR(ir), base=path.dirname(file);
  const source=await localFile(base,ir.source_image);
  const sourceBytes=await fs.readFile(source);
  expect(sha256(sourceBytes)===ir.source_sha256, '源页面 SHA256 不一致。');
  expect(sourceBytes.length>=24 && sourceBytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])), 'source_image 必须为 prepare 生成的 PNG 页面。');
  expect(sourceBytes.readUInt32BE(16)===ir.canvas.source_px[0] && sourceBytes.readUInt32BE(20)===ir.canvas.source_px[1], 'IR 源画布尺寸与实际 PNG 不一致。');
  for (const asset of ir.assets) {
    const assetFile=await localFile(base,asset.path);
    expect(['.png','.jpg','.jpeg'].includes(path.extname(assetFile).toLowerCase()), '只嵌入 PNG/JPEG 资产。');
    expect(sha256(await fs.readFile(assetFile))===asset.sha256, `${asset.id} 资产 SHA256 不一致。`);
    asset._file=assetFile;
  }
  return {ir,info,file,base,source,ir_sha256:sha256(bytes)};
}

export const AUTHORING_FILE='authoring.json';

// Authoring sidecar: binds an author to one exact IR revision. Absent is allowed
// (runs prepared before the sidecar existed); present but stale is a hard
// failure, because the IR was edited after it was recorded.
export async function assertAuthoring(file) {
  file=path.resolve(file);
  const record=path.join(path.dirname(file),AUTHORING_FILE);
  let bytes;
  try { bytes=await fs.readFile(record); }
  catch (error) { if (error.code==='ENOENT') return null; throw error; }
  const text=bytes.toString('utf8');
  let authoring;
  try { authoring=JSON.parse(text.charCodeAt(0)===0xFEFF?text.slice(1):text); }
  catch { throw new Error(AUTHORING_FILE+' 不是合法 JSON。'); }
  const recorded=authoring?.ir_sha256;
  if (typeof recorded!=='string' || !/^[a-f0-9]{64}$/.test(recorded)) throw new Error(AUTHORING_FILE+' 缺少有效的 ir_sha256。');
  const actual=sha256(await fs.readFile(file));
  if (recorded!==actual) {
    throw new Error(AUTHORING_FILE+' 与当前 IR 不符：记录 '+recorded.slice(0,12)+'，实际 '+actual.slice(0,12)+'。完成 IR 后请重新记录 authoring。');
  }
  return authoring;
}

