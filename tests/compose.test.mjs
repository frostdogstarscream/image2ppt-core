import test from 'node:test';
import assert from 'node:assert/strict';
import {validateCompositionCanvases} from '../scripts/compose.mjs';

const page = (source, target=[1280,720]) => ({file:'fixture',ir:{canvas:{source_px:source,target_px:target},unresolved_evidence_ids:[]}});
test('composition permits independently validated sources at the same target size',()=>{
  assert.doesNotThrow(()=>validateCompositionCanvases([page([1672,941]),page([1672,940])]));
});
test('composition still rejects mixed target canvases and unresolved evidence',()=>{
  assert.throws(()=>validateCompositionCanvases([page([1600,900]),page([1200,900],[960,720])]));
  const unresolved=page([1600,900]);unresolved.ir.unresolved_evidence_ids=['missing'];
  assert.throws(()=>validateCompositionCanvases([unresolved]));
});
