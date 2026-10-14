import fs from 'node:fs/promises';
import path from 'node:path';
import { runtime } from './runtime.mjs';
import { sha256 } from './ir.mjs';
import { renderPptx } from './render.mjs';
import { compile } from './compile.mjs';
// Keep helper diagnostics off the machine-readable stdout channel.
const emit = process.stdout.write.bind(process.stdout);
process.stdout.write = process.stderr.write.bind(process.stderr);
try {
  const [cmd,...args]=process.argv.slice(2);
  let result;
  if(cmd==='render') result=await renderPptx(args[0],args[1]);
  else if(cmd==='compile') result=await compile(args[0]);
  else if(cmd==='doctor') {
    const rt=await runtime();
    const helpers={};
    for(const name of ['artifact_tool_utils.mjs','inspect_presentation_package_integrity.py','inspect_presentation_layout_geometry.py']) {
      const file=path.join(rt.skill,'container_tools',name);
      helpers[name]={path:file,sha256:sha256(await fs.readFile(file))};
    }
    result={engine:'@oai/artifact-tool',artifact_version:rt.artifact_version,node_version:process.version,
      node_modules:rt.node_modules,compiler_python:rt.python,presentations_skill:rt.skill,helpers,network_calls:0};
  } else throw new Error('Unsupported bridge command');
  emit(JSON.stringify(result)+'\n');
} catch(error) {console.error(error.stack||error.message);process.exitCode=1;}
