// Re-import a final PPTX with the project's Artifact Tool runtime and export one PNG per slide.
// Usage: node render_pptx.mjs <pptx> <output-dir>
import { renderPptx } from '../../../scripts/render.mjs';

const [file, directory] = process.argv.slice(2);
if (!file || !directory) {
  console.error('usage: node render_pptx.mjs <pptx> <new-output-dir>');
  process.exit(2);
}
const result = await renderPptx(file, directory);
console.log(JSON.stringify({ renderer: result.renderer, origin: result.origin, pages: result.pages.length, pptx_sha256: result.pptx_sha256 }));
