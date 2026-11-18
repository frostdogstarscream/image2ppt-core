// Adapter only: reuse the fixed compiler's existing bounded batch exporter.
import fs from 'node:fs/promises';
import {compose} from '../../scripts/compose.mjs';
const [manifest, output] = process.argv.slice(2);
if (!manifest || !output) throw new Error('Expected IR manifest and new output root');
const files = JSON.parse(await fs.readFile(manifest, 'utf8'));
if (!Array.isArray(files) || files.length < 1 || files.some(x => typeof x !== 'string')) throw new Error('Invalid IR manifest');
// The fixed compiler's batch writer only supports these three node types.
// Its standard writer preserves native tables, charts and their part relations.
const pages = await Promise.all(files.map(async file => JSON.parse(await fs.readFile(file, 'utf8'))));
const batchable = pages.every(page => !page.background.asset_id &&
  page.nodes.every(node => ['text', 'shape', 'image'].includes(node.type)));
console.log(JSON.stringify(await compose(files, output, {exportBatchSize: batchable ? 1 : 0}), null, 2));
