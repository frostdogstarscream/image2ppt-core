// Adapter only: reuse the fixed compiler's bounded batch exporter when eligible,
// falling back to standard compose when native tables/charts or background assets are present.
import fs from 'node:fs/promises';
import {compose} from '../../scripts/compose.mjs';
const [manifest, output] = process.argv.slice(2);
if (!manifest || !output) throw new Error('Expected IR manifest and new output root');
const files = JSON.parse(await fs.readFile(manifest, 'utf8'));
if (!Array.isArray(files) || files.length < 1 || files.some(x => typeof x !== 'string')) throw new Error('Invalid IR manifest');
const irs = await Promise.all(files.map(async f => JSON.parse(await fs.readFile(f, 'utf8'))));
const needsFullExport = irs.some(ir => ir.background?.asset_id || ir.nodes?.some(n => !['text', 'shape', 'image'].includes(n.type)));
console.log(JSON.stringify(await compose(files, output, {exportBatchSize: needsFullExport ? 0 : 1}), null, 2));
