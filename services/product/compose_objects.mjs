// Adapter only: reuse the fixed compiler's existing bounded batch exporter.
import fs from 'node:fs/promises';
import {compose} from '../../scripts/compose.mjs';
const [manifest, output] = process.argv.slice(2);
if (!manifest || !output) throw new Error('Expected IR manifest and new output root');
const files = JSON.parse(await fs.readFile(manifest, 'utf8'));
if (!Array.isArray(files) || files.length < 1 || files.some(x => typeof x !== 'string')) throw new Error('Invalid IR manifest');
console.log(JSON.stringify(await compose(files, output, {exportBatchSize:1}),null,2));
