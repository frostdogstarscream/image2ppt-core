// Use the maintained compiler and its bounded export mode without a second exporter.
import path from 'node:path';
import { compile } from '../scripts/compile.mjs';
import { compose } from '../scripts/compose.mjs';

const emit = process.stdout.write.bind(process.stdout);
process.stdout.write = process.stderr.write.bind(process.stderr);
try {
  const [command, ...args] = process.argv.slice(2);
  let result;
  if (command === 'compile' && args.length === 1) {
    result = await compile(path.resolve(args[0]), { exportBatchSize: 24 });
  } else if (command === 'compose' && args.length >= 2) {
    const [output, ...files] = args;
    result = await compose(files, output, { exportBatchSize: 24 });
  } else {
    throw new Error('Expected compile <IR> or compose <output-root> <IR...>');
  }
  emit(JSON.stringify(result) + '\n');
} catch (error) {
  console.error(error.stack || error.message);
  process.exitCode = 1;
}
