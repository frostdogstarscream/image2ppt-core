// What this project can author, and what the installed engine can actually do.
//
// Two different questions live here, and conflating them is what makes a
// capability list lie:
//
//   compiler  — what THIS pipeline accepts and emits. This is the one that
//               decides whether an author may write a table node, because the
//               IR validator is what rejects it. It is not probed: it is a
//               property of our own source, so it is declared here and the
//               validator derives its schema from the same object.
//   engine    — what the resolved @oai/artifact-tool build supports. That is
//               probed at runtime, because runtime.mjs picks the highest
//               installed version dynamically and a version string is not a
//               stable contract.
//
// Keeping them apart is the point: the engine being able to build a table is
// not the same as this pipeline being able to deliver one.

// Flipping a flag here is what publishes a capability. It stays honest because
// IR_NODE_TYPES below is derived from it, so the schema the validator enforces
// and the list the skill reads move together. Adding a table node type without
// adding the slide.mjs branch is caught by the capability test.
export const COMPILER = Object.freeze({
  native_tables: false,
  native_charts: false,
});

// The accepted IR node kinds, derived rather than duplicated.
export const IR_NODE_TYPES = Object.freeze([
  'text', 'shape', 'image',
  ...(COMPILER.native_tables ? ['table'] : []),
  ...(COMPILER.native_charts ? ['chart'] : []),
]);

// Shape geometries the IR accepts. Narrower than the engine's preset catalog on
// purpose: every entry is one this pipeline has actually compiled.
export const IR_SHAPE_GEOMETRIES = Object.freeze(['rect', 'roundRect', 'ellipse', 'line']);

export const IR_EVIDENCE_KINDS = Object.freeze(['text', 'shape', 'image']);

const NATIVE = Object.freeze(['native_tables', 'native_charts']);

const BASE_SUPPORTED = Object.freeze([
  'single_page_image', 'native_text', 'simple_shapes', 'image_modules',
  'real_groups', 'final_pptx_reimport_preview', 'composed_deck',
]);

const BASE_MISSING = Object.freeze([
  'automatic_ocr', 'pdf_ingestion', 'automatic_inpainting', 'automatic_screenshot_paste',
]);

// The two lists a caller reads. Every native flag lands in exactly one of them,
// so they cannot both or neither mention it.
export function capabilities() {
  const supported = [...BASE_SUPPORTED];
  const not_implemented = [...BASE_MISSING];
  for (const flag of NATIVE) {
    (COMPILER[flag] ? supported : not_implemented).push(flag);
  }
  return { supported, not_implemented };
}

export function compilerFlags() {
  return { ...COMPILER };
}

// Ask the installed engine directly rather than trusting a version number.
// A failure here is recorded, not thrown: doctor must still report the rest.
export function probeEngine(rt) {
  const report = { available: false, tables: null, charts: null, detail: null };
  if (!rt || !rt.api || !rt.api.Presentation) {
    report.detail = 'engine not resolved';
    return report;
  }
  let slide;
  try {
    const presentation = rt.api.Presentation.create({ slideSize: { width: 1280, height: 720 } });
    slide = presentation.slides.add();
    report.available = true;
  } catch (error) {
    report.detail = String((error && error.message) || error);
    return report;
  }
  report.tables = attempt(() => slide.tables.add({ rows: 1, columns: 1, left: 0, top: 0, width: 120, height: 40, values: [['x']] }));
  report.charts = attempt(() => slide.charts.add('bar', {
    position: { left: 0, top: 0, width: 240, height: 180 },
    categories: ['a'], series: [{ name: 'probe', values: [1] }],
  }));
  return report;
}

function attempt(operation) {
  try {
    operation();
    return true;
  } catch (error) {
    return { ok: false, detail: String((error && error.message) || error) };
  }
}

// The doctor payload shared by the Node entry point and the M1 bridge.
export function doctorCapabilities(rt) {
  return {
    ...capabilities(),
    capability_source: {
      note: 'compiler is what this pipeline accepts; engine_probe is what the installed engine can build',
      compiler: compilerFlags(),
      engine_probe: probeEngine(rt),
    },
  };
}
