// Validate model drafts with the annotation kit's actual import rules.
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
const projectRoot = path.resolve(root, '..');
const kitRoot = path.join(projectRoot, 'AIE3905_Annotation_Kit');
const { validateAnnotationImport } = require(path.join(kitRoot, 'source', 'portable_annotation.js'));
const source = process.argv[2] || path.join(root, 'results', 'own_segmentation', 'preannotations.json');
const data = JSON.parse(fs.readFileSync(source, 'utf8'));
const rows = fs.readFileSync(path.join(kitRoot, 'demo', 'pairs.csv'), 'utf8').replace(/^\uFEFF/, '').trim().split(/\r?\n/);
const header = rows.shift().split(',');
const frames = rows.map(line => {
  const values = line.split(',');
  const row = Object.fromEntries(header.map((key, index) => [key, values[index]]));
  return { id: row.frame_id, width: Number(row.width), height: Number(row.height), group: row.time_group };
});
const validated = validateAnnotationImport(data, frames);
console.log(`Annotation kit accepts ${validated.objects.length} draft boxes from ${source}`);
