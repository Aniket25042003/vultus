import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { query } from '../src/db.js';
import { config } from '../src/config.js';
import { storeEmbedding } from '../src/services/modelService.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const getArg = (flag, fallback) => {
  const idx = process.argv.indexOf(flag);
  if (idx !== -1 && process.argv[idx + 1]) {
    return process.argv[idx + 1];
  }
  return fallback;
};

const name = getArg('--name', getArg('-n', ''));
const label = getArg('--label', getArg('-l', ''));
const imagesDir = getArg('--images', getArg('--dir', ''));
const inferenceUrl = process.env.FACE_INFERENCE_URL ?? config.faceInferenceUrl;

if (!name || !label || !imagesDir) {
  console.error(
    'Usage: node scripts/enroll.js --name "Full Name" --label "unique-label" --images /path/to/folder'
  );
  process.exit(1);
}

const resolveDir = path.isAbsolute(imagesDir)
  ? imagesDir
  : path.resolve(__dirname, '..', imagesDir);

const imageFiles = fs
  .readdirSync(resolveDir)
  .filter((file) => /\.(png|jpg|jpeg)$/i.test(file))
  .map((file) => path.join(resolveDir, file));

if (!imageFiles.length) {
  console.error(`No images found in ${resolveDir}`);
  process.exit(1);
}

const upsertStudent = async () => {
  const result = await query(
    `INSERT INTO students (external_label, full_name)
     VALUES ($1, $2)
     ON CONFLICT (external_label)
     DO UPDATE SET full_name = EXCLUDED.full_name
     RETURNING id, full_name`,
    [label, name]
  );
  return result.rows[0];
};

const fetchEmbedding = async (imageBase64) => {
  const response = await fetch(`${inferenceUrl}/predict/embedding`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_base64: imageBase64 }),
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Inference failed: ${response.status} ${body}`);
  }
  const data = await response.json();
  if (!Array.isArray(data.embedding)) {
    throw new Error('Embedding not available from inference service.');
  }
  return data.embedding;
};

const enroll = async () => {
  const student = await upsertStudent();
  let stored = 0;
  for (const file of imageFiles) {
    const imageBase64 = fs.readFileSync(file).toString('base64');
    const embedding = await fetchEmbedding(imageBase64);
    await storeEmbedding({ studentId: student.id, embedding });
    stored += 1;
    console.log(`Stored embedding ${stored}/${imageFiles.length} from ${path.basename(file)}`);
  }
  console.log(`Enrollment complete for ${student.full_name} (${label}).`);
};

enroll().catch((error) => {
  console.error(error);
  process.exit(1);
});
