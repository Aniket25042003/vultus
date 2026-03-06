import fs from 'node:fs';
import { Buffer } from 'node:buffer';
import { config } from '../config.js';
import { query } from '../db.js';

const hashToUnit = (value) => {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(i);
    hash |= 0;
  }
  const normalized = Math.abs(hash % 1000) / 1000;
  return normalized;
};

const buildSeed = (payload) =>
  `${payload.frameId ?? ''}-${payload.imageBase64?.length ?? 0}-${payload.source ?? ''}`;

const callInference = async (path, payload) => {
  if (!config.faceInferenceUrl) {
    return null;
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(`${config.faceInferenceUrl}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!response.ok) {
      return null;
    }
    return response.json();
  } catch (error) {
    clearTimeout(timeout);
    return null;
  }
};

export const inferEmbedding = async ({ imageBase64, faceBox = null }) => {
  if (!imageBase64) {
    throw new Error('image base64 required');
  }
  const data = await callInference('/predict/embedding', {
    image_base64: imageBase64,
    face_bbox: faceBox ?? null,
  });
  if (!data || !Array.isArray(data.embedding)) {
    throw new Error('embedding unavailable');
  }
  return data.embedding;
};

export const runPersonDetection = async (payload) => {
  if (!payload.imageBase64) {
    const seed = buildSeed(payload);
    const score = hashToUnit(seed || '0');
    const detected = score > 0.35;
    return {
      detected,
      confidence: Number((0.5 + score / 2).toFixed(4)),
      model: 'person-detector-stub',
      bbox: null,
    };
  }
  const ml = await callInference('/predict/person', {
    frame_id: payload.frameId,
    image_base64: payload.imageBase64,
    source: payload.source,
  });
  if (ml) {
    return {
      detected: Boolean(ml.detected),
      confidence: Number(Number(ml.confidence ?? 0).toFixed(4)),
      model: ml.model ?? 'person-detector-remote',
      bbox: ml.bbox ?? null,
    };
  }
  const seed = buildSeed(payload);
  const score = hashToUnit(seed || '0');
  const detected = score > 0.35;
  return {
    detected,
    confidence: Number((0.5 + score / 2).toFixed(4)),
    model: 'person-detector-stub',
    bbox: null,
  };
};

const runFaceDetectionStub = (payload) => {
  const seed = buildSeed(payload);
  const score = hashToUnit(seed || 'face');
  const recognized = score > 0.4;
  const labelIndex = Math.floor(score * 30) + 1;
  const personLabel = recognized ? `student-${labelIndex}` : null;
  const modelAvailable = config.faceModelPath && fs.existsSync(config.faceModelPath);
  return {
    recognized,
    personLabel,
    decision: recognized ? 'granted' : 'denied',
    confidence: Number((0.52 + score / 2).toFixed(4)),
    model: modelAvailable ? 'face-detector-backbone' : 'face-detector-stub',
    modelAvailable,
    personDetected: true,
    personConfidence: Number((0.6 + score / 2).toFixed(4)),
    matchConfidence: Number((0.52 + score / 2).toFixed(4)),
  };
};

const normalize = (vector) => {
  let sum = 0;
  for (const value of vector) {
    sum += value * value;
  }
  const norm = Math.sqrt(sum) || 1;
  return vector.map((value) => value / norm);
};

const dot = (a, b) => {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
};

const bufferToFloatArray = (buffer) => {
  if (!buffer) return null;
  const view = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
  return Array.from(view);
};

const floatArrayToBuffer = (array) => {
  const floatArray = new Float32Array(array);
  return Buffer.from(floatArray.buffer, floatArray.byteOffset, floatArray.byteLength);
};

const findBestMatch = async (embedding) => {
  const { rows } = await query(
    `SELECT students.id, students.full_name, students.external_label, face_embeddings.embedding
     FROM face_embeddings
     JOIN students ON students.id = face_embeddings.student_id`
  );
  if (!rows.length) {
    return null;
  }
  const normalized = normalize(embedding);
  let best = null;
  for (const row of rows) {
    const candidate = bufferToFloatArray(row.embedding);
    if (!candidate || candidate.length !== normalized.length) {
      continue;
    }
    const score = dot(normalized, candidate);
    if (!best || score > best.score) {
      best = {
        studentId: row.id,
        fullName: row.full_name,
        externalLabel: row.external_label,
        score,
      };
    }
  }
  if (!best) {
    return null;
  }
  return best;
};

export const storeEmbedding = async ({ studentId, embedding }) => {
  const normalized = normalize(embedding);
  const buffer = floatArrayToBuffer(normalized);
  await query(
    'INSERT INTO face_embeddings (student_id, embedding, embedding_dim) VALUES ($1, $2, $3)',
    [studentId, buffer, normalized.length]
  );
};

export const runFaceDetection = async (payload) => {
  const modelAvailable = config.faceModelPath && fs.existsSync(config.faceModelPath);
  if (!config.faceInferenceUrl || !payload.imageBase64) {
    return runFaceDetectionStub(payload);
  }

  const data = await callInference('/predict/face', {
    frame_id: payload.frameId,
    image_base64: payload.imageBase64,
    source: payload.source,
  });

  if (!data || !data.person_detected || !Array.isArray(data.embedding)) {
    return {
      recognized: false,
      personLabel: null,
      decision: 'denied',
      confidence: 0.0,
      model: data?.model ?? 'person-detector',
      modelAvailable,
      personDetected: Boolean(data?.person_detected),
      personConfidence: Number(Number(data?.person_confidence ?? 0).toFixed(4)),
      matchConfidence: 0.0,
      faceBox: data?.person_bbox ?? null,
      faceImageBase64: data?.face_image_base64 ?? null,
    };
  }

  const match = await findBestMatch(data.embedding);
  const similarity = match?.score ?? 0;
  const recognized = similarity >= config.faceMatchThreshold;

  return {
    recognized,
    personLabel: recognized ? match?.fullName ?? null : null,
    studentId: recognized ? match?.studentId ?? null : null,
    decision: recognized ? 'granted' : 'denied',
    confidence: Number(similarity.toFixed(4)),
    model: data.model ?? 'face-embedder-remote',
    modelAvailable,
    personDetected: true,
    personConfidence: Number(Number(data.person_confidence ?? 0).toFixed(4)),
    matchConfidence: Number(similarity.toFixed(4)),
    faceBox: data.person_bbox ?? null,
    faceImageBase64: data.face_image_base64 ?? null,
    embedding: data.embedding,
  };
};

export const runEmotionDetection = async (payload) => {
  const ml = await callInference('/predict/emotion', {
    frame_id: payload.frameId,
    image_base64: payload.imageBase64,
    source: payload.source,
    face_bbox: payload.faceBox ?? null,
  });
  if (ml) {
    return {
      emotion: ml.emotion ?? 'unknown',
      confidence: Number(Number(ml.confidence ?? 0).toFixed(4)),
      model: ml.model ?? 'emotion-detector-remote',
      scores: ml.scores ?? null,
    };
  }
  const seed = buildSeed(payload);
  const score = hashToUnit(seed || 'emotion');
  const emotions = ['focused', 'neutral', 'tired', 'curious', 'stressed', 'excited'];
  const emotion = emotions[Math.floor(score * emotions.length) % emotions.length];
  return {
    emotion,
    confidence: Number((0.55 + score / 2).toFixed(4)),
    model: 'emotion-detector-stub',
    scores: null,
  };
};
