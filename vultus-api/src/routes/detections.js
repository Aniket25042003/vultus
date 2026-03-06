import { Buffer } from 'node:buffer';
import { Router } from 'express';
import { z } from 'zod';
import { query } from '../db.js';
import { requireAuth } from '../middleware/auth.js';
import {
  runEmotionDetection,
  runFaceDetection,
  runPersonDetection,
} from '../services/modelService.js';

const router = Router();

const detectionSchema = z.object({
  frameId: z.string().optional(),
  imageBase64: z.string().optional(),
  source: z.string().optional(),
});

router.use(requireAuth);

router.post('/person', async (req, res, next) => {
  try {
    const payload = detectionSchema.parse(req.body ?? {});
    const result = await runPersonDetection(payload);
    return res.json({
      ...result,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return next(error);
  }
});

router.post('/face', async (req, res, next) => {
  try {
    const payload = detectionSchema.parse(req.body ?? {});
    const result = await runFaceDetection(payload);
    const systemDecision = result.decision;

    // Convert embedding array to binary buffer for storage
    let embeddingBuffer = null;
    let embeddingDim = null;
    if (Array.isArray(result.embedding) && result.embedding.length > 0) {
      const floatArray = new Float32Array(result.embedding);
      embeddingBuffer = Buffer.from(floatArray.buffer, floatArray.byteOffset, floatArray.byteLength);
      embeddingDim = result.embedding.length;
    }

    const accessInsert = await query(
      `INSERT INTO access_events
        (student_id, person_label, system_decision, decision, confidence, person_confidence, match_confidence, face_snapshot, embedding, embedding_dim, source)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
       RETURNING id`,
      [
        result.studentId ?? null,
        result.personLabel ?? null,
        systemDecision,
        result.decision,
        result.confidence,
        result.personConfidence ?? null,
        result.matchConfidence ?? null,
        result.faceImageBase64 ?? null,
        embeddingBuffer,
        embeddingDim,
        payload.source ?? 'camera-1',
      ]
    );

    let emotion = null;
    if (result.decision === 'granted') {
      const emotionResult = await runEmotionDetection({
        ...payload,
        faceBox: result.faceBox ?? null,
      });
      emotion = emotionResult;
      await query(
        'INSERT INTO emotion_events (student_id, person_label, emotion, confidence, source) VALUES ($1, $2, $3, $4, $5)',
        [
          result.studentId ?? null,
          result.personLabel ?? null,
          emotionResult.emotion,
          emotionResult.confidence,
          payload.source ?? 'camera-1',
        ]
      );
    }
    const { embedding, faceImageBase64, ...safeResult } = result;
    return res.json({
      ...safeResult,
      accessEventId: accessInsert.rows[0]?.id ?? null,
      faceBox: result.faceBox ?? null,
      emotion,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return next(error);
  }
});

router.post('/emotion', async (req, res, next) => {
  try {
    const payload = detectionSchema.parse(req.body ?? {});
    const result = await runEmotionDetection(payload);
    await query(
      'INSERT INTO emotion_events (person_label, emotion, confidence, source) VALUES ($1, $2, $3, $4)',
      [payload.frameId ?? null, result.emotion, result.confidence, payload.source ?? 'camera-1']
    );
    return res.json({
      ...result,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return next(error);
  }
});

export default router;
