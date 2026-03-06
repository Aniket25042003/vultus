import { Router } from 'express';
import { z } from 'zod';
import { query } from '../db.js';
import { config } from '../config.js';
import { inferEmbedding, storeEmbedding } from '../services/modelService.js';
import { requireAuth, requireRole } from '../middleware/auth.js';

const router = Router();

router.use(requireAuth);
router.use(requireRole('admin'));

const slugify = (value) =>
  value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)+/g, '');

const normalizeBase64 = (value) => {
  if (!value) return value;
  const parts = value.split(',');
  return parts.length > 1 ? parts[1] : value;
};

router.post('/enroll', async (req, res, next) => {
  try {
    if (!config.faceInferenceUrl) {
      return res.status(503).json({ error: 'Inference service unavailable.' });
    }

    const payload = z
      .object({
        fullName: z.string().min(2),
        label: z.string().min(1, 'Student ID is required'),
        images: z.array(z.string().min(10)).min(1).max(12),
      })
      .parse(req.body ?? {});

    const label = payload.label?.trim() || slugify(payload.fullName);

    const studentResult = await query(
      `INSERT INTO students (external_label, full_name)
       VALUES ($1, $2)
       ON CONFLICT (external_label)
       DO UPDATE SET full_name = EXCLUDED.full_name
       RETURNING id, external_label, full_name`,
      [label, payload.fullName.trim()]
    );
    const student = studentResult.rows[0];

    const failures = [];
    let stored = 0;
    for (const [index, image] of payload.images.entries()) {
      try {
        const normalized = normalizeBase64(image);
        const embedding = await inferEmbedding({ imageBase64: normalized });
        await storeEmbedding({ studentId: student.id, embedding });
        stored += 1;
      } catch (error) {
        failures.push({ index, error: error?.message ?? 'failed' });
      }
    }

    return res.json({
      studentId: student.id,
      fullName: student.full_name,
      externalLabel: student.external_label,
      stored,
      failed: failures.length,
      failures,
    });
  } catch (error) {
    return next(error);
  }
});

router.get('/', async (req, res, next) => {
  try {
    const result = await query(
      `SELECT
        students.id,
        students.full_name,
        students.external_label,
        COALESCE(embeddings.embedding_count, 0) AS embedding_count,
        embeddings.last_enrolled_at,
        COALESCE(access.access_total, 0) AS access_total,
        COALESCE(access.access_granted, 0) AS access_granted,
        COALESCE(access.access_denied, 0) AS access_denied,
        access.last_access_at,
        COALESCE(emotions.emotion_total, 0) AS emotion_total,
        emotions.last_emotion_at
      FROM students
      LEFT JOIN (
        SELECT student_id,
               COUNT(*)::int AS embedding_count,
               MAX(created_at) AS last_enrolled_at
        FROM face_embeddings
        GROUP BY student_id
      ) embeddings ON embeddings.student_id = students.id
      LEFT JOIN (
        SELECT student_id,
               COUNT(*)::int AS access_total,
               COUNT(*) FILTER (WHERE decision = 'granted')::int AS access_granted,
               COUNT(*) FILTER (WHERE decision = 'denied')::int AS access_denied,
               MAX(created_at) AS last_access_at
        FROM access_events
        GROUP BY student_id
      ) access ON access.student_id = students.id
      LEFT JOIN (
        SELECT student_id,
               COUNT(*)::int AS emotion_total,
               MAX(created_at) AS last_emotion_at
        FROM emotion_events
        GROUP BY student_id
      ) emotions ON emotions.student_id = students.id
      ORDER BY students.full_name ASC`
    );
    return res.json({ students: result.rows });
  } catch (error) {
    return next(error);
  }
});

router.delete('/:id', async (req, res, next) => {
  try {
    const payload = z.object({ id: z.string().uuid() }).parse(req.params);
    const result = await query(
      'DELETE FROM students WHERE id = $1 RETURNING id, full_name, external_label',
      [payload.id]
    );
    if (!result.rows.length) {
      return res.status(404).json({ error: 'Student not found.' });
    }
    return res.json({ student: result.rows[0] });
  } catch (error) {
    return next(error);
  }
});

export default router;
