import { Router } from 'express';
import { z } from 'zod';
import { query } from '../db.js';
import { requireAuth, requireRole } from '../middleware/auth.js';

const router = Router();

router.use(requireAuth);

router.post('/override', requireRole('admin'), async (req, res, next) => {
  try {
    const payload = z
      .object({
        accessEventId: z.string().min(1),
        decision: z.enum(['granted', 'denied']).default('granted'),
        reason: z.string().optional(),
      })
      .parse(req.body ?? {});

    const result = await query(
      `UPDATE access_events
         SET decision = $1,
             overridden = true,
             override_by = $2,
             override_at = NOW(),
             override_reason = $3
       WHERE id = $4
       RETURNING id, person_label, decision, overridden, override_at`,
      [payload.decision, req.user.id, payload.reason ?? null, payload.accessEventId]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Access event not found.' });
    }

    return res.json({
      ...result.rows[0],
      override_at: result.rows[0].override_at,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return next(error);
  }
});

export default router;
