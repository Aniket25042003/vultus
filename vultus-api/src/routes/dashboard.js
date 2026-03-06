import { Router } from 'express';
import { query } from '../db.js';
import { requireAuth, requireRole } from '../middleware/auth.js';

const router = Router();

router.use(requireAuth);

router.get('/teacher', requireRole('teacher'), async (req, res, next) => {
  try {
    const ranges = {
      day: '1 day',
      week: '7 days',
      month: '30 days',
      year: '365 days',
    };

    const rangeKeys = Object.keys(ranges);

    const [emotionRows, accessRows, recentAccess, accessRangeRows, emotionRangeRows] =
      await Promise.all([
        query(
          "SELECT emotion, COUNT(*)::int AS count, AVG(confidence)::float AS avg_confidence FROM emotion_events WHERE created_at > NOW() - INTERVAL '1 day' GROUP BY emotion ORDER BY count DESC"
        ),
        query(
          "SELECT decision, COUNT(*)::int AS count FROM access_events WHERE created_at > NOW() - INTERVAL '1 day' GROUP BY decision"
        ),
        query(
          `SELECT access_events.id,
                access_events.person_label,
                access_events.decision,
                access_events.system_decision,
                access_events.confidence,
                access_events.match_confidence,
                access_events.person_confidence,
                access_events.face_snapshot,
                access_events.source,
                access_events.overridden,
                access_events.override_at,
                access_events.created_at,
                students.full_name,
                students.external_label
         FROM access_events
         LEFT JOIN students ON students.id = access_events.student_id
         ORDER BY access_events.created_at DESC
         LIMIT 20`
        ),
        Promise.all(
          rangeKeys.map((key) =>
            query(
              `SELECT decision, COUNT(*)::int AS count
             FROM access_events
             WHERE created_at > NOW() - INTERVAL '${ranges[key]}'
             GROUP BY decision`
            )
          )
        ),
        Promise.all(
          rangeKeys.map((key) =>
            query(
              `SELECT emotion, COUNT(*)::int AS count
             FROM emotion_events
             WHERE created_at > NOW() - INTERVAL '${ranges[key]}'
             GROUP BY emotion
             ORDER BY count DESC`
            )
          )
        ),
      ]);

    const accessRanges = rangeKeys.reduce((acc, key, idx) => {
      const rows = accessRangeRows[idx].rows;
      const granted = Number(rows.find((row) => row.decision === 'granted')?.count ?? 0);
      const denied = Number(rows.find((row) => row.decision === 'denied')?.count ?? 0);
      acc[key] = {
        granted,
        denied,
        total: granted + denied,
      };
      return acc;
    }, {});

    const emotionRanges = rangeKeys.reduce((acc, key, idx) => {
      const rows = emotionRangeRows[idx].rows;
      const total = rows.reduce((sum, row) => sum + Number(row.count), 0);
      acc[key] = {
        total,
        topEmotion: rows[0]?.emotion ?? null,
      };
      return acc;
    }, {});

    res.json({
      emotionSummary: emotionRows.rows,
      accessSummary: accessRows.rows,
      recentAccess: recentAccess.rows.map((row) => ({
        ...row,
        display_name: row.full_name ?? row.person_label,
        face_snapshot: row.face_snapshot ?? null,
      })),
      accessRanges,
      emotionRanges,
    });
  } catch (error) {
    next(error);
  }
});

router.get('/admin', requireRole('admin'), async (req, res, next) => {
  try {
    const [accessTotals, emotionTotals, decisionRates, dailyAccess, monthlyAccess] =
      await Promise.all([
        query('SELECT COUNT(*)::int AS total FROM access_events'),
        query('SELECT COUNT(*)::int AS total FROM emotion_events'),
        query(
          "SELECT decision, COUNT(*)::int AS count FROM access_events WHERE created_at > NOW() - INTERVAL '7 days' GROUP BY decision"
        ),
        query(
          "SELECT DATE_TRUNC('day', created_at) AS day, COUNT(*)::int AS count FROM access_events WHERE created_at > NOW() - INTERVAL '7 days' GROUP BY day ORDER BY day"
        ),
        query(
          "SELECT DATE_TRUNC('month', created_at) AS month, COUNT(*)::int AS count FROM access_events WHERE created_at > NOW() - INTERVAL '12 months' GROUP BY month ORDER BY month"
        ),
      ]);

    res.json({
      totals: {
        accessEvents: accessTotals.rows[0]?.total ?? 0,
        emotionEvents: emotionTotals.rows[0]?.total ?? 0,
      },
      decisionRates: decisionRates.rows,
      dailyAccess: dailyAccess.rows,
      monthlyAccess: monthlyAccess.rows,
    });
  } catch (error) {
    next(error);
  }
});

export default router;
