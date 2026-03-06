import { Router } from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { z } from 'zod';
import { config } from '../config.js';
import { query } from '../db.js';
import { logger } from '../logger.js';
import { requireAuth, requireRole } from '../middleware/auth.js';

const router = Router();

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
});

const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  fullName: z.string().min(2),
  role: z.enum(['admin', 'teacher']).default('teacher'),
});

router.post('/login', async (req, res, next) => {
  try {
    const body = loginSchema.parse(req.body);
    logger.info(`[AUTH] Login attempt for: ${body.email}`);
    const result = await query('SELECT * FROM users WHERE email = $1', [body.email]);
    const user = result.rows[0];

    if (user) {
      logger.info(`[AUTH] User found: ${user.email} (Role: ${user.role})`);
    } else {
      logger.warn(`[AUTH] User not found: ${body.email}`);
    }

    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    const valid = await bcrypt.compare(body.password, user.password_hash);
    if (!valid) {
      logger.warn(`[AUTH] Invalid password for: ${body.email}`);
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    const token = jwt.sign(
      { id: user.id, role: user.role, email: user.email, name: user.full_name },
      config.jwtSecret,
      { expiresIn: config.jwtExpiresIn }
    );
    return res.json({
      token,
      user: {
        id: user.id,
        email: user.email,
        fullName: user.full_name,
        role: user.role,
      },
    });
  } catch (error) {
    return next(error);
  }
});

router.post('/register', requireAuth, requireRole('admin'), async (req, res, next) => {
  try {
    const body = registerSchema.parse(req.body);
    const hash = await bcrypt.hash(body.password, 10);
    const result = await query(
      'INSERT INTO users (email, password_hash, full_name, role) VALUES ($1, $2, $3, $4) RETURNING id, email, full_name, role',
      [body.email, hash, body.fullName, body.role]
    );
    return res.status(201).json({ user: result.rows[0] });
  } catch (error) {
    return next(error);
  }
});

router.post('/signup', async (req, res, next) => {
  try {
    if (!config.allowPublicSignup) {
      return res.status(403).json({ error: 'Public signup disabled' });
    }
    const body = registerSchema.parse(req.body);
    const hash = await bcrypt.hash(body.password, 10);
    const result = await query(
      'INSERT INTO users (email, password_hash, full_name, role) VALUES ($1, $2, $3, $4) RETURNING id, email, full_name, role',
      [body.email, hash, body.fullName, body.role]
    );
    return res.status(201).json({ user: result.rows[0] });
  } catch (error) {
    return next(error);
  }
});

router.get('/me', requireAuth, async (req, res) => {
  res.json({ user: req.user });
});

export default router;
