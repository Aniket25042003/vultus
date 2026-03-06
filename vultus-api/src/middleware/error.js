import { ZodError } from 'zod';
import { logger } from '../logger.js';

export const notFound = (req, res, next) => {
  res.status(404).json({ error: 'Not found' });
};

export const errorHandler = (err, req, res, next) => {
  if (err instanceof ZodError) {
    return res.status(400).json({ error: 'Invalid request', details: err.errors });
  }
  logger.error({ err }, 'Unhandled error');
  return res.status(500).json({ error: 'Internal server error' });
};
