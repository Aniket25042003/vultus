import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import pinoHttp from 'pino-http';
import { config } from './config.js';
import { logger } from './logger.js';
import healthRouter from './routes/health.js';
import authRouter from './routes/auth.js';
import detectionRouter from './routes/detections.js';
import accessRouter from './routes/access.js';
import studentRouter from './routes/students.js';
import dashboardRouter from './routes/dashboard.js';
import { errorHandler, notFound } from './middleware/error.js';

export const createApp = (options = {}) => {
  const { testing = false } = options;
  const app = express();

  if (!testing) {
    app.use(
      pinoHttp({
        logger,
        redact: ['req.headers.authorization'],
        autoLogging: {
          ignore: (req) => {
            if (req.url.startsWith('/api/dashboard')) return true;
            if (req.url.startsWith('/api/students') && req.method === 'GET') return true;
            return false;
          },
        },
      })
    );
  }

  if (!testing) {
    app.use(helmet());
    app.use(
      cors({
        origin: config.corsOrigin,
        credentials: true,
      })
    );
  }

  if (!testing) {
    app.use(express.json({ limit: '50mb' }));
  }

  if (!testing) {
    app.use(
      rateLimit({
        windowMs: 60 * 1000,
        max: 120,
        standardHeaders: true,
        legacyHeaders: false,
      })
    );
  }

  app.use('/api/health', healthRouter);
  app.use('/api/auth', authRouter);
  app.use('/api/detections', detectionRouter);
  app.use('/api/access', accessRouter);
  app.use('/api/students', studentRouter);
  app.use('/api/dashboard', dashboardRouter);

  app.use(notFound);
  app.use(errorHandler);

  return app;
};
