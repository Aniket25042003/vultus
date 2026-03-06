import dotenv from 'dotenv';

dotenv.config();

const required = (key, fallback) => {
  const value = process.env[key] ?? fallback;
  if (value === undefined || value === '') {
    throw new Error(`Missing required env: ${key}`);
  }
  return value;
};

export const config = {
  env: process.env.NODE_ENV ?? 'development',
  port: Number(process.env.PORT ?? 4000),
  databaseUrl: required('DATABASE_URL', process.env.DATABASE_URL),
  jwtSecret: required('JWT_SECRET', process.env.JWT_SECRET),
  jwtExpiresIn: process.env.JWT_EXPIRES_IN ?? '1d',
  corsOrigin: process.env.CORS_ORIGIN ?? 'http://localhost:5173',
  allowPublicSignup: (process.env.ALLOW_PUBLIC_SIGNUP ?? 'false') === 'true',
  faceModelPath: process.env.FACE_MODEL_PATH ?? '',
  faceInferenceUrl: process.env.FACE_INFERENCE_URL ?? 'http://localhost:8000',
  faceMatchThreshold: Number(process.env.FACE_MATCH_THRESHOLD ?? 0.4),
};
