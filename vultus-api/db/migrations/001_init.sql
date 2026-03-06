CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  full_name TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('admin', 'teacher')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS access_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  person_label TEXT,
  decision TEXT NOT NULL CHECK (decision IN ('granted', 'denied')),
  confidence NUMERIC(5,4) NOT NULL,
  source TEXT DEFAULT 'camera-1',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS emotion_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  person_label TEXT,
  emotion TEXT NOT NULL,
  confidence NUMERIC(5,4) NOT NULL,
  source TEXT DEFAULT 'camera-1',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS system_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  metric TEXT NOT NULL,
  value NUMERIC(10,4) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
