CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  full_name TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('admin', 'teacher')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS students (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  external_label TEXT UNIQUE NOT NULL,
  full_name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS face_embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id UUID NOT NULL REFERENCES students(id) ON DELETE CASCADE,
  embedding BYTEA NOT NULL,
  embedding_dim INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS access_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id UUID REFERENCES students(id) ON DELETE SET NULL,
  person_label TEXT,
  system_decision TEXT CHECK (system_decision IN ('granted', 'denied')),
  decision TEXT NOT NULL CHECK (decision IN ('granted', 'denied')),
  confidence NUMERIC(5,4) NOT NULL,
  person_confidence NUMERIC(5,4),
  match_confidence NUMERIC(5,4),
  face_snapshot TEXT,
  embedding BYTEA,
  embedding_dim INTEGER,
  source TEXT DEFAULT 'camera-1',
  overridden BOOLEAN NOT NULL DEFAULT FALSE,
  override_by UUID REFERENCES users(id) ON DELETE SET NULL,
  override_at TIMESTAMPTZ,
  override_reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS emotion_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id UUID REFERENCES students(id) ON DELETE SET NULL,
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
INSERT INTO users (email, password_hash, full_name, role)
VALUES (
  'admin@vultus.ai',
  '$2a$10$nMwzXU2KRwN0NXFyv0uqj.ecVp2nnpGu76cjO2cnvzeGU0D9YvNmu',
  'Vultus Admin',
  'admin'
) ON CONFLICT DO NOTHING;

-- Password for the seeded admin is: Admin@123
