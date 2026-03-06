# Vultus API

## Setup

1. Create a PostgreSQL database and run the schema + seed.
2. Copy `.env.example` to `.env` and adjust values.
3. Install dependencies and start the server.

```bash
npm install
npm run dev
```

## ML Inference

The API can call the Python inference service at `FACE_INFERENCE_URL` for face detection.
If the service is unavailable, the API falls back to a deterministic stub.

## Database

```bash
./db/migrate.sh
psql "$DATABASE_URL" -f db/seed.sql
```

## Notes
- Seeded admin login: `admin@vultus.ai` / `Admin@123`
- `/api/auth/register` requires admin token.
- `/api/auth/signup` is optional via `ALLOW_PUBLIC_SIGNUP=true`.
