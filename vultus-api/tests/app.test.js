import { describe, it, expect, vi, beforeEach } from 'vitest';
import { EventEmitter } from 'node:events';
import bcrypt from 'bcryptjs';

process.env.JWT_SECRET = 'test-secret';
process.env.DATABASE_URL = 'postgres://test';
process.env.JWT_EXPIRES_IN = '1h';
process.env.CORS_ORIGIN = 'http://localhost:5173';
process.env.FACE_INFERENCE_URL = '';

const queryMock = vi.fn();

vi.mock('../src/db.js', () => ({
  query: (...args) => queryMock(...args),
  pool: { end: vi.fn() },
}));

const { createApp } = await import('../src/app.js');

const app = createApp({ testing: true });

const adminHash = await bcrypt.hash('Admin@123', 10);

const tokenPayload = {
  id: 'user-1',
  role: 'admin',
  email: 'admin@vultus.ai',
  name: 'Admin',
};

const makeAuthHeader = async () => {
  const jwt = await import('jsonwebtoken');
  return `Bearer ${jwt.default.sign(tokenPayload, process.env.JWT_SECRET)}`;
};

const makeResponse = () => {
  const res = new EventEmitter();
  res.statusCode = 200;
  res.headers = {};
  res.body = '';
  res.setHeader = (key, value) => {
    res.headers[key.toLowerCase()] = value;
  };
  res.getHeader = (key) => res.headers[key.toLowerCase()];
  res.writeHead = (statusCode, headers) => {
    res.statusCode = statusCode;
    if (headers) {
      Object.entries(headers).forEach(([k, v]) => res.setHeader(k, v));
    }
  };
  res.write = (chunk) => {
    res.body += chunk?.toString?.() ?? '';
  };
  res.end = (chunk) => {
    if (chunk) res.write(chunk);
    res.emit('finish');
  };
  return res;
};

const makeRequest = ({ method, url, body, headers = {} }) =>
  new Promise((resolve) => {
    const normalizedHeaders = Object.fromEntries(
      Object.entries(headers).map(([key, value]) => [key.toLowerCase(), value])
    );
    const req = {
      method,
      url,
      headers: normalizedHeaders,
      body,
      socket: { remoteAddress: '127.0.0.1' },
      get: (name) => normalizedHeaders[name.toLowerCase()],
      header: (name) => normalizedHeaders[name.toLowerCase()],
    };
    const res = makeResponse();
    res.on('finish', () => resolve(res));
    app.handle(req, res);
  });

beforeEach(() => {
  queryMock.mockReset();
});

describe('health', () => {
  it('returns ok status', async () => {
    const res = await makeRequest({ method: 'GET', url: '/api/health' });
    expect(res.statusCode).toBe(200);
    expect(JSON.parse(res.body).status).toBe('ok');
  });
});

describe('auth', () => {
  it('logs in with valid credentials', async () => {
    queryMock.mockResolvedValueOnce({
      rows: [
        {
          id: 'user-1',
          email: 'admin@vultus.ai',
          password_hash: adminHash,
          full_name: 'Vultus Admin',
          role: 'admin',
        },
      ],
    });

    const res = await makeRequest({
      method: 'POST',
      url: '/api/auth/login',
      body: { email: 'admin@vultus.ai', password: 'Admin@123' },
      headers: { 'Content-Type': 'application/json' },
    });

    const data = JSON.parse(res.body);
    expect(res.statusCode).toBe(200);
    expect(data.token).toBeTruthy();
    expect(data.user.role).toBe('admin');
  });
});

describe('detections', () => {
  it('returns person detection result', async () => {
    const auth = await makeAuthHeader();
    const res = await makeRequest({
      method: 'POST',
      url: '/api/detections/person',
      body: { frameId: 'frame-1' },
      headers: { Authorization: auth, 'Content-Type': 'application/json' },
    });

    const data = JSON.parse(res.body);
    expect(res.statusCode).toBe(200);
    expect(data.model).toContain('person-detector');
  });

  it('returns face detection result and logs event', async () => {
    const auth = await makeAuthHeader();
    queryMock.mockResolvedValueOnce({ rows: [] });
    const res = await makeRequest({
      method: 'POST',
      url: '/api/detections/face',
      body: { frameId: 'frame-2' },
      headers: { Authorization: auth, 'Content-Type': 'application/json' },
    });

    const data = JSON.parse(res.body);
    expect(res.statusCode).toBe(200);
    expect(data.decision).toBeTruthy();
    expect(queryMock).toHaveBeenCalled();
  });

  it('returns emotion detection result and logs event', async () => {
    const auth = await makeAuthHeader();
    queryMock.mockResolvedValueOnce({ rows: [] });
    const res = await makeRequest({
      method: 'POST',
      url: '/api/detections/emotion',
      body: { frameId: 'frame-3' },
      headers: { Authorization: auth, 'Content-Type': 'application/json' },
    });

    const data = JSON.parse(res.body);
    expect(res.statusCode).toBe(200);
    expect(data.emotion).toBeTruthy();
    expect(queryMock).toHaveBeenCalled();
  });
});

describe('dashboard', () => {
  it('returns teacher dashboard', async () => {
    const auth = await makeAuthHeader();
    queryMock
      .mockResolvedValueOnce({ rows: [{ emotion: 'focused', count: 2 }] })
      .mockResolvedValueOnce({ rows: [{ decision: 'granted', count: 3 }] })
      .mockResolvedValueOnce({ rows: [] });

    const res = await makeRequest({
      method: 'GET',
      url: '/api/dashboard/teacher',
      headers: { Authorization: auth },
    });

    const data = JSON.parse(res.body);
    expect(res.statusCode).toBe(200);
    expect(data.emotionSummary.length).toBeGreaterThanOrEqual(1);
  });

  it('returns admin dashboard', async () => {
    const auth = await makeAuthHeader();
    queryMock
      .mockResolvedValueOnce({ rows: [{ total: 10 }] })
      .mockResolvedValueOnce({ rows: [{ total: 5 }] })
      .mockResolvedValueOnce({ rows: [{ decision: 'granted', count: 8 }] })
      .mockResolvedValueOnce({ rows: [{ day: '2025-01-01', count: 4 }] });

    const res = await makeRequest({
      method: 'GET',
      url: '/api/dashboard/admin',
      headers: { Authorization: auth },
    });

    const data = JSON.parse(res.body);
    expect(res.statusCode).toBe(200);
    expect(data.totals.accessEvents).toBe(10);
  });
});
