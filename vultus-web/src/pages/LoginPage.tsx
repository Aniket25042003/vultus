import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import api from '../lib/api';
import { setSession } from '../lib/auth';
import { Button } from '../components/ui/button';
import { Card, CardDescription, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';

const LoginPage = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('admin@vultus.ai');
  const [password, setPassword] = useState('Admin@123');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError('');
    try {
      const response = await api.post('/api/auth/login', { email, password });
      setSession(response.data.token, response.data.user);
      navigate('/dashboard');
    } catch (err: any) {
      setError(err?.response?.data?.error ?? 'Unable to login.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="section-grid min-h-screen">
      <header className="mx-auto flex w-full max-w-5xl items-center justify-between px-6 py-8">
        <Link to="/" className="text-xl font-semibold">
          VULTUS
        </Link>
        <span className="text-sm text-black/50">Secure access console</span>
      </header>

      <main className="mx-auto flex w-full max-w-5xl flex-1 items-center px-6 pb-16">
        <div className="grid w-full gap-10 lg:grid-cols-[1.1fr_0.9fr]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="space-y-4"
          >
            <h1 className="font-display text-4xl">Welcome back.</h1>
            <p className="text-lg text-black/70">
              Authenticate to view access events, classroom sentiment, and operational telemetry.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <Card className="space-y-6">
              <div>
                <CardTitle>Sign in</CardTitle>
                <CardDescription>Use your VULTUS admin or teacher account.</CardDescription>
              </div>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Email</label>
                  <Input
                    type="email"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Password</label>
                  <Input
                    type="password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    required
                  />
                </div>
                {error && <p className="text-sm text-rose-600">{error}</p>}
                <Button type="submit" size="lg" disabled={loading}>
                  {loading ? 'Authenticating…' : 'Continue'}
                </Button>
              </form>
            </Card>
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default LoginPage;
