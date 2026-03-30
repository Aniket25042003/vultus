import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { Card, CardDescription, CardTitle } from '../components/ui/card';

const features = [
  {
    title: 'Adaptive Access Control',
    description:
      'Layered detections that only unlock when a person is present and verified in your campus roster.',
  },
  {
    title: 'Emotion Signals for Teachers',
    description:
      'Live classroom pulse to shape lesson difficulty, energizers, or wrap-ups based on the mood.',
  },
  {
    title: 'Operational Telemetry',
    description: 'Admin-grade analytics on performance, confidence trends, and system health.',
  },
];

const LandingPage = () => (
  <div className="section-grid">
    <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-8">
      <div className="text-xl font-semibold tracking-tight">VULTUS</div>
      <Link to="/login">
        <Button variant="outline">Log in</Button>
      </Link>
    </header>

    <main className="mx-auto max-w-6xl px-6 pb-24">
      <section className="grid gap-12 lg:grid-cols-[1.1fr_0.9fr]">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="space-y-6"
        >
          <p className="text-sm uppercase tracking-[0.3em] text-black/40">Next-gen access</p>
          <h1 className="text-shadow font-display text-5xl leading-tight md:text-6xl">
            Let campus spaces recognize the right people.
          </h1>
          <p className="max-w-xl text-lg text-black/70">
            VULTUS replaces plastic badges with a layered AI guardrail: person detection, identity
            verification, and classroom emotion insights, all captured in one calm dashboard.
          </p>
          <div className="flex flex-wrap gap-4">
            <Link to="/login">
              <Button size="lg">Launch Console</Button>
            </Link>
            <Button size="lg" variant="ghost">
              Request Demo
            </Button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="glass rounded-[32px] p-6"
        >
          <div className="flex items-center justify-between text-sm text-black/60">
            <span>Live Access Feed</span>
            <span>Gate A · 08:32 AM</span>
          </div>
          <div className="mt-6 space-y-4">
            {[
              { name: 'Student 24', status: 'Granted', tone: 'bg-emerald-500/20 text-emerald-700' },
              { name: 'Unknown', status: 'Denied', tone: 'bg-rose-500/20 text-rose-700' },
              { name: 'Student 11', status: 'Granted', tone: 'bg-emerald-500/20 text-emerald-700' },
            ].map((row) => (
              <div
                key={row.name}
                className="flex items-center justify-between rounded-2xl bg-white/70 px-4 py-3"
              >
                <div className="font-medium">{row.name}</div>
                <span className={`rounded-full px-3 py-1 text-xs font-semibold ${row.tone}`}>
                  {row.status}
                </span>
              </div>
            ))}
          </div>
          <div className="mt-6 rounded-2xl bg-ember/10 px-4 py-3 text-sm text-black/70">
            Emotion lens: 62% focused · 18% tired · 12% curious
          </div>
        </motion.div>
      </section>

      <section className="mt-20 grid gap-6 md:grid-cols-3">
        {features.map((feature, index) => (
          <motion.div
            key={feature.title}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            viewport={{ once: true }}
          >
            <Card className="h-full">
              <CardTitle className="text-xl">{feature.title}</CardTitle>
              <CardDescription className="mt-3 text-base">
                {feature.description}
              </CardDescription>
            </Card>
          </motion.div>
        ))}
      </section>

      <section className="mt-20 grid gap-6 md:grid-cols-[0.7fr_1.3fr]">
        <Card className="space-y-4">
          <CardTitle className="text-2xl">Layered trust by design</CardTitle>
          <CardDescription className="text-base">
            Every access decision is captured with confidence scores and audit trails. Teachers get a
            pulse on classroom energy without individual student profiling.
          </CardDescription>
          <Button variant="outline">See Security Brief</Button>
        </Card>
        <Card className="grid gap-6 md:grid-cols-2">
          {[
            { label: 'Camera Latency', value: '82ms' },
            { label: 'Granted Today', value: '1,284' },
            { label: 'Denied Today', value: '14' },
            { label: 'AI Uptime', value: '99.98%' },
          ].map((stat) => (
            <div key={stat.label}>
              <p className="text-sm uppercase tracking-wide text-black/50">{stat.label}</p>
              <p className="mt-2 text-3xl font-semibold">{stat.value}</p>
            </div>
          ))}
        </Card>
      </section>
    </main>

    <footer className="mx-auto mt-auto w-full max-w-6xl border-t border-black/10 px-6 py-8">
      <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
        <p className="text-sm text-black/60">
          © {new Date().getFullYear()} VULTUS Inc. All rights reserved.
        </p>
        <div className="flex items-center space-x-6 text-sm">
          <Link to="/terms" className="text-black/60 transition-colors hover:text-black">
            Terms of Service
          </Link>
          <Link to="/privacy" className="text-black/60 transition-colors hover:text-black">
            Privacy Policy
          </Link>
          <a href="mailto:contact@vultus.com" className="text-black/60 transition-colors hover:text-black">
            Contact
          </a>
        </div>
      </div>
    </footer>
  </div>
);

export default LandingPage;
