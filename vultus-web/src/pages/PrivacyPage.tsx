import { Link } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { Card, CardTitle } from '../components/ui/card';

const PrivacyPage = () => (
  <div className="section-grid min-h-screen">
    <header className="mx-auto w-full max-w-6xl px-6 py-8 flex items-center justify-between">
      <Link to="/" className="text-xl font-semibold tracking-tight hover:opacity-80 transition-opacity">
        VULTUS
      </Link>
      <Link to="/">
        <Button variant="outline">Back to Home</Button>
      </Link>
    </header>

    <main className="mx-auto w-full max-w-4xl px-6 pb-24 pt-12">
      <div className="space-y-8">
        <div>
          <h1 className="font-display text-4xl leading-tight md:text-5xl">Privacy Policy</h1>
          <p className="mt-4 text-black/60">Last updated: {new Date().toLocaleDateString()}</p>
        </div>

        <Card className="space-y-6 !p-8">
          <section className="space-y-4">
            <CardTitle className="text-2xl">1. Information We Collect</CardTitle>
            <p className="text-black/70 leading-relaxed">
              We collect information to provide better services to all our users. The types of personal data we process depend on how you interact with our campus telemetry and access control systems.
            </p>
          </section>

          <section className="space-y-4">
            <CardTitle className="text-2xl">2. How We Use Information</CardTitle>
            <p className="text-black/70 leading-relaxed">
              We use the data we gather from biometric verifications, credential checks, and environmental sensors to maintain security definitions, improve AI models, and generate operational telemetry for designated campus administrators.
            </p>
          </section>

          <section className="space-y-4">
            <CardTitle className="text-2xl">3. Data Protection</CardTitle>
            <p className="text-black/70 leading-relaxed">
              We work hard to protect VULTUS and our users from unauthorized access to or unauthorized alteration, disclosure, or destruction of information we hold. We enact robust encryption paradigms for data in transit and at rest.
            </p>
          </section>
        </Card>
      </div>
    </main>

    <footer className="mx-auto mt-auto w-full max-w-6xl border-t border-black/10 px-6 py-8 text-center text-sm text-black/60">
      © {new Date().getFullYear()} VULTUS Inc. All rights reserved.
    </footer>
  </div>
);

export default PrivacyPage;
