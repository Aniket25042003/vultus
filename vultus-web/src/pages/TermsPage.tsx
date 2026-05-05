import { Link } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { Card, CardTitle } from '../components/ui/card';

const TermsPage = () => (
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
          <h1 className="font-display text-4xl leading-tight md:text-5xl">Terms of Service</h1>
          <p className="mt-4 text-black/60">Last updated: {new Date().toLocaleDateString()}</p>
        </div>

        <Card className="space-y-6 !p-8">
          <section className="space-y-4">
            <CardTitle className="text-2xl">1. Acceptance of Terms</CardTitle>
            <p className="text-black/70 leading-relaxed">
              By accessing and using VULTUS services, you accept and agree to be bound by the terms and provision of this agreement. Any participation in this service will constitute acceptance of this agreement.
            </p>
          </section>

          <section className="space-y-4">
            <CardTitle className="text-2xl">2. Provision of Services</CardTitle>
            <p className="text-black/70 leading-relaxed">
              You agree and acknowledge that VULTUS is entitled to modify, improve or discontinue any of its services at its sole discretion and without notice to you even if it may result in you being prevented from accessing any information contained in it.
            </p>
          </section>

          <section className="space-y-4">
            <CardTitle className="text-2xl">3. Proprietary Rights</CardTitle>
            <p className="text-black/70 leading-relaxed">
              You acknowledge and agree that VULTUS may contain proprietary and confidential information including trademarks, service marks and patents protected by intellectual property laws and international intellectual property treaties.
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

export default TermsPage;
