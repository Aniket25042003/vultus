import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: 'var(--ink)',
        mist: 'var(--mist)',
        breeze: 'var(--breeze)',
        ember: 'var(--ember)',
        glow: 'var(--glow)',
      },
      fontFamily: {
        sans: ['"Space Grotesk"', 'sans-serif'],
        display: ['"Fraunces"', 'serif'],
      },
      boxShadow: {
        glow: '0 10px 30px rgba(255, 122, 61, 0.25)',
      },
    },
  },
  plugins: [],
} satisfies Config;
