import type { Config } from 'tailwindcss';

export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        border: 'hsl(217 24% 18%)',
        input: 'hsl(217 24% 18%)',
        ring: 'hsl(142 71% 45%)',
        background: 'hsl(222 47% 7%)',
        foreground: 'hsl(210 40% 96%)',
        primary: {
          DEFAULT: 'hsl(142 72% 42%)',
          foreground: 'hsl(140 45% 96%)',
        },
        secondary: {
          DEFAULT: 'hsl(220 16% 11%)',
          foreground: 'hsl(210 40% 96%)',
        },
        muted: {
          DEFAULT: 'hsl(220 14% 10%)',
          foreground: 'hsl(215 12% 70%)',
        },
        accent: {
          DEFAULT: 'hsl(151 55% 34%)',
          foreground: 'hsl(140 45% 96%)',
        },
        destructive: {
          DEFAULT: 'hsl(0 84% 60%)',
          foreground: 'hsl(0 0% 98%)',
        },
        card: {
          DEFAULT: 'hsl(222 20% 8%)',
          foreground: 'hsl(210 40% 96%)',
        },
        popover: {
          DEFAULT: 'hsl(222 20% 8%)',
          foreground: 'hsl(210 40% 96%)',
        },
      },
      borderRadius: {
        lg: '0.9rem',
        md: '0.7rem',
        sm: '0.5rem',
      },
      boxShadow: {
        soft: '0 16px 40px rgba(2, 6, 23, 0.35)',
      },
      backgroundImage: {
        grid: 'radial-gradient(circle at 1px 1px, rgba(148,163,184,0.08) 1px, transparent 0)',
      },
    },
  },
  plugins: [],
} satisfies Config;
