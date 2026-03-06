import * as React from 'react';
import { cn } from '../../lib/utils';

export type BadgeProps = React.HTMLAttributes<HTMLSpanElement> & {
  tone?: 'neutral' | 'success' | 'warning';
};

export const Badge = ({ className, tone = 'neutral', ...props }: BadgeProps) => (
  <span
    className={cn(
      'inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold',
      tone === 'neutral' && 'bg-black/10 text-black',
      tone === 'success' && 'bg-emerald-500/15 text-emerald-700',
      tone === 'warning' && 'bg-amber-500/15 text-amber-700',
      className
    )}
    {...props}
  />
);
