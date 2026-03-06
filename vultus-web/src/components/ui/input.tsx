import * as React from 'react';
import { cn } from '../../lib/utils';

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        'w-full rounded-2xl border border-black/10 bg-white/70 px-4 py-3 text-sm focus:border-black/40 focus:outline-none focus:ring-2 focus:ring-ember/40',
        className
      )}
      {...props}
    />
  )
);

Input.displayName = 'Input';
