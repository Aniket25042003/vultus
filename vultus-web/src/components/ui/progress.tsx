import * as React from 'react';
import { cn } from '../../lib/utils';

export const Progress = ({ value, className }: { value: number; className?: string }) => (
  <div className={cn('h-2 w-full rounded-full bg-black/10', className)}>
    <div
      className="h-full rounded-full bg-ember transition-all"
      style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
    />
  </div>
);
