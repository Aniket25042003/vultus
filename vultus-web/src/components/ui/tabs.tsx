import * as React from 'react';
import { cn } from '../../lib/utils';

const TabsContext = React.createContext<{
  value: string;
  onValueChange: (value: string) => void;
} | null>(null);

export const Tabs = ({
  value,
  onValueChange,
  children,
}: {
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
}) => (
  <TabsContext.Provider value={{ value, onValueChange }}>{children}</TabsContext.Provider>
);

export const TabsList = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn('inline-flex gap-2 rounded-full bg-black/5 p-1', className)} {...props} />
);

export const TabsTrigger = ({
  value,
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { value: string }) => {
  const ctx = React.useContext(TabsContext);
  if (!ctx) return null;
  const active = ctx.value === value;
  return (
    <button
      type="button"
      onClick={() => ctx.onValueChange(value)}
      className={cn(
        'rounded-full px-4 py-2 text-sm font-semibold transition',
        active ? 'bg-white shadow-sm' : 'text-black/60 hover:text-black',
        className
      )}
      {...props}
    />
  );
};

export const TabsContent = ({
  value,
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & { value: string }) => {
  const ctx = React.useContext(TabsContext);
  if (!ctx || ctx.value !== value) return null;
  return <div className={cn('mt-6', className)} {...props} />;
};
