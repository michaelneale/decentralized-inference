import { cn } from '../lib/utils';

type MeshLlmWordmarkProps = {
  className?: string;
};

export function MeshLlmWordmark({ className }: MeshLlmWordmarkProps) {
  return (
    <span className={cn('whitespace-nowrap', className)}>
      <span className="text-primary">mesh</span>
      -llm
    </span>
  );
}
