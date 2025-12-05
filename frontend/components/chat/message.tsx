'use client';

import { cn } from '@/lib/utils';
import { Leaf, User } from 'lucide-react';

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
}

export function Message({ role, content }: MessageProps) {
  const isUser = role === 'user';

  return (
    <div className={cn('flex gap-4 p-4', isUser ? 'bg-muted/50' : 'bg-background')}>
      <div className={cn(
        'flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md',
        isUser ? 'bg-primary/10 text-primary' : 'bg-green-600 text-white'
      )}>
        {isUser ? <User className="h-4 w-4" /> : <Leaf className="h-4 w-4" />}
      </div>
      
      <div className="flex-1 space-y-2 overflow-hidden">
        <p className="text-sm font-medium">{isUser ? 'You' : 'Gaia'}</p>
        <div className="prose prose-sm dark:prose-invert max-w-none">
          {content}
        </div>
      </div>
    </div>
  );
}
