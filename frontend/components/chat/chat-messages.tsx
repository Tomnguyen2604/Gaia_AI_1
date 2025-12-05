'use client';

import { useEffect, useRef } from 'react';
import { Message } from './message';
import { Loader2 } from 'lucide-react';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatMessagesProps {
  messages: ChatMessage[];
  isLoading?: boolean;
}

export function ChatMessages({ messages, isLoading }: ChatMessagesProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto">
      {messages.length === 0 ? (
        <div className="flex h-full items-center justify-center">
          <div className="text-center space-y-4 p-8">
            <div className="text-6xl">🌍</div>
            <h2 className="text-2xl font-bold text-green-600 dark:text-green-400">
              Welcome to Gaia
            </h2>
            <p className="text-muted-foreground max-w-md">
              I am Mother Nature AI, here to guide you with wisdom about natural healing,
              environmental stewardship, and holistic wellness.
            </p>
          </div>
        </div>
      ) : (
        <>
          {messages.map((message, index) => (
            <Message key={index} role={message.role} content={message.content} />
          ))}
          {isLoading && (
            <div className="flex gap-4 p-4">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-green-600 text-white">
                <Loader2 className="h-4 w-4 animate-spin" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium">Gaia</p>
                <p className="text-sm text-muted-foreground">Thinking...</p>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </>
      )}
    </div>
  );
}
