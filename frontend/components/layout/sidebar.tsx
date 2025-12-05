'use client';

import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { History, Settings, Trash2, Plus, Download } from 'lucide-react';
import { useState } from 'react';

interface Conversation {
  id: string;
  title: string;
  timestamp: Date;
}

interface SidebarProps {
  conversations: Conversation[];
  currentConversationId?: string;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
  onOpenSettings: () => void;
  onExport: () => void;
}

export function Sidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  onOpenSettings,
  onExport,
}: SidebarProps) {
  return (
    <div className="flex h-full w-64 flex-col border-r bg-muted/10">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center gap-2 mb-4">
          <div className="text-2xl">🌍</div>
          <h1 className="text-xl font-bold text-green-600 dark:text-green-400">Gaia</h1>
        </div>
        <Button onClick={onNewConversation} className="w-full" size="sm">
          <Plus className="h-4 w-4 mr-2" />
          New Chat
        </Button>
      </div>

      {/* Conversations */}
      <div className="flex-1 overflow-y-auto p-2">
        <div className="space-y-1">
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className="group relative"
            >
              <Button
                variant={currentConversationId === conv.id ? 'secondary' : 'ghost'}
                className="w-full justify-start text-left"
                onClick={() => onSelectConversation(conv.id)}
              >
                <History className="h-4 w-4 mr-2 shrink-0" />
                <span className="truncate flex-1">{conv.title}</span>
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="absolute right-1 top-1 h-6 w-6 opacity-0 group-hover:opacity-100"
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteConversation(conv.id);
                }}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t space-y-1">
        <Button
          variant="ghost"
          className="w-full justify-start"
          onClick={onExport}
          disabled={!currentConversationId}
        >
          <Download className="h-4 w-4 mr-2" />
          Export Chat
        </Button>
        <Button
          variant="ghost"
          className="w-full justify-start"
          onClick={onOpenSettings}
        >
          <Settings className="h-4 w-4 mr-2" />
          Settings
        </Button>
        <div className="mt-2 text-xs text-muted-foreground text-center">
          Gaia v1.0 • Gemma-2-2B
        </div>
      </div>
    </div>
  );
}
