'use client';

import { useState, useCallback } from 'react';
import { Sidebar } from '@/components/layout/sidebar';
import { SettingsDialog } from '@/components/layout/settings-dialog';
import { ExportDialog } from '@/components/chat/export-dialog';
import { ChatMessages } from '@/components/chat/chat-messages';
import { ChatInput } from '@/components/chat/chat-input';
import { api, type Message } from '@/lib/api';

interface Conversation {
  id: string;
  title: string;
  timestamp: Date;
  messages: Message[];
}

export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string>();
  const [isLoading, setIsLoading] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isExportOpen, setIsExportOpen] = useState(false);
  const [settings, setSettings] = useState({
    maxTokens: 2048,
    temperature: 0.7,
    topP: 0.9,
    repetitionPenalty: 1.15,
  });

  const currentConversation = conversations.find((c) => c.id === currentConversationId);

  const createNewConversation = useCallback(() => {
    const newConv: Conversation = {
      id: Date.now().toString(),
      title: 'New Conversation',
      timestamp: new Date(),
      messages: [],
    };
    setConversations((prev) => [newConv, ...prev]);
    setCurrentConversationId(newConv.id);
  }, []);

  const handleSendMessage = useCallback(
    async (content: string) => {
      if (!currentConversationId) {
        createNewConversation();
        return;
      }

      const userMessage: Message = { role: 'user', content };
      
      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === currentConversationId
            ? {
                ...conv,
                messages: [...conv.messages, userMessage],
                title: conv.messages.length === 0 ? content.slice(0, 50) : conv.title,
              }
            : conv
        )
      );

      setIsLoading(true);

      try {
        const currentMessages = currentConversation?.messages || [];
        const response = await api.chat({
          messages: [...currentMessages, userMessage],
          max_tokens: settings.maxTokens,
          temperature: settings.temperature,
          top_p: settings.topP,
          repetition_penalty: settings.repetitionPenalty,
        });

        const assistantMessage: Message = {
          role: 'assistant',
          content: response.response,
        };

        setConversations((prev) =>
          prev.map((conv) =>
            conv.id === currentConversationId
              ? { ...conv, messages: [...conv.messages, assistantMessage] }
              : conv
          )
        );
      } catch (error) {
        console.error('Error sending message:', error);
        // Add error message
        const errorMessage: Message = {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please make sure the API server is running.',
        };
        setConversations((prev) =>
          prev.map((conv) =>
            conv.id === currentConversationId
              ? { ...conv, messages: [...conv.messages, errorMessage] }
              : conv
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [currentConversationId, currentConversation, settings, createNewConversation]
  );

  const handleDeleteConversation = useCallback((id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (currentConversationId === id) {
      setCurrentConversationId(undefined);
    }
  }, [currentConversationId]);

  // Create initial conversation if none exists
  if (conversations.length === 0 && !currentConversationId) {
    createNewConversation();
  }

  return (
    <div className="flex h-screen">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={setCurrentConversationId}
        onNewConversation={createNewConversation}
        onDeleteConversation={handleDeleteConversation}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onExport={() => setIsExportOpen(true)}
      />
      
      <div className="flex flex-1 flex-col">
        <ChatMessages
          messages={currentConversation?.messages || []}
          isLoading={isLoading}
        />
        <ChatInput onSend={handleSendMessage} disabled={isLoading} />
      </div>

      <SettingsDialog
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSettingsChange={setSettings}
      />

      <ExportDialog
        isOpen={isExportOpen}
        onClose={() => setIsExportOpen(false)}
        messages={currentConversation?.messages || []}
        conversationTitle={currentConversation?.title || 'Conversation'}
      />
    </div>
  );
}
