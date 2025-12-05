'use client';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Download, FileText, FileJson, X } from 'lucide-react';
import { Message } from '@/lib/api';
import { formatDate } from '@/lib/utils';

interface ExportDialogProps {
  isOpen: boolean;
  onClose: () => void;
  messages: Message[];
  conversationTitle: string;
}

export function ExportDialog({
  isOpen,
  onClose,
  messages,
  conversationTitle,
}: ExportDialogProps) {
  if (!isOpen) return null;

  const exportAsJSON = () => {
    const data = {
      title: conversationTitle,
      exported: new Date().toISOString(),
      messages: messages,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    downloadFile(blob, `${conversationTitle}-${Date.now()}.json`);
  };

  const exportAsMarkdown = () => {
    let markdown = `# ${conversationTitle}\n\n`;
    markdown += `*Exported: ${formatDate(new Date())}*\n\n---\n\n`;
    
    messages.forEach((msg, idx) => {
      const role = msg.role === 'user' ? '👤 You' : '🌍 Gaia';
      markdown += `### ${role}\n\n${msg.content}\n\n`;
      if (idx < messages.length - 1) markdown += '---\n\n';
    });

    const blob = new Blob([markdown], { type: 'text/markdown' });
    downloadFile(blob, `${conversationTitle}-${Date.now()}.md`);
  };

  const exportAsText = () => {
    let text = `${conversationTitle}\n`;
    text += `Exported: ${formatDate(new Date())}\n`;
    text += '='.repeat(50) + '\n\n';
    
    messages.forEach((msg) => {
      const role = msg.role === 'user' ? 'You' : 'Gaia';
      text += `${role}:\n${msg.content}\n\n`;
      text += '-'.repeat(50) + '\n\n';
    });

    const blob = new Blob([text], { type: 'text/plain' });
    downloadFile(blob, `${conversationTitle}-${Date.now()}.txt`);
  };

  const downloadFile = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm">
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Export Conversation</CardTitle>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Choose a format to export your conversation with Gaia.
            </p>
            
            <Button
              onClick={exportAsMarkdown}
              className="w-full justify-start"
              variant="outline"
            >
              <FileText className="h-4 w-4 mr-2" />
              Export as Markdown (.md)
            </Button>

            <Button
              onClick={exportAsJSON}
              className="w-full justify-start"
              variant="outline"
            >
              <FileJson className="h-4 w-4 mr-2" />
              Export as JSON (.json)
            </Button>

            <Button
              onClick={exportAsText}
              className="w-full justify-start"
              variant="outline"
            >
              <Download className="h-4 w-4 mr-2" />
              Export as Text (.txt)
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
