'use client';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { X } from 'lucide-react';

interface SettingsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  settings: {
    maxTokens: number;
    temperature: number;
    topP: number;
    repetitionPenalty: number;
  };
  onSettingsChange: (settings: any) => void;
}

export function SettingsDialog({
  isOpen,
  onClose,
  settings,
  onSettingsChange,
}: SettingsDialogProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm">
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Generation Settings</CardTitle>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Max Tokens */}
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Response Length: {settings.maxTokens} tokens
              </label>
              <input
                type="range"
                min="512"
                max="8192"
                step="256"
                value={settings.maxTokens}
                onChange={(e) =>
                  onSettingsChange({ ...settings, maxTokens: parseInt(e.target.value) })
                }
                className="w-full"
              />
            </div>

            {/* Temperature */}
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Temperature: {settings.temperature.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.1"
                max="1.5"
                step="0.1"
                value={settings.temperature}
                onChange={(e) =>
                  onSettingsChange({ ...settings, temperature: parseFloat(e.target.value) })
                }
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Lower = focused, Higher = creative
              </p>
            </div>

            {/* Top P */}
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Top P: {settings.topP.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={settings.topP}
                onChange={(e) =>
                  onSettingsChange({ ...settings, topP: parseFloat(e.target.value) })
                }
                className="w-full"
              />
            </div>

            {/* Repetition Penalty */}
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Repetition Penalty: {settings.repetitionPenalty.toFixed(2)}
              </label>
              <input
                type="range"
                min="1.0"
                max="2.0"
                step="0.05"
                value={settings.repetitionPenalty}
                onChange={(e) =>
                  onSettingsChange({ ...settings, repetitionPenalty: parseFloat(e.target.value) })
                }
                className="w-full"
              />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
