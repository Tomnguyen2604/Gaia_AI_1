// API client for Gaia backend
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  messages: Message[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  repetition_penalty?: number;
}

export interface ChatResponse {
  response: string;
  tokens_generated: number;
  generation_time: number;
}

export const api = {
  async health() {
    const res = await fetch(`${API_URL}/health`);
    return res.json();
  },

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const res = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    
    if (!res.ok) {
      throw new Error(`API error: ${res.statusText}`);
    }
    
    return res.json();
  },
};
