# 🚀 GraphQL Setup Guide

This guide explains how to use the GraphQL API in Gaia AI.

## 📋 Overview

Gaia AI now supports **GraphQL** alongside the existing REST API. GraphQL provides:

- ✅ **Real-time streaming** of AI responses token-by-token
- ✅ **Flexible queries** - request exactly the data you need
- ✅ **Strong typing** with auto-generated TypeScript types
- ✅ **Single endpoint** for all operations
- ✅ **GraphiQL playground** for testing

## 🔧 Backend Setup

### 1. Install Dependencies

```bash
cd server
pip install -r requirements.txt
```

This installs:
- `strawberry-graphql[fastapi]` - GraphQL server
- `sse-starlette` - Server-sent events for subscriptions

### 2. Start the Server

```bash
cd server
python main.py
```

The GraphQL endpoint will be available at:
- **GraphQL API**: `http://localhost:8000/graphql`
- **GraphiQL Playground**: `http://localhost:8000/graphql` (in browser)

## 🎨 Frontend Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

This installs:
- `@apollo/client` - GraphQL client
- `graphql` - GraphQL core
- `graphql-ws` - WebSocket support for subscriptions

### 2. Configure Environment

Create `.env.local` in the frontend directory:

```bash
NEXT_PUBLIC_GRAPHQL_URL=http://localhost:8000/graphql
NEXT_PUBLIC_GRAPHQL_WS_URL=ws://localhost:8000/graphql
```

### 3. Start the Frontend

```bash
cd frontend
npm run dev
```

## 📖 GraphQL Schema

### Types

```graphql
type Message {
  role: String!
  content: String!
}

type ChatResponse {
  response: String!
  tokensGenerated: Int!
  generationTime: Float!
}

type ChatToken {
  token: String!
  isComplete: Boolean!
  totalTokens: Int
}

type HealthStatus {
  status: String!
  modelLoaded: Boolean!
  ragEnabled: Boolean!
  ragDocuments: Int!
  device: String
  cudaAvailable: Boolean!
}

type ModelInfo {
  name: String!
  version: String!
  contextWindow: Int!
  maxTokens: Int!
}
```

### Queries

#### Health Check

```graphql
query Health {
  health {
    status
    modelLoaded
    ragEnabled
    ragDocuments
    device
    cudaAvailable
  }
}
```

#### Model Information

```graphql
query ModelInfo {
  modelInfo {
    name
    version
    contextWindow
    maxTokens
  }
}
```

### Mutations

#### Send Chat Message

```graphql
mutation Chat($input: ChatInput!) {
  chat(input: $input) {
    response
    tokensGenerated
    generationTime
  }
}
```

**Variables:**
```json
{
  "input": {
    "messages": [
      {
        "role": "user",
        "content": "What are the benefits of chamomile tea?"
      }
    ],
    "maxTokens": 2048,
    "temperature": 0.7,
    "topP": 0.9,
    "repetitionPenalty": 1.15,
    "useRag": true
  }
}
```

### Subscriptions

#### Stream Chat Response

```graphql
subscription ChatStream($input: ChatInput!) {
  chatStream(input: $input) {
    token
    isComplete
    totalTokens
  }
}
```

**Variables:** Same as Chat mutation

## 🎮 Using GraphiQL Playground

1. Start the backend server
2. Open `http://localhost:8000/graphql` in your browser
3. Try the example queries above
4. Use the **Docs** tab to explore the schema

### Example Session

1. **Check Health:**
```graphql
query {
  health {
    status
    modelLoaded
  }
}
```

2. **Get Model Info:**
```graphql
query {
  modelInfo {
    name
    version
  }
}
```

3. **Send a Message:**
```graphql
mutation {
  chat(input: {
    messages: [
      { role: "user", content: "Hello Gaia!" }
    ]
  }) {
    response
    tokensGenerated
  }
}
```

## 💻 Frontend Usage

### Using Queries

```typescript
import { useQuery } from '@apollo/client';
import { HEALTH_QUERY } from '@/lib/graphql/queries';

function HealthCheck() {
  const { data, loading, error } = useQuery(HEALTH_QUERY);
  
  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;
  
  return <div>Status: {data.health.status}</div>;
}
```

### Using Mutations

```typescript
import { useMutation } from '@apollo/client';
import { CHAT_MUTATION } from '@/lib/graphql/mutations';

function ChatComponent() {
  const [sendMessage, { data, loading }] = useMutation(CHAT_MUTATION);
  
  const handleSend = async (message: string) => {
    await sendMessage({
      variables: {
        input: {
          messages: [{ role: 'user', content: message }],
          maxTokens: 2048,
          temperature: 0.7,
        },
      },
    });
  };
  
  return (
    <div>
      {data && <p>{data.chat.response}</p>}
    </div>
  );
}
```

### Using Subscriptions (Real-time Streaming)

```typescript
import { useSubscription } from '@apollo/client';
import { CHAT_STREAM_SUBSCRIPTION } from '@/lib/graphql/subscriptions';

function StreamingChat() {
  const { data } = useSubscription(CHAT_STREAM_SUBSCRIPTION, {
    variables: {
      input: {
        messages: [{ role: 'user', content: 'Tell me about herbs' }],
      },
    },
  });
  
  return (
    <div>
      {data?.chatStream.token}
      {data?.chatStream.isComplete && <p>✓ Complete</p>}
    </div>
  );
}
```

## 🔄 REST API vs GraphQL

Both APIs are available and work side-by-side:

| Feature | REST API | GraphQL API |
|---------|----------|-------------|
| Endpoint | `/chat`, `/health` | `/graphql` |
| Real-time streaming | ❌ No | ✅ Yes (subscriptions) |
| Flexible queries | ❌ Fixed response | ✅ Request what you need |
| Type safety | ⚠️ Manual types | ✅ Auto-generated |
| Playground | ❌ No | ✅ GraphiQL |

## 🐛 Troubleshooting

### GraphQL endpoint not found
- Ensure the backend server is running
- Check that `strawberry-graphql[fastapi]` is installed

### Subscriptions not working
- Verify WebSocket URL in `.env.local`
- Check that `sse-starlette` is installed
- Ensure firewall allows WebSocket connections

### TypeScript errors
- Run `npm install` to install all dependencies
- Restart your IDE/editor

## 📚 Additional Resources

- [Strawberry GraphQL Docs](https://strawberry.rocks/)
- [Apollo Client Docs](https://www.apollographql.com/docs/react/)
- [GraphQL Spec](https://graphql.org/)
