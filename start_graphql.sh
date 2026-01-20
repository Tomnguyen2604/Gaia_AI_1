#!/bin/bash
# Start Gaia AI with GraphQL support

cd "$(dirname "$0")/server"

echo "🚀 Starting Gaia AI with GraphQL..."
echo "📍 GraphQL endpoint: http://localhost:8000/graphql"
echo "🎮 GraphiQL playground: http://localhost:8000/graphql (in browser)"
echo ""

python3 main.py
