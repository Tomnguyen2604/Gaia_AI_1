"""
RAG (Retrieval-Augmented Generation) System for Gaia
Adds knowledge retrieval to enhance responses with scientific citations
"""

import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json

class GaiaRAG:
    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        """Initialize RAG system with embedding model and vector database"""
        print("🔄 Initializing Gaia RAG system...")
        
        # Initialize embedding model (lightweight, fast)
        print("📥 Loading embedding model...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize ChromaDB (local vector database)
        print("💾 Setting up vector database...")
        self.client = chromadb.Client(Settings(
            persist_directory=knowledge_base_path,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="gaia_knowledge",
            metadata={"description": "Natural medicine and health knowledge base"}
        )
        
        print(f"✅ RAG system ready! Knowledge base: {self.collection.count()} documents")
    
    def add_document(self, text: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None):
        """Add a document to the knowledge base"""
        if doc_id is None:
            doc_id = f"doc_{self.collection.count()}"
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
    
    def add_documents_batch(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add multiple documents at once"""
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        documents = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return documents
    
    def build_context(self, query: str, n_results: int = 3) -> str:
        """Build context string from retrieved documents"""
        results = self.search(query, n_results)
        
        if not results:
            return ""
        
        context_parts = ["Here is relevant information from my knowledge base:\n"]
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"\n[Source {i}]")
            context_parts.append(result['text'])
            
            # Add source URL if available
            if 'source_url' in result['metadata']:
                context_parts.append(f"Reference: {result['metadata']['source_url']}")
        
        context_parts.append("\nBased on this information and my knowledge:\n")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name,
            "embedding_model": "all-MiniLM-L6-v2"
        }

def load_knowledge_from_csv(rag: GaiaRAG, csv_path: str):
    """Load knowledge from CSV file (like natural_medicine_articles.csv)"""
    import csv
    
    print(f"📚 Loading knowledge from {csv_path}...")
    
    documents = []
    metadatas = []
    ids = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Combine instruction and output for better context
            doc_text = f"Q: {row['instruction']}\nA: {row['output']}"
            documents.append(doc_text)
            
            metadatas.append({
                'source': 'natural_medicine',
                'source_url': row.get('source_url', ''),
                'topic': row['instruction'][:50]
            })
            
            ids.append(f"med_{i}")
    
    if documents:
        rag.add_documents_batch(documents, metadatas, ids)
        print(f"✅ Loaded {len(documents)} documents from {csv_path}")
    
    return len(documents)

# Example usage
if __name__ == "__main__":
    # Initialize RAG
    rag = GaiaRAG()
    
    # Load natural medicine knowledge
    if os.path.exists("data/natural_medicine_articles.csv"):
        load_knowledge_from_csv(rag, "data/natural_medicine_articles.csv")
    
    # Test search
    query = "What are the benefits of turmeric?"
    results = rag.search(query, n_results=2)
    
    print(f"\n🔍 Search results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(result['text'][:200] + "...")
        print(f"Source: {result['metadata'].get('source_url', 'N/A')}")
    
    # Show stats
    print(f"\n📊 Stats: {rag.get_stats()}")
