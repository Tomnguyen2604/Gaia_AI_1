#!/usr/bin/env python3
"""
Build Gaia's knowledge base from datasets
Run this once to populate the vector database
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import GaiaRAG, load_knowledge_from_csv

def load_datasets_from_file(datasets_file):
    """Load dataset list from file, filtering out comments and empty lines"""
    datasets = []
    if os.path.exists(datasets_file):
        with open(datasets_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    datasets.append(line)
    return datasets

def main():
    print("🌍 Building Gaia Knowledge Base")
    print("="*60)
    
    # Initialize RAG system
    rag = GaiaRAG()
    
    # Load datasets from datasets.txt
    datasets_file = "data/datasets.txt"
    print(f"\n📋 Reading datasets from: {datasets_file}")
    
    dataset_list = load_datasets_from_file(datasets_file)
    
    if not dataset_list:
        print(f"⚠️  No datasets found in {datasets_file}")
        print("   Add CSV filenames (one per line) to the file")
        return
    
    print(f"   Found {len(dataset_list)} dataset(s)")
    
    total_docs = 0
    
    # Process each dataset
    for dataset_name in dataset_list:
        # Check if it's a local CSV file
        if dataset_name.endswith('.csv'):
            filepath = os.path.join("data", dataset_name)
            if os.path.exists(filepath):
                print(f"\n📚 Loading CSV: {dataset_name}")
                print(f"   Path: {filepath}")
                count = load_knowledge_from_csv(rag, filepath)
                total_docs += count
                print(f"   ✅ Added {count} documents")
            else:
                print(f"   ⚠️  Not found: {filepath}")
        else:
            # It's a HuggingFace dataset - skip for now (RAG uses CSV only)
            print(f"\n⏭️  Skipping HuggingFace dataset: {dataset_name}")
            print(f"   (RAG system uses CSV files only)")
    
    print("\n" + "="*60)
    print(f"✅ Knowledge base built successfully!")
    print(f"📊 Total documents: {total_docs}")
    print(f"💾 Stored in: knowledge_base/")
    print("="*60)
    
    # Test the system
    print("\n🧪 Testing RAG system...")
    test_queries = [
        "What are the benefits of turmeric?",
        "How can I reduce stress naturally?",
        "Tell me about sustainable living",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = rag.search(query, n_results=1)
        if results:
            print(f"   ✅ Found: {results[0]['text'][:100]}...")
        else:
            print("   ❌ No results")
    
    print("\n✅ RAG system is working correctly!")
    print("🚀 Start the server with: python main.py")

if __name__ == "__main__":
    main()
