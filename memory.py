"""
Long-term memory system using ChromaDB and sentence-transformers.
Implements semantic search over conversation history.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import time

class ConversationMemory:
    """
    Handles storage and retrieval of conversation history using vector embeddings.
    """
    def __init__(self, storage_path: str = "./workspace/.memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.storage_path))
        
        # Use a default embedding function from chromadb (which uses sentence-transformers)
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create the 'conversations' collection
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            embedding_function=self.emb_fn
        )

    def add_message(self, role: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Adds a message to the vector store.
        """
        if not text or not text.strip():
            return

        msg_id = str(uuid.uuid4())
        timestamp = time.time()
        
        meta = {
            "role": role,
            "timestamp": timestamp,
            **(metadata or {})
        }
        
        self.collection.add(
            documents=[text],
            metadatas=[meta],
            ids=[msg_id]
        )

    def query_memories(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the most semantically similar messages from history.
        """
        if not query or not query.strip():
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        memories = []
        # results['documents'][0] contains the list of matching documents
        # results['metadatas'][0] contains the corresponding metadata
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            memories.append({
                "text": doc,
                "role": meta.get("role", "unknown"),
                "timestamp": meta.get("timestamp", 0)
            })
            
        return memories

    def clear(self):
        """Clears all stored memories."""
        self.client.delete_collection("conversations")
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            embedding_function=self.emb_fn
        )
