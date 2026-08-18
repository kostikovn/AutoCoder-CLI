"""
Memory Context Plugin.
Injects relevant past conversation fragments into the current prompt using semantic search.
"""

import os
from typing import List, Dict, Any, Optional
from core import IContextPlugin, ClientConfig
from memory import ConversationMemory
from loguru import logger

class MemoryContextPlugin(IContextPlugin):
    def __init__(self):
        workspace_dir = os.getenv("WORKSPACE_DIR", "./workspace")
        memory_path = os.path.join(workspace_dir, ".memory")
        self.memory = ConversationMemory(storage_path=memory_path)
        logger.info(f"MemoryContextPlugin initialized with storage at {memory_path}")

    def pre_process(self, messages: List[Dict[str, Any]], config: ClientConfig, llm_client: Optional[Any] = None) -> List[Dict[str, Any]]:
        # We only perform semantic retrieval for User messages to avoid self-looping on AI's previous answers
        # and to focus on the user's intent.
        if not messages:
            return messages

        
        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            return messages
            
        query = last_msg.get("content", "")
        if not query:
            return messages

        try:
            # Retrieve top 3 relevant memories
            memories = self.memory.query_memories(query, top_k=3)
            
            if memories:
                # Format memories into a single string
                memory_block = "\\n".join([
                    f"[{m['role'].upper()}]: {m['text']}" 
                    for m in memories
                ])
                
                # Inject as a system message before the current conversation
                # We place it after the main system prompt (index 0)
                context_msg = {
                    "role": "system", 
                    "content": f"Relevant memories from past conversations:\\n{memory_block}"
                }
                
                # Insert after the first system message
                messages.insert(1, context_msg)
                logger.debug(f"Injected {len(memories)} memories into context.")
                
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            
        return messages

    def post_process(self, response: Any, messages: List[Dict[str, Any]], config: ClientConfig) -> Any:
        # After a full exchange is complete, we save both user and assistant messages to memory.
        # Since post_process is called per response, we save the last user msg and this response.
        
        try:
            # Save the user message that triggered this response
            for msg in messages:
                if msg.get("role") == "user":
                    # Note: in a real scenario we might want to avoid duplicate saving.
                    # For now, we save the last user message.
                    pass 
            
            # We only save if we have a valid response
            if response and hasattr(response, 'choices'):
                ai_text = response.choices[0].message.content
                # Save the last user message and the AI response
                # To avoid duplicates, we typically do this in the AIClient or a specific handler,
                # but here we implement it as a plugin for consistency.
                
                # Find the last user message
                last_user_msg = None
                for m in reversed(messages):
                    if m.get("role") == "user":
                        last_user_msg = m.get("content", "")
                        break
                
                if last_user_msg:
                    self.memory.add_message("user", last_user_msg)
                
                self.memory.add_message("assistant", ai_text)
                logger.debug("Conversation exchange saved to long-term memory.")
                
        except Exception as e:
            logger.error(f"Saving to memory failed: {e}")
            
        return response
