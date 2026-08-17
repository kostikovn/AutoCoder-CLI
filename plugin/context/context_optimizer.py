"""
Professional Context Optimizer Plugin.
Implements dynamic summarization and hard pruning to maintain optimal context window.
"""

import tiktoken
from typing import List, Dict, Any, Optional
from core import IContextPlugin, ClientConfig
from loguru import logger

class ContextOptimizerPlugin(IContextPlugin):
    def __init__(self, model_encoding: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(model_encoding)

    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Calculates total tokens in the conversation history."""
        total = 0
        for msg in messages:
            # Basic token count for each message
            # In a production environment, we'd use the specific chat template overhead
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(self.encoding.encode(content))
            else:
                total += 0 # Handle non-string content if necessary
        return total

    def pre_process(self, messages: List[Dict[str, Any]], config: ClientConfig, llm_client: Optional[Any] = None) -> List[Dict[str, Any]]:
        if not messages:
            return messages

        current_tokens = self._count_tokens(messages)
        
        # Check if we are above the summary threshold
        if current_tokens >= config.max_context_tokens * config.summary_threshold:
            logger.info(f"Context size ({current_tokens}) exceeded threshold. Starting optimization...")
            
            # If we have an LLM client, we attempt summarization
            if llm_client:
                return self._summarize_context(messages, config, llm_client)
            else:
                logger.warning("LLM client not provided. Falling back to hard pruning.")
                return self._hard_prune(messages, config)
        
        # Final check: even if not summarized, ensure we are below hard limit
        if current_tokens > config.max_context_tokens:
            return self._hard_prune(messages, config)

        return messages

    def _summarize_context(self, messages: List[Dict[str, Any]], config: ClientConfig, llm_client: Any) -> List[Dict[str, Any]]:
        """
        Summarizes the middle part of the conversation to free up space.
        Preserves System Prompt and the last few messages.
        """
        if len(messages) <= 3:
            return messages

        system_prompt = messages[0]
        recent_history = messages[-3:] # Keep last 3 messages for continuity
        to_summarize = messages[1:-3]

        if not to_summarize:
            return messages

        try:
            # Construct a prompt for summarization
            history_text = "\\n".join([f"{m['role']}: {m['content']}" for m in to_summarize])
            summary_prompt = (
                "Summarize the following conversation history. "
                "Keep all technical details, decisions, and key facts. "
                "Be concise but comprehensive. "
                f"\\n\\nHistory:\\n{history_text}"
            )

            # Use a simple completion request for the summary
            # We use a minimal config for the summary call to avoid recursion/overhead
            response = llm_client.create_completion(
                messages=[{"role": "user", "content": summary_prompt}],
                model=config.model,
                temperature=0.3,
                max_tokens=1000
            )
            
            summary = response.choices[0].message.content
            
            # New history: System + Summary + Recent
            new_messages = [
                system_prompt,
                {"role": "system", "content": f"[Context Summary]: {summary}"},
                *recent_history
            ]
            
            logger.info("Context successfully summarized.")
            return new_messages

        except Exception as e:
            logger.error(f"Summarization failed: {e}. Falling back to hard prune.")
            return self._hard_prune(messages, config)

    def _hard_prune(self, messages: List[Dict[str, Any]], config: ClientConfig) -> List[Dict[str, Any]]:
        """FIFO pruning: removes oldest messages while keeping the system prompt."""
        if len(messages) <= 1:
            return messages

        system_prompt = messages[0]
        remaining = messages[1:]
        
        # Remove oldest until we fit in the limit
        while remaining and (self._count_tokens([system_prompt] + remaining) > config.max_context_tokens):
            remaining.pop(0)
            
        return [system_prompt] + remaining

    def post_process(self, response: Any, messages: List[Dict[str, Any]], config: ClientConfig) -> Any:
        return response
