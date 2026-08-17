"""
Verification script for the Context Optimizer Plugin.
"""

import os
from ai_client import OpenAIClientAdapter, ToolManager, ContextPluginManager, AIClient
from core import ClientConfig, ToolContext
from handlers import SecureFileSystemHandler, SafeScriptExecutor
from plugin.context.context_optimizer import ContextOptimizerPlugin
from loguru import logger

def test_optimizer():
    # 1. Setup
    config = ClientConfig(
        model="local-model", 
        max_context_tokens=500, # Small limit to trigger pruning easily
        summary_threshold=0.6
    )
    
    # Mock LLM Client to avoid real API calls during unit test
    class MockLLMClient:
        def create_completion(self, messages, **kwargs):
            class MockResponse:
                class Choices:
                    class Message:
                        content = "This is a summary of the conversation."
                        def __init__(self): pass
                choices = [Choices.Message()]
            return MockResponse()

    llm_client = MockLLMClient()
    tool_manager = ToolManager()
    
    # Setup Context Manager with the Optimizer Plugin
    ctx_manager = ContextPluginManager()
    ctx_manager.plugins.append(ContextOptimizerPlugin())
    
    # 2. Create a very long history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! " * 100},
        {"role": "assistant", "content": "Hi there! " * 100},
        {"role": "user", "content": "Tell me more. " * 100},
        {"role": "assistant", "content": "Sure! " * 100},
        {"role": "user", "content": "Final question. " * 100},
    ]
    
    logger.info(f"Original messages count: {len(messages)}")
    
    # 3. Apply pre-processing
    processed = ctx_manager.apply_pre_process(messages, config, llm_client)
    
    logger.info(f"Processed messages count: {len(processed)}")
    
    # 4. Assertions
    assert len(processed) < len(messages), "History should have been pruned or summarized"
    assert processed[0]["role"] == "system", "System prompt must be preserved"
    
    # Check if summarization happened (should contain the summary string from MockLLMClient)
    has_summary = any("[Context Summary]" in str(m.get("content", "")) or "summary" in str(m.get("content", "")).lower() for m in processed)
    logger.info(f"Summarization occurred: {has_summary}")

if __name__ == "__main__":
    test_optimizer()
    logger.info("Verification test passed!")
