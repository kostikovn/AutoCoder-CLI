import os
import shutil
from typing import List, Dict, Any, Optional
from loguru import logger

# Add workspace to path so we can import core, ai_client, etc.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import ClientConfig, ToolContext, IContextPlugin
from ai_client import AIClient, ToolManager, ContextPluginManager
from memory import ConversationMemory
from plugin.context.memory_plugin import MemoryContextPlugin
from plugin.context.context_optimizer import ContextOptimizerPlugin

# --- Mocks ---

class MockResponse:
    def __init__(self, content):
        self.choices = [self.Choice(self.Message(content))]
    class Choice:
        def __init__(self, message):
            self.message = message
    class Message:
        def __init__(self, content):
            self.content = content
            self.tool_calls = None

class MockLLMClient:
    def create_completion(self, messages, tools=None, tool_choice="auto", **kwargs):
        # Simply return a mock response
        return MockResponse("This is a mock AI response.")

# --- Test Suite ---

def test_rag_memory():
    print("\n--- Testing RAG Memory (Long-term) ---")
    test_mem_dir = os.path.abspath("./tests/.test_memory_dir")
    if os.path.exists(test_mem_dir):
        shutil.rmtree(test_mem_dir)
    os.makedirs(test_mem_dir)
    
    # Set environment variable so plugin uses this path
    os.environ["WORKSPACE_DIR"] = test_mem_dir
    
    plugin = MemoryContextPlugin()
    config = ClientConfig()
    
    # 1. Save a fact
    messages = [{"role": "user", "content": "My secret code is 'GOLDEN-APPLE-777'"},
                {"role": "assistant", "content": "I have remembered your secret code."}]
    
    # Save messages via post_process
    mock_resp = MockResponse("I have remembered your secret code.")
    plugin.post_process(mock_resp, messages, config)
    
    # 2. Clear current session and ask about the fact
    new_messages = [{"role": "user", "content": "What is my secret code?"}]
    processed = plugin.pre_process(new_messages, config)
    
    # Check if memory was injected
    found = any("GOLDEN-APPLE-777" in m["content"] for m in processed if m["role"] == "system")
    if found:
        print("SUCCESS: RAG Memory: Fact successfully retrieved and injected.")
    else:
        print("FAILURE: RAG Memory: Fact not found in processed messages.")
        print(f"Processed messages: {processed}")
    
    shutil.rmtree(test_mem_dir)

def test_context_optimization():
    print("\n--- Testing Context Optimization (Pruning/Summarization) ---")
    plugin = ContextOptimizerPlugin()
    config = ClientConfig(max_context_tokens=1000) # Set a low limit for testing
    
    # Create a huge conversation (50 long messages)
    messages = []
    for i in range(50):
        messages.append({"role": "user", "content": "This is a very long message designed to fill up the context window " * 10})
        messages.append({"role": "assistant", "content": "I am responding to your very long message with another long response " * 10})
    
    original_len = len(messages)
    processed = plugin.pre_process(messages, config)
    processed_len = len(processed)
    
    print(f"Original messages: {original_len}")
    print(f"Processed messages: {processed_len}")
    
    if processed_len < original_len:
        print("SUCCESS: Context Optimization: Conversation was successfully pruned/summarized.")
    else:
        print("FAILURE: Context Optimization: Conversation length was not reduced.")

def test_full_pipeline_stability():
    print("\n--- Testing Full Pipeline Stability ---")
    # This test ensures that adding multiple plugins doesn't crash the system
    config = ClientConfig()
    context_manager = ContextPluginManager()
    
    # Manually add plugins to simulate the loaded state
    # We use a temp memory path
    test_mem_dir = os.path.abspath("./tests/.test_pipeline_dir")
    if os.path.exists(test_mem_dir): shutil.rmtree(test_mem_dir)
    os.makedirs(test_mem_dir)
    os.environ["WORKSPACE_DIR"] = test_mem_dir
    
    context_manager.plugins.append(MemoryContextPlugin())
    context_manager.plugins.append(ContextOptimizerPlugin())
    
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    
    try:
        # Test pre-process
        processed = context_manager.apply_pre_process(messages, config)
        # Test post-process
        mock_resp = MockResponse("I am doing great!")
        context_manager.apply_post_process(mock_resp, processed, config)
        print("SUCCESS: Full Pipeline: Pre and Post processing executed without errors.")
    except Exception as e:
        print(f"FAILURE: Full Pipeline: Crashed with error: {e}")
    finally:
        if os.path.exists(test_mem_dir): shutil.rmtree(test_mem_dir)

if __name__ == "__main__":
    try:
        test_rag_memory()
        test_context_optimization()
        test_full_pipeline_stability()
    except Exception as e:
        print(f"Test suite failed: {e}")
