"""
AIClient - Orchestrator for LLM interactions and tool usage.
Handles smart context injection and conversation flow.
"""

import os
import json
from typing import List, Dict, Any, Optional, Protocol, Generator
from loguru import logger

# Import necessary components from handlers and context_manager
from handlers import SecureFileSystemHandler, SafeScriptExecutor

class EngineProtocol(Protocol):
    """LLM engine protocol."""
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                 temperature: Optional[float] = None, **kwargs) -> str: ...
    @property
    def is_loaded(self) -> bool: ...

class ToolRegistry:
    TOOLS: List[Dict[str, Any]] = [
        {"type": "function", "function": {"name": "read_file", "description": "Read file content", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "list_directory", "description": "List directory contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "create_file", "description": "Create file with content", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
        {"type": "function", "function": {"name": "create_directory", "description": "Create directory", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "delete_file", "description": "Delete file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "delete_directory", "description": "Delete directory recursively", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "rename_item", "description": "Rename/move file or directory", "parameters": {"type": "object", "properties": {"old_path": {"type": "string"}, "new_path": {"type": "string"}}, "required": ["old_path", "new_path"]}}},
        {"type": "function", "function": {"name": "edit_file", "description": "Overwrite file content", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
        {"type": "function", "function": {"name": "run_python_script", "description": "Run a Python script with optional console input. Returns stdout, stderr, and exit code.", "parameters": {"type": "object", "properties": {"script_path": {"type": "string"}, "input_data": {"type": "string", "description": "Optional console input to feed to stdin"}}, "required": ["script_path"]}}},
        {"type": "function", "function": {"name": "search_file", "description": "Recursively search for a pattern (string) within all code and text files in a specified directory.", "parameters": {"type": "object", "properties": {"directory_path": {"type": "string"}, "pattern": {"type": "string"}}, "required": ["directory_path", "pattern"]}}},
    ]

    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        return cls.TOOLS

class AIClient:
    def __init__(self, 
                 openai_client=None,
                 fs_handler=None,
                 code_executor=None,
                 system_prompt: str = "You are a helpful assistant.",
                 max_iterations: int = 200):
        self.openai_client = openai_client
        self.fs_handler = fs_handler
        self.code_executor = code_executor
        self.max_iterations = max_iterations
        
        if self.fs_handler and self.code_executor:
            # Use the passed handler instance instead of a global variable
            self.tool_map = {
                "read_file": lambda **kw: self.fs_handler.read_file(kw.get("path", "")),
                "list_directory": lambda **kw: self.fs_handler.list_directory(kw.get("path", "")),
                "create_file": lambda **kw: self.fs_handler.create_file(kw.get("path", ""), kw.get("content", "")),
                "create_directory": lambda **kw: self.fs_handler.create_directory(kw.get("path", "")),
                "delete_file": lambda **kw: self.fs_handler.delete_file(kw.get("path", "")),
                "delete_directory": lambda **kw: self.fs_handler.delete_directory(kw.get("path", "")),
                "rename_item": lambda **kw: self.fs_handler.rename_item(kw.get("old_path", ""), kw.get("new_path", "")),
                "edit_file": lambda **kw: self.fs_handler.edit_file(kw.get("path", ""), kw.get("content", "")),
            }
        else:
            self.tool_map = {}
        
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    
    def _log(self, msg: str):
        logger.info(msg)

    @staticmethod
    def _sdk_message_to_dict(message: Any) -> Dict[str, Any]:
        if isinstance(message, dict): return message
        if hasattr(message, "model_dump"): return message.model_dump(exclude_none=True)
        if hasattr(message, "dict"): return message.dict(exclude_none=True)
        return {"role": getattr(message, "role", "assistant"), "content": getattr(message, "content", "") or ""}

    def add_user_message(self, user_message: str):
        """Immediately adds user message to history (before LLM call)."""
        if not user_message or not isinstance(user_message, str):
            return False
        
        self.messages.append({"role": "user", "content": user_message})
        return True

    def chat(self, user_message: str) -> str:
        """Standard non-streaming chat."""
        full_response = []
        for chunk in self.chat_stream(user_message):
            full_response.append(chunk)
        return "".join(full_response)

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """Streaming version of the chat orchestrator."""
        self._log("-" * 50)
        self._log(f"[User] {user_message}")
        
        if not self.add_user_message(user_message):
            yield "Error: Invalid or empty message provided."
            return

        use_tools = bool(ToolRegistry.get_tools())

        for i in range(self.max_iterations):
            self._log(f"[Iteration {i+1}] Sending request to LLM...")
            try:
                api_kwargs = {
                    "model": os.getenv("MODEL", "local-model"),
                    "messages": self.messages,
                    "temperature": 0.6,
                    "max_tokens": 262144
                }

                if use_tools:
                    api_kwargs["tools"] = ToolRegistry.get_tools()
                    api_kwargs["tool_choice"] = "auto"

                # Check if AI wants to use tools (non-streaming call)
                response = self.openai_client.chat.completions.create(**api_kwargs)
                message = response.choices[0].message
                tool_calls = getattr(message, 'tool_calls', None)
                
                if tool_calls and len(tool_calls) > 0:
                    self._log(f"[AI] Decided to use tools ({len(tool_calls)} call(s))")
                    self.messages.append(self._sdk_message_to_dict(message))
                    for tc in tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                            self._log(f"[Tool] Executing: {tc.function.name}({json.dumps(args, ensure_ascii=False)})")
                            func = self.tool_map.get(tc.function.name)
                            result = func(**args) if func else f"Error: Unknown tool '{tc.function.name}'"
                            full_result = result if isinstance(result, str) else json.dumps(result.__dict__)
                            self._log(f"[Result] {full_result[:300]}...")
                            self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": full_result})
                        except Exception as e:
                            self._log(f"[Error] Tool error: {e}")
                            self.messages.append({"role": "tool", "tool_call_id": getattr(tc, 'id', 'unknown'), "content": f"Error: {e}"})
                    continue
                else:
                    # FINAL RESPONSE - Now we stream the actual content
                    final_content = message.content
                    self.messages.append(self._sdk_message_to_dict(message))
                    self._log(f"[AI] {final_content}")
                    
                    # To provide a smooth UI experience, we yield the already received content 
                    # in chunks since the tool-check call already gave us the answer.
                    for i in range(0, len(final_content), 15):
                        yield final_content[i:i+15]
                    return

            except Exception as e:
                self._log(f"[Error] {e}")
                yield f"Error communicating with AI: {e}"
                return

        yield "Error: Max iterations reached."
