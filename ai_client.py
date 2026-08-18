"""
AIClient - Orchestrator for LLM interactions and tool usage.
Handles smart context injection and conversation flow.
"""

import os
import json
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol, Generator, Callable
from loguru import logger
from openai import OpenAI

from handlers import IFileSystemHandler, IScriptExecutor
from core import ClientConfig, ToolContext, IContextPlugin

class LLMResponse(Protocol):
    """Protocol for LLM response message."""
    content: Optional[str]
    tool_calls: Optional[List[Any]]

class LLMCompletion(Protocol):
    """Protocol for LLM completion object."""
    choices: List[Any]

class ILLMClient(Protocol):
    """Interface for LLM clients to ensure Dependency Inversion Principle."""
    def create_completion(self, 
                          messages: List[Dict[str, Any]], 
                          tools: Optional[List[Dict[str, Any]]] = None, 
                          tool_choice: str = "auto", 
                          **kwargs) -> LLMCompletion: ...

class OpenAIClientAdapter(ILLMClient):
    """Adapter for OpenAI client to satisfy ILLMClient interface."""
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def create_completion(self, 
                          messages: List[Dict[str, Any]], 
                          tools: Optional[List[Dict[str, Any]]] = None, 
                          tool_choice: str = "auto", 
                          **kwargs) -> LLMCompletion:
        return self.client.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )

class ToolManager:
    """
    Handles registration and execution of tools.
    Supports dynamic loading of tools from plugins directory.
    """
    def __init__(self, context: ToolContext = None):
        self._tools_definitions: Dict[str, Dict[str, Any]] = {}
        self._tools_implementations: Dict[str, Callable] = {}
        self.context = context

    def register_tool(self, name: str, definition: Dict[str, Any], implementation: Callable):
        """Registers a tool with its definition and implementation."""
        self._tools_definitions[name] = definition
        self._tools_implementations[name] = implementation
        logger.debug(f"Tool registered: {name}")

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Executes a registered tool, passing the context."""
        if name not in self._tools_implementations:
            return f"Error: Tool '{name}' is not registered."
        
        try:
            # Pass context as the first argument if the implementation expects it
            return self._tools_implementations[name](self.context, **args)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error executing tool {name}: {e}"

    def get_all_definitions(self) -> List[Dict[str, Any]]:
        """Returns tool definitions in the format expected by LLMs."""
        return [
            {"type": "function", "function": defn} 
            for defn in self._tools_definitions.values()
        ]

    def load_plugins(self, plugins_dir: str):
        """Dynamically loads tools from the specified directory."""
        path = Path(plugins_dir)
        if not path.exists() or not path.is_dir():
            logger.warning(f"Plugins directory {plugins_dir} not found.")
            return

        for file in path.glob("*.py"):
            if file.name == "__init__.py":
                continue
            
            try:
                spec = importlib.util.spec_from_file_location(file.stem, file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, "TOOL_DEFINITION") and hasattr(module, "run"):
                    name = module.TOOL_DEFINITION["name"]
                    self.register_tool(name, module.TOOL_DEFINITION, module.run)
                else:
                    logger.warning(f"Plugin {file.name} is missing TOOL_DEFINITION or run function.")
            except Exception as e:
                logger.error(f"Failed to load plugin {file.name}: {e}")

class ContextPluginManager:
    """
    Handles loading and execution of context plugins.
    Ensures conversation flow can be modified by external plugins.
    """
    def __init__(self):
        self.plugins: List[IContextPlugin] = []

    def load_plugins(self, plugins_dir: str):
        """Dynamically loads context plugins from directory."""
        path = Path(plugins_dir)
        if not path.exists() or not path.is_dir():
            logger.warning(f"Context plugins directory {plugins_dir} not found.")
            return

        for file in path.glob("*.py"):
            if file.name.startswith("_") or file.name == "__init__.py":
                continue
            try:
                spec = importlib.util.spec_from_file_location(file.stem, file)

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Expect a class that implements IContextPlugin
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, IContextPlugin) and attr is not IContextPlugin:
                        try:
                            # Try to instantiate the plugin. 
                            # If it fails due to missing args, we skip it here 
                            # because complex plugins should be added manually in bootstrap.
                            plugin_instance = attr()
                            self.plugins.append(plugin_instance)
                            logger.debug(f"Context plugin loaded: {attr_name} from {file.name}")
                        except TypeError as e:
                            logger.warning(f"Could not auto-load plugin {attr_name} from {file.name}: {e}. "
                                           f"This plugin likely requires manual initialization in bootstrap.")
            except Exception as e:
                logger.error(f"Failed to load context plugin {file.name}: {e}")


    def apply_pre_process(self, messages: List[Dict[str, Any]], config: ClientConfig, llm_client: Optional[Any] = None) -> List[Dict[str, Any]]:
        processed_messages = list(messages)
        for plugin in self.plugins:
            processed_messages = plugin.pre_process(processed_messages, config, llm_client)
        
        # Consolidate all system messages into a single one at the beginning
        # This prevents errors in strict models (like Qwen/Llama-3) that require 
        # exactly one system message at the start.
        system_contents = []
        non_system_messages = []
        
        for msg in processed_messages:
            if msg.get("role") == "system":
                system_contents.append(msg.get("content", ""))
            else:
                non_system_messages.append(msg)
        
        if not system_contents:
            # Fallback to ensure a system message always exists
            system_contents.append(config.system_prompt)
            
        consolidated_system_msg = {
            "role": "system", 
            "content": "\\n\\n".join(system_contents)
        }
        
        return [consolidated_system_msg] + non_system_messages


    def apply_post_process(self, response: Any, messages: List[Dict[str, Any]], config: ClientConfig) -> Any:
        processed_response = response
        for plugin in self.plugins:
            processed_response = plugin.post_process(processed_response, messages, config)
        return processed_response

class ToolRegistry:
    """
    Provides default tool definitions.
    Used to populate ToolManager.
    """
    DEFAULT_TOOLS = [
        {"name": "read_file", "description": "Read file content", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "list_directory", "description": "List directory contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "create_file", "description": "Create file with content", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
        {"name": "create_directory", "description": "Create directory", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "delete_file", "description": "Delete file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "delete_directory", "description": "Delete directory recursively", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "rename_item", "description": "Rename/move file or directory", "parameters": {"type": "object", "properties": {"old_path": {"type": "string"}, "new_path": {"type": "string"}}, "required": ["old_path", "new_path"]}},
        {"name": "edit_file", "description": "Overwrite file content", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
        {"name": "run_python_script", "description": "Run a Python script with optional console input. Returns stdout, stderr, and exit code.", "parameters": {"type": "object", "properties": {"script_path": {"type": "string"}, "input_data": {"type": "string", "description": "Optional console input to feed to stdin"}}, "required": ["script_path"]}},
        {"name": "search_file", "description": "Recursively search for a pattern (string) within all code and text files in a specified directory.", "parameters": {"type": "object", "properties": {"directory_path": {"type": "string"}, "pattern": {"type": "string"}}, "required": ["directory_path", "pattern"]}},
    ]

    @classmethod
    def get_defaults(cls) -> List[Dict[str, Any]]:
        return cls.DEFAULT_TOOLS

class AIClient:
    """
    Orchestrator for LLM interactions and tool usage.
    Depends on abstractions (ILLMClient, ToolManager) to satisfy SOLID.
    """
    def __init__(self, 
                 llm_client: ILLMClient,
                 tool_manager: ToolManager,
                 context_manager: ContextPluginManager,
                 config: ClientConfig,
                 system_prompt: Optional[str] = None,
                 max_iterations: Optional[int] = None):
        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.context_manager = context_manager
        self.config = config
        self.max_iterations = max_iterations or config.max_iterations
        
        prompt = system_prompt or config.system_prompt
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": prompt}]
    
    def _log(self, msg: str):
        logger.info(msg)

    @staticmethod
    def _message_to_dict(message: Any) -> Dict[str, Any]:
        if isinstance(message, dict): return message
        if hasattr(message, "model_dump"): return message.model_dump(exclude_none=True)
        if hasattr(message, "dict"): return message.dict(exclude_none=True)
        return {"role": getattr(message, "role", "assistant"), "content": getattr(message, "content", "") or ""}

    def add_user_message(self, user_message: str) -> bool:
        if not user_message or not isinstance(user_message, str):
            return False
        self.messages.append({"role": "user", "content": user_message})
        return True

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        self._log("-" * 50)
        self._log(f"[User] {user_message}")
        
        if not self.add_user_message(user_message):
            yield "Error: Invalid or empty message provided."
            return

        tools = self.tool_manager.get_all_definitions()

        for i in range(self.max_iterations):
            self._log(f"[Iteration {i+1}] Sending request to LLM...")
            try:
                # Apply pre-processing context plugins
                current_messages = self.context_manager.apply_pre_process(self.messages, self.config, self.llm_client)
                
                response = self.llm_client.create_completion(
                    messages=current_messages,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )

                
                # Apply post-processing context plugins
                response = self.context_manager.apply_post_process(response, self.messages, self.config)
                
                message = response.choices[0].message
                tool_calls = getattr(message, 'tool_calls', None)
                
                if tool_calls and len(tool_calls) > 0:
                    self._log(f"[AI] Decided to use tools ({len(tool_calls)} call(s))")
                    self.messages.append(self._message_to_dict(message))
                    
                    for tc in tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                            self._log(f"[Tool] Executing: {tc.function.name}({json.dumps(args, ensure_ascii=False)})")
                            
                            result = self.tool_manager.execute_tool(tc.function.name, args)
                            
                            full_result = result if isinstance(result, str) else json.dumps(result.__dict__)
                            self._log(f"[Result] {full_result[:300]}...")
                            self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": full_result})
                        except Exception as e:
                            self._log(f"[Error] Tool error: {e}")
                            self.messages.append({"role": "tool", "tool_call_id": getattr(tc, 'id', 'unknown'), "content": f"Error: {e}"})
                    continue
                else:
                    final_content = message.content or ""
                    self.messages.append(self._message_to_dict(message))
                    self._log(f"[AI] {final_content}")
                    
                    for i in range(0, len(final_content), 15):
                        yield final_content[i:i+15]
                    return

            except Exception as e:
                self._log(f"[Error] {e}")
                yield f"Error communicating with AI: {e}"
                return

        yield "Error: Max iterations reached."

    def chat(self, user_message: str) -> str:
        return "".join(list(self.chat_stream(user_message)))
