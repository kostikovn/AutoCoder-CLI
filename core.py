"""
Core abstractions and configurations for the AI Console Editor.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Protocol, runtime_checkable

from handlers import IFileSystemHandler, IScriptExecutor

@dataclass(frozen=True)
class ClientConfig:
    """Configuration for the AI Client."""
    model: str = "local-model"
    temperature: float = 0.6
    max_tokens: int = 262144
    system_prompt: str = "You are a helpful assistant."
    max_iterations: int = 200

@dataclass
class ToolContext:
    """Context provided to tools during execution."""
    fs_handler: IFileSystemHandler
    executor: IScriptExecutor
    extra: Dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class IContextPlugin(Protocol):
    """Interface for context plugins to modify conversation history."""
    def pre_process(self, messages: List[Dict[str, Any]], config: ClientConfig) -> List[Dict[str, Any]]:
        """Modify messages before sending to LLM."""
        ...

    def post_process(self, response: Any, messages: List[Dict[str, Any]], config: ClientConfig) -> Any:
        """Modify LLM response or messages after receiving response."""
        ...
