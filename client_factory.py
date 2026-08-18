import os
import sys
import requests
from pathlib import Path
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

from ai_client import AIClient, OpenAIClientAdapter, ToolManager, ToolContext, ContextPluginManager
from handlers import SecureFileSystemHandler, SafeScriptExecutor
from core import ClientConfig

def get_remote_context_size(base_url: Optional[str], api_key: Optional[str] = None) -> Optional[int]:
    """Fetches the max context size from the llama-server /slots endpoint."""
    if not base_url:
        return None
    try:
        url = base_url.replace("/v1", "")
        if not url.endswith("/"):
            url += "/"
        
        request_url = f"{url}slots"
        logger.info(f"Fetching context size from: {request_url}")
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        response = requests.get(request_url, headers=headers, timeout=2)
        if response.status_code == 200:
            slots = response.json()
            if slots and isinstance(slots, list):
                return slots[0].get("n_ctx")
        else:
            logger.info(f"Server returned non-200 status for /slots: {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not fetch remote context size from {base_url}: {e}")
    return None

def create_ai_client() -> AIClient:
    """
    Bootstrap function to wire up dependencies.
    """
    current_dir = Path(__file__).parent.absolute()
    load_dotenv(current_dir / "settings.env")
    
    workspace_dir = os.getenv("WORKSPACE_DIR", "./workspace")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
    remote_ctx = get_remote_context_size(base_url, api_key)
    
    if remote_ctx:
        logger.info(f"Dynamic context window detected: {remote_ctx} tokens")
    else:
        logger.info("Could not detect remote context window, using default.")

    config = ClientConfig(
        model=os.getenv("MODEL", "local-model"),
        temperature=float(os.getenv("TEMPERATURE", 0.6)),
        max_tokens=int(os.getenv("MAX_TOKENS", 262144)),
        system_prompt=os.getenv("SYSTEM_PROMPT", "You are a helpful assistant."),
        max_iterations=int(os.getenv("MAX_ITERATIONS", 200)),
        max_context_tokens=remote_ctx if remote_ctx else 8192
    )
    
    llm_client = OpenAIClientAdapter(
        api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    fs_handler = SecureFileSystemHandler(workspace_dir)
    executor = SafeScriptExecutor(workspace_dir)
    
    tool_context = ToolContext(
        fs_handler=fs_handler,
        executor=executor
    )
    tool_manager = ToolManager(context=tool_context)
    tool_manager.load_plugins("plugin/tools")
    
    context_manager = ContextPluginManager()
    context_manager.load_plugins("plugin/context")
    
    return AIClient(
        llm_client=llm_client,
        tool_manager=tool_manager,
        context_manager=context_manager,
        config=config
    )
