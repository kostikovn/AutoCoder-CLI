"""
GUI Application for AI Console Editor.
Uses CustomTkinter for a modern look and feel.
Implements an asynchronous chat interface with a semantic context side-panel.
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional

# --- PATH CONFIGURATION ---
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    import customtkinter as ctk
except ImportError:
    print("Error: 'customtkinter' package not found. Please install it: pip install customtkinter")
    exit(1)

from loguru import logger
from dotenv import load_dotenv

from ai_client import AIClient, OpenAIClientAdapter, ToolManager, ToolRegistry, ContextPluginManager
from handlers import SecureFileSystemHandler, SafeScriptExecutor
from core import ClientConfig, ToolContext

load_dotenv(current_dir / "settings.env")

def create_ai_client() -> AIClient:
    """
    Bootstrap function to wire up dependencies.
    Satisfies Dependency Inversion by constructing the graph here.
    """
    workspace_dir = os.getenv("WORKSPACE_DIR", "./workspace")
    
    # 1. Configuration
    config = ClientConfig(
        model=os.getenv("MODEL", "local-model"),
        temperature=float(os.getenv("TEMPERATURE", 0.6)),
        max_tokens=int(os.getenv("MAX_TOKENS", 262144)),
        system_prompt=os.getenv("SYSTEM_PROMPT", "You are a helpful assistant."),
        max_iterations=int(os.getenv("MAX_ITERATIONS", 200))
    )
    
    # 2. LLM Client
    llm_client = OpenAIClientAdapter(
        api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    # 3. Handlers
    fs_handler = SecureFileSystemHandler(workspace_dir)
    executor = SafeScriptExecutor(workspace_dir)
    
    # 4. Tool Context & Manager
    tool_context = ToolContext(
        fs_handler=fs_handler,
        executor=executor
    )
    tool_manager = ToolManager(context=tool_context)
    tool_manager.load_plugins("plugin/tools")

    # 5. Context Plugin Manager
    context_manager = ContextPluginManager()
    context_manager.load_plugins("plugin/context")

    # 6. AI Client
    return AIClient(
        llm_client=llm_client,
        tool_manager=tool_manager,
        context_manager=context_manager,
        config=config
    )




    # 4. AI Client
    return AIClient(
        llm_client=llm_client,
        tool_manager=tool_manager,
        system_prompt=os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
    )

class ConsoleEditorGUI(ctk.CTk):
    def __init__(self, ai_client: AIClient):
        super().__init__()
        self.ai_client = ai_client

        # --- Configuration ---
        self.title("AI Console Editor - Semantic GUI")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # --- UI Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # 1. Main Chat Area
        self.chat_frame = ctk.CTkFrame(self, corner_radius=0)
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1)

        # Chat Display (Scrollable)
        self.chat_display = ctk.CTkTextbox(self.chat_frame, state="disabled", wrap="word")
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Input Area
        self.input_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.input_frame.grid_columnconfigure(1, weight=1)

        self.user_input = ctk.CTkTextbox(self.input_frame, height=100, wrap="word")
        self.user_input.grid(row=0, column=0, sticky="ew", padx=0)
        self.user_input.bind("<Control-Return>", lambda e: self.send_message())

        self.send_button = ctk.CTkButton(self.input_frame, text="Send", width=100, command=self.send_message)
        self.send_button.grid(row=0, column=1, padx=(10, 0), sticky="sw")

        # Status Bar
        self.status_bar = ctk.CTkLabel(self, text="Ready", anchor="w", padx=10)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def clear_chat(self):
        """Clears the chat UI and resets AI client history."""
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        
        system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
        self.ai_client.messages = [{"role": "system", "content": system_prompt}]
        
        self.update_status("Chat reset")

    def update_status(self, text: str):
        self.status_bar.configure(text=text)

    def update_textbox(self, textbox: ctk.CTkTextbox, text: str):
        textbox.configure(state="normal")
        textbox.delete("1.0", "end")
        textbox.insert("1.0", text)
        textbox.configure(state="disabled")

    def append_chat(self, role: str, message: str):
        """Appends a full message to the chat display."""
        self.chat_display.configure(state="normal")
        prefix = "User: " if role == "user" else "AI: "
        if role == "system": prefix = "System: "
        self.chat_display.insert("end", f"{prefix}{message}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def append_token(self, token: str):
        """Appends a single token to the last AI message."""
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", token)
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def send_message(self):
        user_text = self.user_input.get("1.0", "end-1c").strip()
        if not user_text:
            return
        
        self.user_input.delete("1.0", "end")
        self.append_chat("user", user_text)
        threading.Thread(target=self._process_ai_response, args=(user_text,), daemon=True).start()

    def _process_ai_response(self, text: str):
        try:
            self.update_status("AI is thinking...")
            self.after(0, lambda: self.append_chat("AI: ", "")) 
            
            full_response = ""
            for token in self.ai_client.chat_stream(text):
                full_response += token
                self.after(0, lambda t=token: self.append_token(t))

            self.after(0, lambda: self.append_chat("system", "\n")) 
            self.after(0, lambda: self.update_status("Ready"))
        except Exception as e:
            logger.error(f"Chat error: {e}")
            self.after(0, lambda: self.append_chat("system", f"Error: {str(e)}"))
            self.after(0, lambda: self.update_status("Error occurred"))

if __name__ == "__main__":
    ai_client = create_ai_client()
    app = ConsoleEditorGUI(ai_client)
    app.mainloop()
