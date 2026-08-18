import os
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from client_factory import create_ai_client

def main():
    """
    Terminal-based entry point for AutoCoder-CLI.
    Provides a simple loop for interacting with the AI agent.
    """
    # Path configuration
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    
    # Initialize the AI client
    try:
        ai_client = create_ai_client()
    except Exception as e:
        logger.error(f"Failed to initialize AI client: {e}")
        sys.exit(1)

    # Set the system prompt from env
    system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
    ai_client.messages = [{"role": "system", "content": system_prompt}]

    print("\n" + "="*50)
    print(" AutoCoder-CLI: Terminal Mode")
    print(" Type 'exit' or 'quit' to stop. Type 'clear' to reset history.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("User > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "clear":
                ai_client.messages = [{"role": "system", "content": system_prompt}]
                print("Chat history cleared.\n")
                continue

            print("AI > ", end="", flush=True)
            
            full_response = ""
            # Use the chat_stream to provide a real-time experience in the terminal
            for token in ai_client.chat_stream(user_input):
                print(token, end="", flush=True)
                full_response += token
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
