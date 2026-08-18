# AutoCoder-CLI

A powerful AI-driven CLI for autonomous software engineering, developed via vibe coding. Empower LLMs to act as full-stack developers with a comprehensive toolset for local file manipulation, headless JS-enabled browsing, and direct code execution. Transform your terminal into an agentic environment for building and debugging at scale.

## 🚀 Key Features

AutoCoder-CLI provides a suite of tools that allow an AI agent to interact with your system and the web autonomously:

### 📂 File System Management
- **Read & Write**: Full CRUD operations for files and directories (`read_file`, `create_file`, `edit_file`, `delete_file`).
- **Organization**: Create directories, rename items, and manage folder structures (`create_directory`, `rename_item`, `delete_directory`).
- **Discovery**: List directory contents and perform recursive pattern searches across the codebase (`list_directory`, `search_file`).

### 🌐 Advanced Web Browsing
- **Headless Browser**: A full-featured browser based on Chromium that renders JavaScript, bypassing static HTML placeholders.
- **AI-Ready Content**: Automatically cleans HTML noise (scripts, styles, navs) and converts pages to **Markdown** for optimal LLM token usage.
- **Interactivity**: Ability to click elements, fill forms, and take screenshots of the rendered page.

### ⚙️ Code Execution
- **Python Runtime**: Execute Python scripts directly from the CLI, allowing the agent to run tests, data processing scripts, or custom automation.

## 🛠 Installation

### Prerequisites
- Python 3.10+

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/kostikovn/AutoCoder-CLI.git
   cd AutoCoder-CLI
   ```
2. Run the installation script to install dependencies and the Chromium browser:

   **Windows:**
   ```bash
   .\install.bat
   ```
   **macOS / Linux:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

## 📖 Usage

Start the application to begin interacting with the AI agent. The agent will automatically use the registered tools to perform tasks based on your requests.

```bash
python main.py
```
*(Replace `main.py` with the actual entry point of the application)*

## 📜 License

This project is licensed under the **MIT License**. Feel free to use, fork, and modify it, provided that the original copyright notice is preserved.
