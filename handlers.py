"""
Handlers for file system operations and code execution.
Implements SRP by isolating I/O logic from AI orchestration.
"""

import os
import shutil
import subprocess
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class ExecutionResult:
    """Immutable result of a script execution."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    message: str


class IFileSystemHandler(ABC):
    """Interface for file system operations to ensure Liskov Substitution Principle."""

    @abstractmethod
    def read_file(self, path: str) -> str: ...

    @abstractmethod
    def list_directory(self, path: str) -> str: ...

    @abstractmethod
    def create_file(self, path: str, content: str) -> str: ...

    @abstractmethod
    def create_directory(self, path: str) -> str: ...

    @abstractmethod
    def delete_file(self, path: str) -> str: ...

    @abstractmethod
    def delete_directory(self, path: str) -> str: ...

    @abstractmethod
    def rename_item(self, old_path: str, new_path: str) -> str: ...

    @abstractmethod
    def edit_file(self, path: str, content: str) -> str: ...

    @abstractmethod
    def search_files(self, directory_path: str, pattern: str) -> str: ...


class SecureFileSystemHandler(IFileSystemHandler):
    """
    Implementation of IFileSystemHandler that restricts access to a specific workspace.
    Prevents Path Traversal attacks by resolving and validating paths.
    """

    def __init__(self, workspace_dir: str):
        self.base = Path(workspace_dir).resolve()
        self.base.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        """Resolves path and ensures it stays within the workspace."""
        resolved = (self.base / path).resolve()
        if not str(resolved).startswith(str(self.base)):
            raise PermissionError(f"Access denied: {path} is outside the workspace")
        return resolved

    def read_file(self, path: str) -> str:
        try:
            return self._resolve(path).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file {path}: {e}"

    def list_directory(self, path: str) -> str:
        try:
            p = self._resolve(path)
            if not p.is_dir():
                return f"Error: {path} is not a directory"
            
            items = [f"{i.name}/" if i.is_dir() else i.name for i in p.iterdir()]
            return "\n".join(sorted(items)) if items else "Empty directory"
        except Exception as e:
            return f"Error listing directory {path}: {e}"

    def create_file(self, path: str, content: str) -> str:
        try:
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Successfully created/updated: {path}"
        except Exception as e:
            return f"Error creating file {path}: {e}"

    def create_directory(self, path: str) -> str:
        try:
            self._resolve(path).mkdir(parents=True, exist_ok=True)
            return f"Successfully created directory: {path}"
        except Exception as e:
            return f"Error creating directory {path}: {e}"

    def delete_file(self, path: str) -> str:
        try:
            self._resolve(path).unlink()
            return f"Deleted file: {path}"
        except Exception as e:
            return f"Error deleting file {path}: {e}"

    def delete_directory(self, path: str) -> str:
        try:
            p = self._resolve(path)
            if p.is_dir():
                shutil.rmtree(p)
                return f"Deleted directory: {path}"
            return f"Error: {path} is not a directory"
        except Exception as e:
            return f"Error deleting directory {path}: {e}"

    def rename_item(self, old_path: str, new_path: str) -> str:
        try:
            old = self._resolve(old_path)
            new = self._resolve(new_path)
            old.rename(new)
            return f"Renamed: {old_path} -> {new_path}"
        except Exception as e:
            return f"Error renaming item from {old_path}: {e}"

    def edit_file(self, path: str, content: str) -> str:
        # In this implementation, editing is equivalent to creating/overwriting
        return self.create_file(path, content)

    def search_files(self, directory_path: str, pattern: str) -> str:
        try:
            root = self._resolve(directory_path)
            if not root.is_dir():
                return f"Error: {directory_path} is not a valid directory."

            results: List[str] = []
            # Search only in common text/code files to avoid binary noise
            for filepath in root.rglob('*'):
                if filepath.is_file() and filepath.suffix.lower() in ('.py', '.txt', '.md', '.json', '.env'):
                    try:
                        content = filepath.read_text(encoding="utf-8", errors="ignore")
                        for line_num, line in enumerate(content.splitlines(), 1):
                            if pattern in line:
                                rel_path = filepath.relative_to(self.base)
                                results.append(f"File: {rel_path} (Line {line_num}): {line.strip()[:100]}")
                    except Exception:
                        continue
            
            return "\n---\n".join(results) if results else f"Pattern '{pattern}' not found."
        except Exception as e:
            return f"Error during search: {e}"


class SafeScriptExecutor:
    """
    Executes Python scripts in a controlled environment.
    Ensures scripts are run within the workspace boundaries.
    """

    def __init__(self, workspace_dir: str):
        self.base = Path(workspace_dir).resolve()
        self.base.mkdir(parents=True, exist_ok=True)

    def execute(self, script_path: str, input_data: Optional[str] = None, timeout: int = 30) -> ExecutionResult:
        try:
            script = (self.base / script_path).resolve()
            if not str(script).startswith(str(self.base)) or not script.exists():
                return ExecutionResult(False, "", f"Script not found or access denied: {script_path}", 1, "FileError")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            process = subprocess.run(
                ["python", str(script)],
                input=input_data.encode() if input_data else None,
                capture_output=True,
                text=False,
                timeout=timeout,
                cwd=str(self.base),
                env=env
            )

            stdout = process.stdout.decode("utf-8", errors="replace")
            stderr = process.stderr.decode("utf-8", errors="replace")
            
            return ExecutionResult(
                success=(process.returncode == 0),
                stdout=stdout[:10000], # Limit output size to prevent memory overflow
                stderr=stderr[:10000],
                return_code=process.returncode,
                message="Executed"
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(False, "", "Execution timed out", 124, "TimeoutError")
        except Exception as e:
            return ExecutionResult(False, "", str(e), 1, f"ExecutionError: {type(e).__name__}")
