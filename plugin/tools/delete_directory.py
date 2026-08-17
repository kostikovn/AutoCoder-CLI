TOOL_DEFINITION = {"name": "delete_directory", "description": "Delete directory recursively", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}

def run(context, **kwargs):
    return context.fs_handler.delete_directory(kwargs.get("path", ""))
