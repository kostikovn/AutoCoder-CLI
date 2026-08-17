TOOL_DEFINITION = {"name": "list_directory", "description": "List directory contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}

def run(context, **kwargs):
    return context.fs_handler.list_directory(kwargs.get("path", ""))
