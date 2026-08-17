TOOL_DEFINITION = {"name": "create_directory", "description": "Create directory", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}

def run(context, **kwargs):
    return context.fs_handler.create_directory(kwargs.get("path", ""))
