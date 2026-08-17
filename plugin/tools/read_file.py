TOOL_DEFINITION = {"name": "read_file", "description": "Read file content", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}

def run(context, **kwargs):
    return context.fs_handler.read_file(kwargs.get("path", ""))
