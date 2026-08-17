TOOL_DEFINITION = {"name": "create_file", "description": "Create file with content", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}

def run(context, **kwargs):
    return context.fs_handler.create_file(kwargs.get("path", ""), kwargs.get("content", ""))
