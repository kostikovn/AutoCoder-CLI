TOOL_DEFINITION = {"name": "delete_file", "description": "Delete file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}

def run(context, **kwargs):
    return context.fs_handler.delete_file(kwargs.get("path", ""))
