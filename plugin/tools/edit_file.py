TOOL_DEFINITION = {"name": "edit_file", "description": "Overwrite file content", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}

def run(context, **kwargs):
    return context.fs_handler.edit_file(kwargs.get("path", ""), kwargs.get("content", ""))
