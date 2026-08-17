TOOL_DEFINITION = {"name": "rename_item", "description": "Rename/move file or directory", "parameters": {"type": "object", "properties": {"old_path": {"type": "string"}, "new_path": {"type": "string"}}, "required": ["old_path", "new_path"]}}

def run(context, **kwargs):
    return context.fs_handler.rename_item(kwargs.get("old_path", ""), kwargs.get("new_path", ""))
