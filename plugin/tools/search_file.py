TOOL_DEFINITION = {"name": "search_file", "description": "Recursively search for a pattern (string) within all code and text files in a specified directory.", "parameters": {"type": "object", "properties": {"directory_path": {"type": "string"}, "pattern": {"type": "string"}}, "required": ["directory_path", "pattern"]}}

def run(context, **kwargs):
    return context.fs_handler.search_files(kwargs.get("directory_path", ""), kwargs.get("pattern", ""))
