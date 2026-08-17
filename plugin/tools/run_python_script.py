TOOL_DEFINITION = {"name": "run_python_script", "description": "Run a Python script with optional console input. Returns stdout, stderr, and exit code.", "parameters": {"type": "object", "properties": {"script_path": {"type": "string"}, "input_data": {"type": "string", "description": "Optional console input to feed to stdin"}}, "required": ["script_path"]}}

def run(context, **kwargs):
    return context.executor.execute(kwargs.get("script_path", ""), input_data=kwargs.get("input_data", ""))
