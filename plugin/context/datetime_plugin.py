from core import IContextPlugin, ClientConfig
from datetime import datetime

class DateTimeContextPlugin(IContextPlugin):
    """Adds current date and time to the system prompt."""
    def pre_process(self, messages, config):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Inject into the system message
        if messages and messages[0]["role"] == "system":
            original_system = messages[0]["content"]
            messages[0]["content"] = f"{original_system}\n\nCurrent Date and Time: {now}"
        else:
            messages.insert(0, {"role": "system", "content": f"Current Date and Time: {now}"})
        return messages

    def post_process(self, response, messages, config):
        return response
