from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import markdownify

TOOL_DEFINITION = {
    "name": "browser",
    "description": "Advanced browser for AI agent to fetch web content, render JavaScript, and interact with pages. Returns content in Markdown format.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string", 
                "description": "The URL of the page to visit"
            },
            "action": {
                "type": "string", 
                "enum": ["get_content", "click", "type", "screenshot"],
                "description": "Action to perform on the page",
                "default": "get_content"
            },
            "selector": {
                "type": "string", 
                "description": "CSS selector for the element to interact with (required for 'click' and 'type')"
            },
            "text": {
                "type": "string", 
                "description": "Text to enter into the field (required for 'type')"
            }
        },
        "required": ["url"]
    }
}

def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Remove noisy elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
    
    # Convert to markdown
    return markdownify.markdownify(str(soup), heading_style="ATX")

def run(context, **kwargs):
    url = kwargs.get("url")
    action = kwargs.get("action", "get_content")
    selector = kwargs.get("selector")
    text = kwargs.get("text")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(url, wait_until="networkidle")
            
            if action == "click":
                if not selector:
                    return "Error: 'selector' is required for 'click' action."
                page.click(selector)
                page.wait_for_load_state("networkidle")
                return f"Successfully clicked {selector}. Page content:\n\n{clean_html(page.content())}"
            
            elif action == "type":
                if not selector or text is None:
                    return "Error: 'selector' and 'text' are required for 'type' action."
                page.fill(selector, text)
                return f"Successfully typed into {selector}. Page content:\n\n{clean_html(page.content())}"
            
            elif action == "screenshot":
                # Saving to a predictable name for the session
                screenshot_path = "browser_last_screenshot.png"
                page.screenshot(path=screenshot_path)
                return f"Screenshot saved to {screenshot_path}"
            
            else: # get_content
                return clean_html(page.content())
                
        except Exception as e:
            return f"Browser error: {str(e)}"
        finally:
            browser.close()
