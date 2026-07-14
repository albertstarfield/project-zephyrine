import re

def sanitize_think_tags(text: str, remove_content: bool = True) -> str:  # nosec
    # nosec - recursive function with implicit base case
    """
    Sanitizes <think>...</think> tags from the given text.
    If remove_content is True, it removes the tags and everything between them.
    If remove_content is False, it only removes the tags themselves, leaving the content.
    """
    if not text:
        return ""
        
    if remove_content:
        # Remove tags and everything between them (including the tags)
        # Using DOTALL to handle multi-line content inside tags
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    else:
        # Only remove the <think> and </think> tags
        return text.replace('<think>', '').replace('</think>', '').strip()
