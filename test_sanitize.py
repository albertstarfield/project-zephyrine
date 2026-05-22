import sys
import re
sys.path.append('Adelaide_Lite/python')
from ollamaCallModifier import sanitize_think_tags

print("content=True :", repr(sanitize_think_tags('<think>hello</think> test', remove_content=True)))
print("content=False:", repr(sanitize_think_tags('<think>hello</think> test', remove_content=False)))
