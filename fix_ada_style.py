import re
import sys

def fix_style(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # 3. TRAILING SPACES: Remove all trailing spaces
        line = line.rstrip() + '\n'
        new_lines.append(line)

    # 1. LINE LENGTH: Split long lines (<= 80 chars)
    # This is a bit complex, let's do it carefully for comments first
    processed_lines = []
    for line in new_lines:
        if len(line.rstrip()) > 80:
            if line.strip().startswith('--'):
                # Split comment
                indent = line[:line.find('--')]
                comment_content = line.strip()[2:].strip()
                words = comment_content.split(' ')
                current_line = indent + '--'
                for word in words:
                    if len(current_line + ' ' + word) <= 80:
                        current_line += ' ' + word
                    else:
                        processed_lines.append(current_line + '\n')
                        current_line = indent + '--  ' + word
                processed_lines.append(current_line + '\n')
            else:
                # Code line - try to split on comma or operator
                # For now, let's keep it and I'll fix manually or with better logic
                processed_lines.append(line)
        else:
            processed_lines.append(line)

    with open(file_path, 'w') as f:
        f.writelines(processed_lines)

if __name__ == "__main__":
    fix_style('Adelaide_Lite/src/model_manager.adb')
