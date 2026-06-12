
import sys
import re

def fix_all(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 1. Surgical fix for Dequeue and Declare blocks
    # Ensure Dequeue call is correct and within begin/end if inside declare
    step1_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == "declare":
            j = i + 1
            decls = []
            found_begin = False
            found_dequeue = False
            while j < len(lines):
                s = lines[j].strip()
                if s == "begin":
                    found_begin = True
                    break
                if "ELP_Queue.Dequeue" in lines[j]:
                    found_dequeue = True
                    break
                if s.startswith("end") or s.startswith("procedure") or s.startswith("function") or s.startswith("task") or s.startswith("package"):
                    break
                decls.append(lines[j])
                j += 1
            
            if found_dequeue and not found_begin:
                step1_lines.append(line) # declare
                step1_lines.extend(decls)
                step1_lines.append("begin\n")
                step1_lines.append("   ELP_Queue.Dequeue_Level (Level);\n")
                step1_lines.append("end;\n")
                i = j + 1
                continue
        
        if "ELP_Queue.Dequeue" in line:
            # Standardize Dequeue call
            line = re.sub(r'ELP_Queue\.Dequeue.*\(.*\);', 'ELP_Queue.Dequeue_Level (Level);', line)
            
        step1_lines.append(line)
        i += 1

    # 2. Re-indentation (3 spaces)
    reindented = []
    level = 0
    
    # Keywords
    dec_before = ['end', 'else', 'elsif', 'exception', 'when']
    inc_after = ['declare', 'begin', 'loop', 'else', 'elsif', 'exception', 'then', 'is', 'do', 'record']
    
    for line in step1_lines:
        stripped = line.strip()
        if not stripped:
            reindented.append("\n")
            continue
            
        lower = stripped.lower()
        clean = re.sub(r'--.*', '', lower).strip()
        
        # Decrease level BEFORE
        if lower.startswith('end ') or lower == 'end;' or lower.startswith('end;') or \
           lower == 'else' or lower.startswith('else ') or \
           lower == 'elsif' or lower.startswith('elsif ') or \
           lower == 'exception' or (lower.startswith('when ') and not 'case ' in lower):
            level -= 1
            
        if level < 0: level = 0
        
        indent = " " * (level * 3)
        reindented.append(indent + stripped + "\n")
        
        # Increase level AFTER
        if clean == 'declare' or clean == 'begin' or clean == 'loop' or \
           clean == 'else' or clean.startswith('elsif') or clean == 'exception' or \
           clean == 'record' or clean.endswith(' record') or \
           ((clean.endswith(' is') or clean == 'is') and not clean.endswith(';')) or \
           ((clean.endswith(' then') or clean == 'then') and not clean.endswith(';')) or \
           ((clean.endswith(' do') or clean == 'do') and not clean.endswith(';')) or \
           (clean.startswith('case ') and clean.endswith(' is')):
            level += 1

    # 3. Line length wrapping (79 chars to be safe)
    final_lines = []
    for line in reindented:
        if len(line.rstrip()) > 79:
            stripped = line.strip()
            indent = line[:len(line) - len(stripped) - 1] # -1 for \n
            if stripped.startswith('--'):
                # Wrap comment
                prefix = indent + "-- "
                content = stripped[2:].strip()
                words = content.split()
                curr = prefix
                for w in words:
                    if len(curr + w) <= 79:
                        curr += w + " "
                    else:
                        final_lines.append(curr.rstrip() + "\n")
                        curr = prefix + w + " "
                final_lines.append(curr.rstrip() + "\n")
            else:
                # Wrap code
                split_points = [', ', ' and ', ' or ', ' => ', ' & ', ' := ']
                found = False
                for sp in split_points:
                    pos = line.rfind(sp, 0, 79)
                    if pos != -1:
                        final_lines.append(line[:pos + len(sp)].rstrip() + "\n")
                        final_lines.append(indent + "   " + line[pos + len(sp):].lstrip())
                        found = True
                        break
                if not found:
                    final_lines.append(line)
        else:
            final_lines.append(line)

    with open(file_path, 'w') as f:
        f.writelines(final_lines)

if __name__ == "__main__":
    fix_all('Adelaide_Lite/src/model_manager.adb')
