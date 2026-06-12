
import sys
import re

def surgical_fix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Match "declare" followed by declarations and then a Dequeue call without "begin"
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
                if s == "end;" or s.startswith("procedure") or s.startswith("function") or s.startswith("task") or s.startswith("package"):
                    break
                decls.append(lines[j])
                j += 1
            
            if found_dequeue and not found_begin:
                # We found a declare block without begin that has a Dequeue call
                indent = line[:line.find("declare")]
                new_lines.append(line) # declare
                for d in decls:
                    new_lines.append(d)
                
                new_lines.append(indent + "   begin\n")
                new_lines.append(indent + "      ELP_Queue.Dequeue_Level (Level);\n")
                new_lines.append(indent + "   end;\n")
                
                # Move i past the Dequeue line
                i = j + 1
                continue
        
        # Replace any other Dequeue calls to the correct format
        if "ELP_Queue.Dequeue" in line and "ELP_Queue.Dequeue_Level (Level);" not in line:
            # Preserve indentation
            indent = line[:line.find("ELP_Queue.Dequeue")]
            line = indent + "ELP_Queue.Dequeue_Level (Level);\n"
            
        new_lines.append(line)
        i += 1

    with open(file_path, 'w') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    surgical_fix('Adelaide_Lite/src/model_manager.adb')
