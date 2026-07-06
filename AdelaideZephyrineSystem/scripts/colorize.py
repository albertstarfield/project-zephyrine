import os
import re

PREFIX_COLORS = {
    "[Main]": "AnsiAda.Foreground (AnsiAda.Light_Blue)",
    "[Server]": "AnsiAda.Foreground (AnsiAda.Green)",
    "[Async]": "AnsiAda.Foreground (AnsiAda.Yellow)",
    "[DB]": "AnsiAda.Foreground (AnsiAda.Magenta)",
    "[Knowledge]": "AnsiAda.Foreground (AnsiAda.Cyan)",
    "[WCET]": "AnsiAda.Foreground (AnsiAda.Light_Red)",
    "[Hybrid]": "AnsiAda.Foreground (AnsiAda.Light_Magenta)",
    "[Model]": "AnsiAda.Foreground (AnsiAda.Light_Cyan)",
    "[Thought]": "AnsiAda.Foreground (AnsiAda.Light_Yellow)",
    "[Idle]": "AnsiAda.Foreground (AnsiAda.Grey)",
    "[FATAL]": "AnsiAda.Foreground (AnsiAda.Red)",
    "[Log]": "AnsiAda.Foreground (AnsiAda.Light_Grey)",
}

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Need to add "with AnsiAda;" if we do replacements
    has_changes = False
    
    # We want to replace Put_Line ("[Prefix] ...") with Put_Line (AnsiAda.Foreground (Color) & "[Prefix]" & AnsiAda.Reset & " ...")
    # There are variations like Put_Line ("[Prefix] " & ...)
    for prefix, color in PREFIX_COLORS.items():
        # Match `Put_Line ("[Prefix] `
        # Or `Put_Line ("[Prefix]"`
        pattern = r'(Put_Line\s*\(\s*")(' + re.escape(prefix) + r')(\s*[^"]*")'
        
        def repl(m):
            nonlocal has_changes
            has_changes = True
            # m.group(1) is `Put_Line ("`
            # m.group(2) is `[Prefix]`
            # m.group(3) is ` text"`
            # Wait, `Put_Line ("` includes the quote.
            # Let's just do it simpler:
            # Replace `"[Prefix]` with `Color & "[Prefix]" & AnsiAda.Reset & "` if it's inside Put_Line.
            return 'Put_Line (' + color + ' & "' + m.group(2) + '" & AnsiAda.Reset & "' + m.group(3)

        content = re.sub(pattern, repl, content)
        
        pattern2 = r'(Put_Line\s*\(\s*")(' + re.escape(prefix) + r')("\s*&)'
        def repl2(m):
            nonlocal has_changes
            has_changes = True
            return 'Put_Line (' + color + ' & "' + m.group(2) + '" & AnsiAda.Reset & ' + m.group(3)
        content = re.sub(pattern2, repl2, content)

    if has_changes:
        if "with AnsiAda;" not in content:
            # find first with
            content = re.sub(r'^(with\s+[A-Za-z0-9_.]+;)', r'with AnsiAda;\n\1', content, count=1, flags=re.MULTILINE)
        
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated {filepath}")

for root, _, files in os.walk("src"):
    for file in files:
        if file.endswith(".adb") or file.endswith(".ads"):
            process_file(os.path.join(root, file))
