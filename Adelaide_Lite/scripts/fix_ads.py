with open('src/model_manager.ads', 'r') as f:
    lines = f.read().splitlines()

new_lines = []
for line in lines:
    if 'procedure Generate_Speculative' in line:
        continue
    # It might span multiple lines, let's just find the start and skip until `);` or we could just use regex
    new_lines.append(line)

# Wait, Generate_Speculative declaration in model_manager.ads:
#    procedure Generate_Speculative (Kind : Model_Kind;
#                                    Prompt : String;
#                                    ...
#                                    Done : out Boolean);
