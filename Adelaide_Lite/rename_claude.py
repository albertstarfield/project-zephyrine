import os

files_to_update = [
    "src/claudealike_helper.ads",
    "src/claudealike_helper.adb",
    "src/adelaide_server_pkg.adb"
]

for file in files_to_update:
    if os.path.exists(file):
        with open(file, "r") as f:
            content = f.read()
        
        # Replace case sensitively
        new_content = content.replace("Claude_Client", "Claudealike_Helper")
        new_content = new_content.replace("claude_client", "claudealike_helper")
        
        with open(file, "w") as f:
            f.write(new_content)
        print(f"Updated {file}")
