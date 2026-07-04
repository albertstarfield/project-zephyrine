#!/usr/bin/env python3
"""
Hook System Tool - Pre/post tool execution hooks for Adelaide Lite.

Usage: python3 hook.py <command> [args...]

Commands:
  list                  - List available hooks
  add <event> <script>  - Add a hook
  remove <event> <id>   - Remove a hook
  run <event> [data]    - Run hooks for event

Events:
  pre-tool              - Before tool execution
  post-tool             - After tool execution
  pre-commit            - Before git commit
  post-commit           - After git commit
  pre-build             - Before build
  post-build            - After build

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import sys
import os
import json
import subprocess
from datetime import datetime
from trace_utils import init_trace, trace_print, trace_result


HOOKS_FILE = os.path.join(os.path.dirname(__file__), ".hooks.json")


def load_hooks():
    """Load hooks from file."""
    if os.path.exists(HOOKS_FILE):
        with open(HOOKS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_hooks(hooks):
    """Save hooks to file."""
    with open(HOOKS_FILE, "w") as f:
        json.dump(hooks, f, indent=2)


def run_hooks(event, data=None):
    """Run all hooks for an event."""
    hooks = load_hooks()
    if event not in hooks:
        return True
    
    for hook in hooks[event]:
        script = hook.get("script")
        if not script or not os.path.exists(script):
            print(f"WARNING: Hook script not found: {script}")
            continue
        
        trace_print("hook", "run", str(hook.get("name", script)))
        try:
            env = os.environ.copy()
            if data:
                env["HOOK_DATA"] = json.dumps(data)
            
            result = subprocess.run(
                ["python3", script],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            
            if result.returncode != 0:
                trace_print("hook", "error", f"Hook failed: {result.stderr}")
                return False
            
            if result.stdout:
                print(result.stdout)
        except subprocess.TimeoutExpired:
            print(f"Hook timed out: {script}")
        except Exception as e:
            trace_print("hook", "error", f"{e}")
    
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    args = sys.argv[2:]

    init_trace()

    if cmd == "list":
        hooks = load_hooks()
        if not hooks:
            print("No hooks configured")
        else:
            for event, hook_list in hooks.items():
                print(f"\n{event}:")
                for hook in hook_list:
                    print(f"  - {hook.get('name', hook.get('script'))}")

    elif cmd == "add":
        if len(args) < 2:
            print("ERROR: Usage: hook.py add <event> <script>")
            return 1
        event = args[0]
        script = args[1]
        
        hooks = load_hooks()
        if event not in hooks:
            hooks[event] = []
        
        hooks[event].append({
            "name": os.path.basename(script),
            "script": script,
            "added": datetime.now().isoformat()
        })
        
        save_hooks(hooks)
        trace_print("hook", "add", event)

    elif cmd == "remove":
        if len(args) < 2:
            print("ERROR: Usage: hook.py remove <event> <id>")
            return 1
        event = args[0]
        hook_id = int(args[1])
        
        hooks = load_hooks()
        if event in hooks and len(hooks[event]) > hook_id:
            hooks[event].pop(hook_id)
            save_hooks(hooks)
            trace_print("hook", "remove", event)
        else:
            print("ERROR: Hook not found")

    elif cmd == "run":
        if not args:
            print("ERROR: Usage: hook.py run <event> [data]")
            return 1
        event = args[0]
        data = args[1] if len(args) > 1 else None
        success = run_hooks(event, data)
        return 0 if success else 1

    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
