#!/usr/bin/env python3
"""
Todo Tool - Task management for Adelaide Lite.

Usage: python3 todo.py <command> [args...]

Commands:
  add <task>              - Add a new task
  list                    - List all tasks
  done <id>               - Mark task as done
  remove <id>             - Remove a task
  clear                   - Clear all done tasks
  search <query>          - Search tasks

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import sys
import os
import json
from datetime import datetime


TODO_FILE = os.path.join(os.path.dirname(__file__), ".todos.json")


def load_todos():
    """Load todos from file."""
    if os.path.exists(TODO_FILE):
        with open(TODO_FILE, "r") as f:
            return json.load(f)
    return {"tasks": [], "next_id": 1}


def save_todos(todos):
    """Save todos to file."""
    with open(TODO_FILE, "w") as f:
        json.dump(todos, f, indent=2)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    args = sys.argv[2:]

    todos = load_todos()

    if cmd == "add":
        if not args:
            print("ERROR: Usage: todo.py add <task>")
            return 1
        task = " ".join(args)
        new_task = {
            "id": todos["next_id"],
            "task": task,
            "done": False,
            "created": datetime.now().isoformat()
        }
        todos["tasks"].append(new_task)
        todos["next_id"] += 1
        save_todos(todos)
        print(f"OK: Added task #{new_task['id']}: {task}")

    elif cmd == "list":
        if not todos["tasks"]:
            print("No tasks")
        else:
            for task in todos["tasks"]:
                status = "[x]" if task["done"] else "[ ]"
                print(f"{task['id']}. {status} {task['task']}")

    elif cmd == "done":
        if not args:
            print("ERROR: Usage: todo.py done <id>")
            return 1
        task_id = int(args[0])
        for task in todos["tasks"]:
            if task["id"] == task_id:
                task["done"] = True
                save_todos(todos)
                print(f"OK: Marked task #{task_id} as done")
                return 0
        print(f"ERROR: Task #{task_id} not found")

    elif cmd == "remove":
        if not args:
            print("ERROR: Usage: todo.py remove <id>")
            return 1
        task_id = int(args[0])
        for i, task in enumerate(todos["tasks"]):
            if task["id"] == task_id:
                todos["tasks"].pop(i)
                save_todos(todos)
                print(f"OK: Removed task #{task_id}")
                return 0
        print(f"ERROR: Task #{task_id} not found")

    elif cmd == "clear":
        todos["tasks"] = [t for t in todos["tasks"] if not t["done"]]
        save_todos(todos)
        print("OK: Cleared done tasks")

    elif cmd == "search":
        if not args:
            print("ERROR: Usage: todo.py search <query>")
            return 1
        query = " ".join(args).lower()
        found = [t for t in todos["tasks"] if query in t["task"].lower()]
        if found:
            for task in found:
                status = "[x]" if task["done"] else "[ ]"
                print(f"{task['id']}. {status} {task['task']}")
        else:
            print("No matching tasks")

    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
