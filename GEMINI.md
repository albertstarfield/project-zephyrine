# Adelaide-Lite Engineering Standards

## Mandates

### 1. Protective Logic (Metal Lock & ELP Queue)
The following mechanisms are **VITAL** for backend stability on macOS and **MUST NOT** be removed or disabled unless explicitly requested by the user:
- **Global Metal Lock:** Serializes GPU access in `model_manager.adb` to prevent `SIGTRAP/SIGABRT`.
- **Descriptive ELP Queue:** Tracks task sources (Indexing, Chat, Speculation) to diagnose "1 pending" issues.
- **Task Stack Expansion:** Stacks for all background tasks MUST remain at **8MB** or higher to prevent recursion overflows.

### 2. Comment Preservation
- Any comment tagged with `[VITAL-DO-NOT-REMOVE]` or `[Debug] DO NOT REMOVE` is **PERMANENT**.
- Agents must NOT "clean up" these comments or the logic they accompany.
- Diagnostic `Put_Line` checkpoints in `Get_Single_Embedding` are intentional and required for long-term health monitoring.

### 3. Build & Style
- Ada code must adhere to GNAT style checks (line length <= 80, 3-space indentation) to ensure successful `alr build`.
- When fixing style, **DO NOT** delete the logic or the protective comments. Use line-splitting to remain compliant.
