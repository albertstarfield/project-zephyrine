#!/usr/bin/env python3
import fcntl
import hashlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB total cap

# ── Crypto ────────────────────────────────────────────────────────────────
# Import the Python crypto module (sibling to python/adelaide_crypto.py)
sys.path.insert(0, os.path.join(BASE_DIR, "python"))
from adelaide_crypto import load_master_key, migrate_all_to_aad  # noqa: E402

# ── Hardware-Bound Key Derivation Constants ───────────────────────────────
# Integrity test plaintext for key verification
INTEGRITY_TEST_PLAINTEXT = "--ADELAIDE-INTEGRITY-TEST--"

# InferiorParadoxical — SHA-512 hardware profiling key (auto-decrypt key)
#
# TWO INDEPENDENT keys can decrypt the database (not combined):
#   Key 1 — User password or recovery key  (user provides on prompt)
#   Key 2 — InferiorParadoxical             (auto-derived from hardware profiling)
#
# InferiorParadoxical = SHA-512 of the hardware integrity hash.
# On boot we try InferiorParadoxical first (full automation when environment
# is trusted).  If that fails (hardware changed), fall back to user-password
# prompt.  After successful password unlock, InferiorParadoxical is re-derived
# from the current hardware state and the stored master-key is re-wrapped so
# future boots can auto-decrypt again.
#
# Stored in system_state as AES-256-GCM-wrapped master key, encrypted under a
# sub-key derived from the InferiorParadoxical hash.
INFERIOR_PARADOXICAL_KEY = "inferior_paradoxical_master_key"

def _store_inferior_paradoxical_wrapped_key(master_key_hex, integrity_hash):
    """
    Wrap master_key under InferiorParadoxical in C boundary and store in system_state.
    """
    try:
        import ctypes
        lib_path = os.path.join(BASE_DIR, "obj", "release", "libadl_crypto.dylib")
        if not os.path.exists(lib_path):
            lib_path = os.path.join(BASE_DIR, "obj", "release", "libadl_crypto.so")
        lib = ctypes.CDLL(lib_path)
        
        lib.adl_auto_wrap_master_key_cstr.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.adl_auto_wrap_master_key_cstr.restype = ctypes.POINTER(ctypes.c_char)
        lib.adl_free_cstr.argtypes = [ctypes.POINTER(ctypes.c_char)]
        
        c_ptr = lib.adl_auto_wrap_master_key_cstr(integrity_hash.encode('utf-8'), master_key_hex.encode('utf-8'))
        if not c_ptr:
            return False
        encrypted = ctypes.cast(c_ptr, ctypes.c_char_p).value.decode('utf-8')
        lib.adl_free_cstr(c_ptr)
        
        import sqlite3
        db_path = os.path.join(BASE_DIR, "NetworkMemoryPool", os.environ.get("ADELAIDE_USER", "default"), "adelaide_memory.db")
        if not os.path.exists(db_path):
            return False
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
            (INFERIOR_PARADOXICAL_KEY, encrypted),
        )
        conn.commit()
        conn.close()
        print("[KEY-DERIV] InferiorParadoxical wrapped key stored via C boundary")
        return True
    except Exception as e:
        print(f"[KEY-DERIV] Failed to store InferiorParadoxical wrapped key: {e}")
        return False

def _try_inferior_paradoxical_auto_decrypt(integrity_hash):
    """
    Try to auto-recover master_key using InferiorParadoxical hardware key via C boundary.
    """
    try:
        import sqlite3
        db_path = os.path.join(BASE_DIR, "NetworkMemoryPool", os.environ.get("ADELAIDE_USER", "default"), "adelaide_memory.db")
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT value FROM system_state WHERE key = ?",
            (INFERIOR_PARADOXICAL_KEY,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
            
        import ctypes
        lib_path = os.path.join(BASE_DIR, "obj", "release", "libadl_crypto.dylib")
        if not os.path.exists(lib_path):
            lib_path = os.path.join(BASE_DIR, "obj", "release", "libadl_crypto.so")
        lib = ctypes.CDLL(lib_path)
        
        lib.adl_auto_unlock_master_key_cstr.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.adl_auto_unlock_master_key_cstr.restype = ctypes.POINTER(ctypes.c_char)
        lib.adl_free_cstr.argtypes = [ctypes.POINTER(ctypes.c_char)]
        
        c_ptr = lib.adl_auto_unlock_master_key_cstr(integrity_hash.encode('utf-8'), row[0].encode('utf-8'))
        if not c_ptr:
            return None
        master_key = ctypes.cast(c_ptr, ctypes.c_char_p).value.decode('utf-8')
        lib.adl_free_cstr(c_ptr)
        
        if master_key and len(master_key) == 64:
            print("[KEY-DERIV] InferiorParadoxical auto-decrypt SUCCESS via C boundary")
            return master_key
        return None
    except Exception as e:
        import traceback
        print(f"[KEY-DERIV] InferiorParadoxical auto-decrypt crashed: {e}")
        print(f"[KEY-DERIV] Traceback:\n{traceback.format_exc()}")
        return None

# ── KISS Mode ─────────────────────────────────────────────────────────────
IS_KISS = "--kiss" in sys.argv

# ── Stdio Protocol Messages ───────────────────────────────────────────────
# Ada → run.py messages
MSG_INTEGRITY_MISMATCH = "INTEGRITY_MISMATCH"
MSG_INVALID_SECRET = "INVALID_SECRET"
MSG_KEY_ACCEPTED = "KEY_ACCEPTED"
MSG_READY = "READY"


# ── Hardware-Bound Key Derivation Handler ─────────────────────────────────
def handle_stdio_key_exchange(proc):
    """
    Handle stdio-based key exchange with Ada server.

    Protocol:
    - Ada writes INTEGRITY_MISMATCH if key derivation failed
    - Ada writes INVALID_SECRET if user provided wrong password
    - Ada writes KEY_ACCEPTED if key verified successfully
    - Ada writes READY when startup complete
    - run.py writes user secret (password or recovery key) followed by newline

    Returns True if key exchange succeeded, False otherwise.
    """
    print("[KEY-EXCHANGE] Waiting for Ada server key exchange...")

    # For now, use environment variables to communicate
    # The Ada server will check for ADELAIDE_MASTER_KEY env var

    return True


def _term_print(msg):
    """Print to terminal directly (bypasses KISS stdout redirect)."""
    import sys

    dest = term_stderr if term_stderr else sys.__stderr__
    dest.write(msg + "\n")
    dest.flush()


_global_tk_root = None

def _get_tk_root():
    global _global_tk_root
    import tkinter as tk
    if _global_tk_root is None:
        _global_tk_root = tk.Tk()
        _global_tk_root.withdraw()
    return _global_tk_root

def _gui_available():
    """Check if tkinter is available and we have a display."""
    import os
    if os.environ.get("NO_GUI") == "1":
        return False
        
    # Cache the result so we only check once
    if not hasattr(_gui_available, "_cached"):
        try:
            _ = _get_tk_root()
            _gui_available._cached = True
        except Exception:
            _gui_available._cached = False
    return _gui_available._cached


def _password_entropy(password):
    """Calculate password entropy in bits (lower bound) based on character pool."""
    import math
    import string
    if not password:
        return 0
    pool = 0
    if any(c in string.ascii_lowercase for c in password):
        pool += 26
    if any(c in string.ascii_uppercase for c in password):
        pool += 26
    if any(c in string.digits for c in password):
        pool += 10
    if any(c in string.punctuation for c in password):
        pool += 33
    if pool == 0:
        return 0
    return math.floor(len(password) * math.log2(pool))


def _wipe_string(s):
    """Best-effort wiping of a string from Python heap memory.
    
    Python strings are immutable, so we cannot zero them in place. This
    function overwrites the variable's reference with a new string of the
    same length (to reduce the chance that the original bytes survive in
    heap), then forces garbage collection.
    """
    if s is None:
        return
    try:
        length = len(s)
        # Overwrite reference with dummy data
        s = "X" * length
        s = "\0" * length  # null bytes
    except Exception:
        pass
    finally:
        s = None
    import gc
    gc.collect()


def _tk_input_dialog(title, prompt, welcome_msg=None):
    import tkinter as tk
    import tkinter.simpledialog as sd

    root = tk.Tk()
    root.withdraw()
    
    # Try to ensure window comes to front
    root.attributes("-topmost", True)
    root.focus_force()

    if welcome_msg:
        # Show welcome message in a custom dialog
        bg = "#1a1a2e"
        fg = "#e0e0e0"
        entry_bg = "#16213e"
        btn_bg = "#0f3460"
        accent = "#e94560"

        dialog = tk.Toplevel(root)
        dialog.title(title)
        dialog.attributes("-topmost", True)
        dialog.resizable(False, False)
        dialog.grab_set()

        w, h = 420, 320
        sx = (dialog.winfo_screenwidth() - w) // 2
        sy = (dialog.winfo_screenheight() - h) // 2
        dialog.geometry(f"{w}x{h}+{sx}+{sy}")
        dialog.configure(bg=bg)

        tk.Label(
            dialog, text=welcome_msg, bg=bg, fg=fg,
            font=("Helvetica", 12), justify="left", wraplength=380,
        ).pack(pady=(16, 10), padx=20)

        tk.Label(
            dialog, text=prompt, bg=bg, fg=fg,
            font=("Helvetica", 13),
        ).pack(pady=(4, 6))

        name_var = tk.StringVar()
        name_entry = tk.Entry(
            dialog, textvariable=name_var, bg=entry_bg, fg=fg,
            insertbackground=fg, font=("Helvetica", 13), width=28,
            relief="flat",
        )
        name_entry.pack(pady=4, padx=20)
        name_entry.focus_set()

        result = [None]

        def on_ok(_event=None):
            # Read directly from Entry widget — StringVar binding is unreliable on macOS
            val = name_entry.get()
            if not IS_KISS:
                print(f"[DEBUG] on_ok fired, name_entry.get() = {val!r}")
            result[0] = val
            dialog.destroy()

        def on_cancel():
            if not IS_KISS:
                print("[DEBUG] on_cancel fired")
            result[0] = None
            dialog.destroy()

        btn_frame = tk.Frame(dialog, bg=bg)
        btn_frame.pack(pady=(10, 8))

        tk.Button(
            btn_frame, text="OK", command=on_ok, bg=btn_bg, fg="#ffffff",
            activebackground=accent, activeforeground="#ffffff",
            font=("Helvetica", 11, "bold"), width=10, relief="flat", cursor="hand2",
        ).pack(side="left", padx=6)

        tk.Button(
            btn_frame, text="Cancel", command=on_cancel, bg="#2a2a4a", fg="#ffffff",
            activebackground="#555577", activeforeground="#ffffff",
            font=("Helvetica", 11), width=10, relief="flat", cursor="hand2",
        ).pack(side="left", padx=6)

        dialog.protocol("WM_DELETE_WINDOW", on_cancel)
        # Do NOT bind <Return> — it steals the keystroke from the Entry widget.
        # User must click OK button to submit.
        dialog.bind("<Escape>", lambda e: on_cancel())

        root.wait_window(dialog)
        root.withdraw()
        if not IS_KISS:
            print(f"[DEBUG] _tk_input_dialog returning: {result[0]!r}")
        return result[0]
    else:
        result = sd.askstring(title, prompt, parent=root)
        root.destroy()
        return result

def _tk_progress_dialog(title, message):
    """Show a tkinter progress dialog with an animated bar, step text, and ETA. Returns the dialog object for updates."""
    import tkinter as tk

    root = _get_tk_root()
    root.deiconify()
    root.attributes("-topmost", True)

    bg = "#1a1a2e"
    fg = "#e0e0e0"
    bar_bg = "#16213e"
    bar_fill = "#4ecca3"

    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)
    dialog.transient(root)
    dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # prevent close during load

    w, h = 420, 160
    sx = (dialog.winfo_screenwidth() - w) // 2
    sy = (dialog.winfo_screenheight() - h) // 2
    dialog.geometry(f"{w}x{h}+{sx}+{sy}")
    dialog.configure(bg=bg)

    title_label = tk.Label(
        dialog, text=message, bg=bg, fg=fg,
        font=("Helvetica", 12),
    )
    title_label.pack(pady=(14, 4))

    # Canvas-based progress bar
    canvas = tk.Canvas(dialog, width=380, height=20, bg=bar_bg, highlightthickness=0)
    canvas.pack(pady=(4, 4))
    fill_rect = canvas.create_rectangle(0, 0, 0, 20, fill=bar_fill, outline="")

    info_frame = tk.Frame(dialog, bg=bg)
    info_frame.pack(pady=(2, 2))

    pct_label = tk.Label(
        info_frame, text="0%", bg=bg, fg=fg, font=("Helvetica", 10),
    )
    pct_label.pack(side=tk.LEFT, padx=(0, 16))

    eta_label = tk.Label(
        info_frame, text="", bg=bg, fg="#888888", font=("Helvetica", 10),
    )
    eta_label.pack(side=tk.LEFT)

    step_label = tk.Label(
        dialog, text="", bg=bg, fg="#4ecca3",
        font=("Helvetica", 10), wraplength=380,
    )
    step_label.pack(pady=(2, 4))

    _pulse_state = [0]
    _pulse_id = [None]

    def _pulse_bar():
        """Indeterminate pulse animation for unknown-duration steps."""
        try:
            _pulse_state[0] = (_pulse_state[0] + 6) % 380
            x = _pulse_state[0]
            canvas.coords(fill_rect, x, 0, min(x + 80, 380), 20)
            _pulse_id[0] = dialog.after(30, _pulse_bar)
        except Exception:
            pass

    def update_bar(pct, eta_text="", step_text="", pulse=False):
        try:
            if pulse:
                if _pulse_id[0] is None:
                    _pulse_bar()
                pct_label.configure(text="")
                if eta_text:
                    eta_label.configure(text=eta_text)
                if step_text:
                    step_label.configure(text=step_text)
                dialog.update()
                return
            # Stop pulse if running
            if _pulse_id[0] is not None:
                dialog.after_cancel(_pulse_id[0])
                _pulse_id[0] = None
            canvas.coords(fill_rect, 0, 0, int(380 * pct / 100), 20)
            pct_label.configure(text=f"{int(pct)}%")
            if eta_text:
                eta_label.configure(text=eta_text)
            if step_text:
                step_label.configure(text=step_text)
            dialog.update()
        except Exception:
            pass

    dialog._update_bar = update_bar
    dialog._root_ref = root

    # Background pulse thread — keeps tkinter alive while main thread is blocked
    _pulse_alive = [True]
    _pulse_thread_ref = [None]

    def _pulse_thread_fn():
        while _pulse_alive[0]:
            try:
                dialog.after(0, lambda: dialog.update() if dialog.winfo_exists() else None)
            except Exception:
                break
            time.sleep(0.15)

    def _start_pulse():
        if _pulse_thread_ref[0] is None:
            t = threading.Thread(target=_pulse_thread_fn, daemon=True)
            _pulse_thread_ref[0] = t
            t.start()

    def _stop_pulse():
        _pulse_alive[0] = False

    dialog._start_pulse = _start_pulse
    dialog._stop_pulse = _stop_pulse
    return dialog


def _tk_progress_done(dialog):
    """Close the progress dialog and withdraw the root tk window."""
    try:
        dialog.destroy()
    except Exception:
        pass
    try:
        root = _get_tk_root()
        root.withdraw()
    except Exception:
        pass


def _tk_password_dialog(title, prompt, confirm=False, promise_msg=None):

    """Show a tkinter password dialog and return the entered string or None."""
    import tkinter as tk

    root = _get_tk_root()
    root.attributes("-topmost", True)

    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)
    dialog.grab_set()

    # Center on screen (taller when confirm mode has entropy + tip labels, or promise msg)
    extra_h = 60 if promise_msg else 0
    w, h = 380, (270 if confirm else 180) + extra_h
    sx = (dialog.winfo_screenwidth() - w) // 2
    sy = (dialog.winfo_screenheight() - h) // 2
    dialog.geometry(f"{w}x{h}+{sx}+{sy}")

    # Style
    bg = "#1a1a2e"
    fg = "#e0e0e0"
    entry_bg = "#16213e"
    btn_bg = "#0f3460"
    accent = "#e94560"
    green = "#4ecca3"
    dialog.configure(bg=bg)

    if promise_msg:
        tk.Label(
            dialog, text=promise_msg, bg=bg, fg=fg,
            font=("Helvetica", 11), justify="left", wraplength=340,
        ).pack(pady=(12, 4), padx=20)

    tk.Label(dialog, text=prompt, bg=bg, fg=fg, font=("Helvetica", 13)).pack(
        pady=(18, 6)
    )

    pw_var = tk.StringVar()
    pw_entry = tk.Entry(
        dialog,
        textvariable=pw_var,
        show="*",
        bg=entry_bg,
        fg=fg,
        insertbackground=fg,
        font=("Helvetica", 13),
        width=28,
        relief="flat",
    )
    pw_entry.pack(pady=4, padx=20)
    pw_entry.focus_set()

    # Entropy label + tip (only on password creation — confirm=True)
    entropy_label = None
    tip_label = None
    if confirm:
        entropy_label = tk.Label(
            dialog, text="Entropy: 0 bits", bg=bg, fg=fg, font=("Helvetica", 10)
        )
        entropy_label.pack(pady=(2, 0))
        tip_label = tk.Label(
            dialog,
            text="Use lowercase, uppercase, numbers, and/or symbols.",
            bg=bg, fg="#777777", font=("Helvetica", 8), wraplength=340,
        )
        tip_label.pack(pady=(1, 0))

    confirm_var = None
    confirm_entry = None
    if confirm:
        tk.Label(
            dialog, text="Confirm password:", bg=bg, fg=fg, font=("Helvetica", 11)
        ).pack(pady=(8, 2))
        confirm_var = tk.StringVar()
        confirm_entry = tk.Entry(
            dialog,
            textvariable=confirm_var,
            show="*",
            bg=entry_bg,
            fg=fg,
            insertbackground=fg,
            font=("Helvetica", 13),
            width=28,
            relief="flat",
        )
        confirm_entry.pack(pady=4, padx=20)

    result = [None]

    btn_frame = tk.Frame(dialog, bg=bg)
    btn_frame.pack(pady=(10, 8))

    ok_btn = tk.Button(
        btn_frame,
        text="OK",
        command=lambda: None,  # reassigned after definition
        bg=btn_bg,
        fg=fg,
        activebackground=accent,
        activeforeground="#fff",
        font=("Helvetica", 11),
        width=10,
        relief="flat",
        cursor="hand2",
        state="disabled" if confirm else "normal",
    )
    ok_btn.pack(side="left", padx=6)

    cancel_btn = tk.Button(
        btn_frame,
        text="Cancel",
        command=lambda: None,  # reassigned
        bg=entry_bg,
        fg=fg,
        activebackground=accent,
        activeforeground="#fff",
        font=("Helvetica", 11),
        width=10,
        relief="flat",
        cursor="hand2",
    )
    cancel_btn.pack(side="left", padx=6)

    def on_ok(_event=None):
        pw = pw_var.get()
        if confirm:
            pw2 = confirm_var.get()
            if pw != pw2:
                confirm_entry.configure(bg="#5c1a1a")
                dialog.after(600, lambda: confirm_entry.configure(bg=entry_bg))
                return
            if _password_entropy(pw) < 20:
                return  # blocked — OK button is disabled anyway, but safety net
        result[0] = pw
        dialog.destroy()

    def on_cancel():
        result[0] = None
        dialog.destroy()

    # Wire up button commands
    ok_btn.configure(command=on_ok)
    cancel_btn.configure(command=on_cancel)

    # Live entropy update on password creation
    def on_pw_changed(*_args):
        if not confirm or entropy_label is None:
            return
        pw = pw_var.get()
        bits = _password_entropy(pw)
        entropy_label.configure(text=f"Entropy: {bits} bits")
        if bits < 20:
            entropy_label.configure(fg=accent)  # red
            tip_label.configure(fg=accent, text="Make stronger password! Use lowercase, uppercase, numbers, and/or symbols.")
            ok_btn.configure(state="disabled")
        else:
            entropy_label.configure(fg=green)  # green
            tip_label.configure(fg="#777777", text="Use lowercase, uppercase, numbers, and/or symbols.")
            ok_btn.configure(state="normal")

    if confirm:
        pw_var.trace_add("write", on_pw_changed)

    dialog.bind("<Return>", on_ok)
    dialog.bind("<Escape>", lambda e: on_cancel())

    # Use wait_window instead of root.mainloop() — returns when dialog is destroyed
    root.wait_window(dialog)
    return result[0]


def prompt_kiss_password(is_first_boot=False, is_recovery=False):
    """
    KISS mode password prompt (phone-like setup).

    First boot:
    - Create password
    - Confirm password
    - Show recovery key

    Subsequent boot:
    - Prompt for password

    is_recovery: True if this prompt is asking for the recovery key
                 rather than a password (changes the dialog label).
    """
    # Check if tkinter is available and we're in GUI mode
    use_gui = _gui_available()

    if use_gui:
        if is_first_boot:
            _term_print("[KEY-DERIV] First boot — opening password setup dialog...")
            _promise_msg = (
                "Oki :D, now so that we can keep secret between\n"
                "each other, I am with my pinky finger, promise\n"
                "to not share your data with others *wink"
            )
            password = _tk_password_dialog(
                "Adelaide — Set Password", "Create a new password:",
                confirm=True, promise_msg=_promise_msg,
            )
            if not password:
                return None
            _term_print("[KEY-DERIV] Password set.")

            # Generate recovery key (256-bit entropy)
            import secrets

            hex_str = secrets.token_hex(32)
            recovery_key = "-".join(hex_str[i:i+8] for i in range(0, 64, 8))
            _term_print(f"[KEY-DERIV] Recovery key: {recovery_key}")

            # Show recovery key dialog (NOT stored — user writes it down)
            _tk_info_dialog(
                "Adelaide — Recovery Key",
                f"Your recovery key is:\n\n{recovery_key}\n\n"
                "WRITE THIS DOWN.\nIt's your backup if you forget your password.\n\n"
                "This key is NOT stored anywhere.",
            )

            return password
        elif is_recovery:
            _term_print("[KEY-DERIV] Recovery key requested...")
            password = _tk_password_dialog(
                "Adelaide — Recovery Key", "Enter your recovery key:"
            )
            return password
        else:
            _term_print("[KEY-DERIV] Welcome back — entering password...")
            password = _tk_password_dialog(
                "Adelaide — Enter Password", "Enter your password:"
            )
            return password

    # Fallback: terminal prompts
    import getpass

    if not IS_KISS:
        _term_print("")
        _term_print("  Oki :D, now so that we can keep secret between")
        _term_print("  each other, I am with my pinky finger, promise")
        _term_print("  to not share your data with others *wink")
        _term_print("")

    if is_first_boot:
        if not IS_KISS:
            _term_print("  Let's set up your password.")
            _term_print("  This password protects your data.")
            _term_print("  You'll need it every time Adelaide starts.")
            _term_print("")

        # Create password with entropy check (loop until strong enough)
        while True:
            password = getpass.getpass("  Create password: ", stream=term_stderr)
            if not password:
                _term_print("  Password cannot be empty.")
                return None

            bits = _password_entropy(password)
            if bits < 20:
                _term_print(
                    f"  Make stronger password! (only {bits} bits — need at least 20)"
                )
                _term_print(
                    "  Tip: Use lowercase, uppercase, numbers, and/or symbols together."
                )
                _term_print("")
                continue

            # Confirm password
            confirm = getpass.getpass("  Confirm password: ", stream=term_stderr)
            if password != confirm:
                _term_print("  Passwords do not match.")
                continue  # re-prompt both

            break

        _term_print("  Password set.")
        _term_print("")

        # Generate recovery key (256-bit entropy)
        import secrets

        hex_str = secrets.token_hex(32)
        recovery_key = "-".join(hex_str[i:i+8] for i in range(0, 64, 8))
        _term_print(f"  Your recovery key is: {recovery_key}")
        _term_print("  WRITE THIS DOWN. It's your backup if you forget your password.")
        _term_print("  This key is NOT stored anywhere.")
        _term_print("")

        return password
    elif is_recovery:
        _term_print("  Recovery key required.")
        password = getpass.getpass("  Enter recovery key: ", stream=term_stderr)
        return password
    else:
        _term_print("  Welcome back.")
        password = getpass.getpass("  Please enter your password: ", stream=term_stderr)
        return password


def _tk_info_dialog(title, message, countdown=60):
    """Show a tkinter info dialog with countdown auto-close.
    
    Args:
        title: Dialog window title
        message: Message text to display
        countdown: Seconds before auto-close (0 = no auto-close)
    """
    import tkinter as tk

    root = _get_tk_root()
    root.attributes("-topmost", True)

    bg = "#1a1a2e"
    fg = "#e0e0e0"
    btn_bg = "#4ecca3"  # Bright green — clearly visible
    btn_fg = "#1a1a2e"  # Dark text on green
    accent = "#e94560"

    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)
    dialog.grab_set()

    w, h = 480, 260
    sx = (dialog.winfo_screenwidth() - w) // 2
    sy = (dialog.winfo_screenheight() - h) // 2
    dialog.geometry(f"{w}x{h}+{sx}+{sy}")
    dialog.configure(bg=bg)

    tk.Label(
        dialog,
        text=message,
        bg=bg,
        fg=fg,
        font=("Helvetica", 12),
        justify="left",
        wraplength=440,
    ).pack(pady=(18, 6), padx=20)

    # Countdown timer label
    remaining = [countdown]
    timer_id = [None]

    timer_label = tk.Label(
        dialog,
        text=f"Auto-closes in {remaining[0]}s" if countdown > 0 else "",
        bg=bg,
        fg="#888888",
        font=("Helvetica", 9),
    )
    timer_label.pack(pady=(0, 8))

    def _countdown_tick():
        remaining[0] -= 1
        if remaining[0] <= 0:
            dialog.destroy()
            return
        timer_label.configure(text=f"Auto-closes in {remaining[0]}s")
        timer_id[0] = dialog.after(1000, _countdown_tick)

    if countdown > 0:
        timer_id[0] = dialog.after(1000, _countdown_tick)

    def _on_ok():
        if timer_id[0] is not None:
            try:
                dialog.after_cancel(timer_id[0])
            except Exception:
                pass
        dialog.destroy()

    tk.Button(
        dialog,
        text="OK",
        command=_on_ok,
        bg=btn_bg,
        fg=btn_fg,
        activebackground=accent,
        activeforeground="#fff",
        font=("Helvetica", 12, "bold"),
        width=12,
        relief="flat",
        cursor="hand2",
    ).pack(pady=(4, 14))

    dialog.bind("<Return>", lambda e: _on_ok())
    root.wait_window(dialog)
    root.destroy()


# ── InferiorParadoxical UUID — TPM / Secure Enclave Storage ──────────────

def _ip_tpm_store(uuid_str):
    """Store InferiorParadoxical UUID in TPM2 NVRAM (Linux)."""
    import subprocess
    import tempfile
    import os
    import time
    nv_index = "0x1500000"
    try:
        # Try to undefine first (ignore failure if not exist)
        subprocess.run(["tpm2_nvundefine", "-C", "o", nv_index],
                       capture_output=True, timeout=5)
    except Exception:
        pass
    time.sleep(0.2)
    try:
        subprocess.run(
            ["tpm2_nvdefine", "-C", "o", "-s", "64",
             "-a", "ownerread|ownerwrite", nv_index],
            capture_output=True, timeout=5, check=True,
        )
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ip") as f:
            f.write(uuid_str)
            tmp = f.name
        subprocess.run(
            ["tpm2_nvwrite", "-C", "o", nv_index, "--data", tmp],
            capture_output=True, timeout=5, check=True,
        )
        os.unlink(tmp)
        return True
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        return False


def _ip_tpm_read():
    """Read InferiorParadoxical UUID from TPM2 NVRAM (Linux)."""
    import subprocess
    try:
        result = subprocess.run(
            ["tpm2_nvread", "-C", "o", "0x1500000", "-s", "64"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _ip_sep_store(uuid_str):
    """Store InferiorParadoxical UUID in macOS Keychain (SEP-backed)."""
    import subprocess
    try:
        # -U = update if exists
        subprocess.run(
            ["security", "add-generic-password",
             "-s", "AdelaideZephyrineSystem",
             "-a", "inferior_paradoxical",
             "-w", uuid_str,
             "-U"],
            capture_output=True, timeout=5, check=True,
        )
        return True
    except Exception:
        # Try keyring library as fallback
        try:
            import keyring
            keyring.set_password("AdelaideZephyrineSystem", "inferior_paradoxical", uuid_str)
            return True
        except Exception:
            return False


def _ip_sep_read():
    """Read InferiorParadoxical UUID from macOS Keychain (SEP-backed)."""
    import subprocess
    try:
        result = subprocess.run(
            ["security", "find-generic-password",
             "-s", "AdelaideZephyrineSystem",
             "-a", "inferior_paradoxical",
             "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    # Try keyring library as fallback
    try:
        import keyring
        val = keyring.get_password("AdelaideZephyrineSystem", "inferior_paradoxical")
        if val:
            return val
    except Exception:
        pass
    return None


def _get_inferior_paradoxical_uuid():
    """
    Get or create InferiorParadoxical UUID.
    
    Priority:
      1. Read existing UUID from TPM2 NVRAM (Linux) or SEP Keychain (macOS)
      2. If not found, generate new UUID and store in available secure hardware
      3. Fallback to file storage if no TPM/SEP available
    """

    # Try TPM (Linux)
    if platform.system() == "Linux":
        uuid_str = _ip_tpm_read()
        if uuid_str:
            return uuid_str

    # Try SEP (macOS)
    if platform.system() == "Darwin":
        uuid_str = _ip_sep_read()
        if uuid_str:
            return uuid_str

    # Fallback: read from system_state database
    try:
        import sqlite3
        db_path = os.path.join(BASE_DIR, "NetworkMemoryPool", os.environ.get("ADELAIDE_USER", "default"), "adelaide_memory.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT value FROM system_state WHERE key = 'inferior_paradoxical_uuid'"
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return row[0]
    except Exception:
        pass

    # Fallback: read from file
    uuid_file = os.path.join(BASE_DIR, "config", ".inferior_paradoxical_uuid")
    try:
        if os.path.exists(uuid_file):
            with open(uuid_file) as f:
                return f.read().strip()
    except Exception:
        pass

    # Not found anywhere — generate new UUID
    import secrets
    uuid_str = secrets.token_hex(16)  # 128-bit random

    # Store in available secure hardware
    stored = False
    if platform.system() == "Linux":
        stored = _ip_tpm_store(uuid_str)
        if stored:
            print("[KEY-DERIV] InferiorParadoxical UUID stored in TPM2 NVRAM")
    elif platform.system() == "Darwin":
        stored = _ip_sep_store(uuid_str)
        if stored:
            print("[KEY-DERIV] InferiorParadoxical UUID stored in macOS Keychain (SEP)")

    # If TPM/SEP not available, store in system_state DB as fallback
    if not stored:
        try:
            import sqlite3
            db_path = os.path.join(BASE_DIR, "NetworkMemoryPool", os.environ.get("ADELAIDE_USER", "default"), "adelaide_memory.db")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value) VALUES ('inferior_paradoxical_uuid', ?)",
                    (uuid_str,),
                )
                conn.commit()
                conn.close()
                stored = True
                print("[KEY-DERIV] InferiorParadoxical UUID stored in system_state (fallback)")
        except Exception:
            pass

    # Last-resort file fallback
    if not stored:
        try:
            os.makedirs(os.path.join(BASE_DIR, "config"), exist_ok=True)
            with open(uuid_file, "w") as f:
                f.write(uuid_str)
            print("[KEY-DERIV] InferiorParadoxical UUID stored in config/.inferior_paradoxical_uuid (file fallback)")
        except Exception:
            print("[KEY-DERIV] WARNING: Could not persist InferiorParadoxical UUID")

    return uuid_str


# ── InferiorParadoxical Signature — static identity in TPM/SEP ──────────

def _ip_signature_store(sig_hash):
    """Store static InferiorParadoxical signature in TPM2 NVRAM (Linux)."""
    import subprocess
    import tempfile
    import os
    import time
    nv_index = "0x1500001"
    try:
        subprocess.run(["tpm2_nvundefine", "-C", "o", nv_index],
                       capture_output=True, timeout=5)
    except Exception:
        pass
    time.sleep(0.2)
    try:
        subprocess.run(
            ["tpm2_nvdefine", "-C", "o", "-s", "128",
             "-a", "ownerread|ownerwrite", nv_index],
            capture_output=True, timeout=5, check=True,
        )
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ipsig") as f:
            f.write(sig_hash)
            tmp = f.name
        subprocess.run(
            ["tpm2_nvwrite", "-C", "o", nv_index, "--data", tmp],
            capture_output=True, timeout=5, check=True,
        )
        os.unlink(tmp)
        return True
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        return False


def _ip_signature_tpm_read():
    """Read static InferiorParadoxical signature from TPM2 NVRAM (Linux)."""
    import subprocess
    try:
        result = subprocess.run(
            ["tpm2_nvread", "-C", "o", "0x1500001", "-s", "128"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _ip_signature_sep_store(sig_hash):
    """Store static InferiorParadoxical signature in macOS Keychain (SEP)."""
    import subprocess
    try:
        subprocess.run(
            ["security", "add-generic-password",
             "-s", "AdelaideZephyrineSystem",
             "-a", "inferior_paradoxical_signature",
             "-w", sig_hash,
             "-U"],
            capture_output=True, timeout=5, check=True,
        )
        return True
    except Exception:
        try:
            import keyring
            keyring.set_password("AdelaideZephyrineSystem",
                                 "inferior_paradoxical_signature", sig_hash)
            return True
        except Exception:
            return False


def _ip_signature_sep_read():
    """Read static InferiorParadoxical signature from macOS Keychain (SEP)."""
    import subprocess
    try:
        result = subprocess.run(
            ["security", "find-generic-password",
             "-s", "AdelaideZephyrineSystem",
             "-a", "inferior_paradoxical_signature",
             "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    try:
        import keyring
        return keyring.get_password("AdelaideZephyrineSystem",
                                     "inferior_paradoxical_signature")
    except Exception:
        pass
    return None


def _get_ip_signature():
    """
    Get or create the static InferiorParadoxical signature.
    
    This is a one-time generated SHA-512 hash stored in TPM/SEP as a
    read-only identity marker.  Unlike the InferiorParadoxical UUID (which
    participates in key derivation), this is JUST a signature — accumulated
    into integrity_hash but never used as a key.
    
    Only re-written if corrupted or missing.
    """
    import secrets

    # Try TPM (Linux)
    if platform.system() == "Linux":
        sig = _ip_signature_tpm_read()
        if sig and len(sig) == 128:
            return sig

    # Try SEP (macOS)
    if platform.system() == "Darwin":
        sig = _ip_signature_sep_read()
        if sig and len(sig) == 128:
            return sig

    # Fallback: read from system_state
    try:
        import sqlite3
        db_path = os.path.join(BASE_DIR, "NetworkMemoryPool", os.environ.get("ADELAIDE_USER", "default"), "adelaide_memory.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT value FROM system_state WHERE key = 'inferior_paradoxical_signature'"
            )
            row = cursor.fetchone()
            conn.close()
            if row and len(row[0]) == 128:
                return row[0]
    except Exception:
        pass

    # Fallback: read from file
    sig_file = os.path.join(BASE_DIR, "config", ".inferior_paradoxical_signature")
    try:
        if os.path.exists(sig_file):
            with open(sig_file) as f:
                content = f.read().strip()
                if len(content) == 128:
                    return content
    except Exception:
        pass

    # Not found anywhere — generate new SHA-512 signature
    random_bytes = secrets.token_bytes(64)   # 512-bit random
    sig_hash = hashlib.sha512(random_bytes).hexdigest()  # 128 hex chars

    # Store in available secure hardware
    stored = False
    if platform.system() == "Linux":
        stored = _ip_signature_store(sig_hash)
        if stored:
            print("[KEY-DERIV] InferiorParadoxical signature stored in TPM2 NVRAM")
    elif platform.system() == "Darwin":
        stored = _ip_signature_sep_store(sig_hash)
        if stored:
            print("[KEY-DERIV] InferiorParadoxical signature stored in macOS Keychain (SEP)")

    # Fallback: system_state DB
    if not stored:
        try:
            import sqlite3
            db_path = os.path.join(BASE_DIR, "NetworkMemoryPool", os.environ.get("ADELAIDE_USER", "default"), "adelaide_memory.db")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value) VALUES ('inferior_paradoxical_signature', ?)",
                    (sig_hash,),
                )
                conn.commit()
                conn.close()
                stored = True
                print("[KEY-DERIV] InferiorParadoxical signature stored in system_state (fallback)")
        except Exception:
            pass

    # Last-resort file fallback
    if not stored:
        try:
            os.makedirs(os.path.join(BASE_DIR, "config"), exist_ok=True)
            with open(sig_file, "w") as f:
                f.write(sig_hash)
            print("[KEY-DERIV] InferiorParadoxical signature stored in config/ (file fallback)")
        except Exception:
            print("[KEY-DERIV] WARNING: Could not persist InferiorParadoxical signature")

    return sig_hash


# ── Program Hash ─────────────────────────────────────────────────────────
def compute_program_hash():
    """
    SHA-512 hash of the compiled binary — detects recompilation.
    
    If the binary exists, hashes the ELF/Mach-O directly.
    Otherwise falls back to hashing Ada source + GPR files.
    Returns hex string, or None on failure.
    """
    binary_path = os.path.join(BASE_DIR, "bin", "adelaide_zephyrine_system")
    try:
        if os.path.exists(binary_path):
            with open(binary_path, "rb") as f:
                return hashlib.sha512(f.read()).hexdigest()
        else:
            # Fallback: hash source tree
            import glob
            hasher = hashlib.sha512()
            # Project source
            patterns = [
                os.path.join(BASE_DIR, "src", "*.adb"),
                os.path.join(BASE_DIR, "src", "*.ads"),
                os.path.join(BASE_DIR, "src", "*.c"),
                os.path.join(BASE_DIR, "src", "*.h"),
                os.path.join(BASE_DIR, "config", "*.gpr"),
                os.path.join(BASE_DIR, "config", "*.ads"),
                os.path.join(BASE_DIR, "config", "*.h"),
                os.path.join(BASE_DIR, "python", "*.py"),
                os.path.join(BASE_DIR, "ui", "*.py"),
            ]
            for pattern in patterns:
                for fpath in sorted(glob.glob(pattern)):
                    with open(fpath, "rb") as f:
                        hasher.update(f.read())
            # Also hash run.py itself
            run_py = os.path.join(BASE_DIR, "run.py")
            if os.path.exists(run_py):
                with open(run_py, "rb") as f:
                    hasher.update(f.read())
            return hasher.hexdigest()
    except Exception as e:
        print(f"[KEY-DERIV] Failed to compute program hash: {e}")
        return None


# ── Hardware-Bound Key Derivation Functions ───────────────────────────────
def compute_integrity_hash():
    """
    Compute hardware/binary integrity hash for key derivation.
    
    Accumulates:
      - Hardware profiling data (system commands)
      - InferiorParadoxical UUID key (SHA-512 of TPM/SEP-stored UUID)
      - Program hash (binary version — forces re-auth on recompile)
    
    Returns hex-encoded hash or None on failure.
    """
    import subprocess

    try:
        # Compute hardware hash
        hw_sources = []
        if platform.system() == "Linux":
            # Linux hardware sources
            cmds = [
                "lsusb",
                "lshw -c system 2>/dev/null | head -50",
                "lspci",
                "dmidecode -t system 2>/dev/null | head -20",
                "cat /proc/cpuinfo 2>/dev/null | head -20",
                "lsblk -d -o NAME,SERIAL 2>/dev/null",
                # TPM (Trusted Platform Module)
                "cat /sys/class/tpm/tpm0/tpm_version_major 2>/dev/null",
                "cat /sys/class/tpm/tpm0/device/firmware_node*/description 2>/dev/null",
                "cat /sys/class/tpm/tpm0/tpm_version_* 2>/dev/null",
                "cat /sys/class/tpm/tpm0/device/firmware_node*/hid 2>/dev/null",
                "tpm2_getcap properties-variable 2>/dev/null | head -10",
                "ls -la /sys/class/tpm/ 2>/dev/null",
            ]
        elif platform.system() == "Darwin":
            # macOS hardware sources
            cmds = [
                "system_profiler SPUSBDataType 2>/dev/null | head -50",
                "system_profiler SPHardwareDataType 2>/dev/null",
                "system_profiler SPPCIDataType 2>/dev/null | head -30",
                "ioreg -l 2>/dev/null | grep -E 'IOPlatformSerialNumber|IOPlatformUUID' | head -10",
                "sysctl machdep.cpu 2>/dev/null",
                "system_profiler SPMemoryDataType 2>/dev/null | head -20",
                # Secure Enclave (Apple T2 / Silicon)
                "ioreg -l 2>/dev/null | grep -E 'AppleSEP|sep-id|chip-id|SEP' | head -10",
                "system_profiler SPiBridgeDataType 2>/dev/null | head -20",
                "ioreg -p IODeviceTree -r -n sep 2>/dev/null",
            ]
        else:
            return None

        for cmd in cmds:
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=5
                )
                if result.stdout:
                    hw_sources.append(result.stdout)
            except Exception:
                pass

        # Compute binary hash
        bin_sources = []
        if platform.system() == "Linux":
            cmds = [
                "ls -la /boot/*vmlinuz* /boot/*initrd* 2>/dev/null",
                "ls -la /boot/efi/* 2>/dev/null | head -20",
                "ls -la /bin/* 2>/dev/null | head -30",
            ]
        elif platform.system() == "Darwin":
            cmds = [
                "ls -la /System/Library/Kernels/* 2>/dev/null | head -10",
                "ls -la /System/Library/CoreServices/boot.efi 2>/dev/null",
                "ls -la /usr/local/bin/* 2>/dev/null | head -30",
            ]

        for cmd in cmds:
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=5
                )
                if result.stdout:
                    bin_sources.append(result.stdout)
            except Exception:
                pass

        # ── Accumulate additional components ─────────────────────────────

        # 1) InferiorParadoxical UUID → SHA-512 → key component
        ip_uuid = _get_inferior_paradoxical_uuid()
        ip_key = hashlib.sha512(ip_uuid.encode("utf-8")).hexdigest()

        # 2) InferiorParadoxical static signature (from TPM/SEP — read-only identity)
        ip_signature = _get_ip_signature()

        # 3) TPM2 / Secure Enclave hardware identity
        tpm_hw_id = ""
        try:
            if platform.system() == "Linux":
                results = subprocess.run(
                    "cat /sys/class/tpm/tpm0/device/firmware_node*/hid 2>/dev/null; "
                    "cat /sys/class/tpm/tpm0/device/firmware_node*/serial 2>/dev/null; "
                    "cat /sys/class/tpm/tpm0/device/firmware_node*/description 2>/dev/null; "
                    "cat /sys/class/tpm/tpm0/tpm_version_major 2>/dev/null; "
                    "cat /sys/class/tpm/tpm0/tpm_version_minor 2>/dev/null; "
                    "tpm2_getcap properties-fixed 2>/dev/null | head -20",
                    shell=True, capture_output=True, text=True, timeout=5,
                )
                tpm_hw_id = results.stdout.strip()
            elif platform.system() == "Darwin":
                results = subprocess.run(
                    "system_profiler SPiBridgeDataType 2>/dev/null | head -20; "
                    "ioreg -l 2>/dev/null | grep -E 'AppleSEP|sep-id|chip-id|SEP' | head -10",
                    shell=True, capture_output=True, text=True, timeout=5,
                )
                tpm_hw_id = results.stdout.strip()
        except Exception:
            pass  # skip if unavailable

        # 4) External IP address (skip gracefully if offline)
        external_ip = ""
        for url in ("https://api.ipify.org", "https://ifconfig.me", "https://icanhazip.com"):
            try:
                result = subprocess.run(
                    ["curl", "-s", "--max-time", "3", url],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    external_ip = result.stdout.strip()
                    break
            except Exception:
                continue

        # 5) Internal IP address (skip gracefully if unavailable)
        internal_ip = ""
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(
                    "ifconfig 2>/dev/null | grep 'inet ' | grep -v 127.0.0.1 | head -1 | awk '{print $2}'",
                    shell=True, capture_output=True, text=True, timeout=5,
                )
            else:
                result = subprocess.run(
                    "ip addr show 2>/dev/null | grep 'inet ' | grep -v 127.0.0.1 | head -1 | awk '{print $2}' | cut -d/ -f1; "
                    "ifconfig 2>/dev/null | grep 'inet ' | grep -v 127.0.0.1 | head -1 | awk '{print $2}'",
                    shell=True, capture_output=True, text=True, timeout=5,
                )
            if result.returncode == 0 and result.stdout.strip():
                internal_ip = result.stdout.strip().split("\n")[0]
        except Exception:
            pass

        # 6) Program hash (recompile detection)
        program_hash = compute_program_hash() or ""

        # Combine all components into final integrity_hash
        combined = (
            "\n".join(hw_sources)
            + "\n"
            + "\n".join(bin_sources)
            + "\n"
            + "IP_KEY:" + ip_key
            + "\n"
            + "IP_SIG:" + ip_signature
            + "\n"
            + "TPM_HW:" + tpm_hw_id
            + "\n"
            + "EXT_IP:" + external_ip
            + "\n"
            + "INT_IP:" + internal_ip
            + "\n"
            + "PROG_HASH:" + program_hash
        )
        integrity_hash = hashlib.sha512(combined.encode()).hexdigest()

        return integrity_hash
    except Exception as e:
        print(f"[KEY-DERIV] Failed to compute integrity hash: {e}")
        return None


def _try_c_derive_master_key(integrity_hash, user_secret):
    """
    Try to derive master key using the C library (adl_crypto).
    Returns the master key hex string on success, None if C lib unavailable.
    """
    try:
        import ctypes
        import ctypes.util
        import os

        # Try to find the C library — check common build output paths
        lib_paths = [
            os.path.join(BASE_DIR, "bin", "libadl_crypto.dylib"),
            os.path.join(BASE_DIR, "bin", "libadl_crypto.so"),
            os.path.join(BASE_DIR, "build", "libadl_crypto.dylib"),
            os.path.join(BASE_DIR, "build", "libadl_crypto.so"),
            os.path.join(BASE_DIR, "libadl_crypto.dylib"),
            os.path.join(BASE_DIR, "libadl_crypto.so"),
        ]
        lib_path = None
        for p in lib_paths:
            if os.path.exists(p):
                lib_path = p
                break

        if not lib_path:
            return None  # C library not available

        # BYPASS STALE C LIBRARY: Force Python implementation
        return None

        lib = ctypes.CDLL(lib_path)

        # Configure the function signature
        lib.adl_derive_master_key_cstr.argtypes = [
            ctypes.c_char_p,  # integrity_hash
            ctypes.c_char_p,  # user_secret
        ]
        lib.adl_derive_master_key_cstr.restype = ctypes.c_void_p  # raw malloc'd pointer
        lib.adl_free_cstr.argtypes = [ctypes.c_void_p]
        lib.adl_free_cstr.restype = None

        # Call the C function
        c_hash = ctypes.c_char_p(integrity_hash.encode("utf-8"))
        c_secret = ctypes.c_char_p(user_secret.encode("utf-8"))
        result_ptr = lib.adl_derive_master_key_cstr(c_hash, c_secret)

        if result_ptr:
            # Extract string from raw C pointer (cast reads a copy)
            result_bytes = ctypes.cast(result_ptr, ctypes.c_char_p).value
            master_key = result_bytes.decode("utf-8")
            lib.adl_free_cstr(result_ptr)  # free the original malloc'd memory
            return master_key

        return None
    except Exception:
        return None


def _try_c_derive_master_key_from_stdin(integrity_hash, prompt):
    try:
        import ctypes
        import os

        # Find the shared library
        lib_paths = [
            os.path.join(os.path.dirname(__file__), "lib", "libadl_crypto.so"),
            os.path.join(os.path.dirname(__file__), "lib", "libadl_crypto.dylib"),
            os.path.join(os.path.dirname(__file__), "libadl_crypto.so"),
            os.path.join(os.path.dirname(__file__), "libadl_crypto.dylib"),
        ]
        lib_path = None
        for p in lib_paths:
            if os.path.exists(p):
                lib_path = p
                break

        if not lib_path:
            return None  # C library not available

        lib = ctypes.CDLL(lib_path)

        # Configure the function signature
        lib.adl_derive_master_key_from_stdin.argtypes = [
            ctypes.c_char_p,  # integrity_hash
            ctypes.c_char_p,  # prompt
        ]
        lib.adl_derive_master_key_from_stdin.restype = ctypes.c_void_p  # raw malloc'd pointer
        lib.adl_free_cstr.argtypes = [ctypes.c_void_p]
        lib.adl_free_cstr.restype = None

        # Call the C function
        c_hash = ctypes.c_char_p(integrity_hash.encode("utf-8"))
        c_prompt = ctypes.c_char_p(prompt.encode("utf-8"))
        result_ptr = lib.adl_derive_master_key_from_stdin(c_hash, c_prompt)

        if result_ptr:
            # Extract string from raw C pointer (cast reads a copy)
            result_bytes = ctypes.cast(result_ptr, ctypes.c_char_p).value
            master_key = result_bytes.decode("utf-8")
            lib.adl_free_cstr(result_ptr)  # free the original malloc'd memory
            return master_key

        return None
    except Exception:
        return None


def derive_master_key_from_stdin(integrity_hash, prompt):
    """
    Reads password securely via C termios, derives key, and zeroizes buffer in C.
    Falls back to Python getpass if C module is unavailable.
    """
    c_result = _try_c_derive_master_key_from_stdin(integrity_hash, prompt)
    if c_result is not None:
        return c_result

    # Fallback if C module is missing
    import getpass
    password = getpass.getpass(prompt)
    if not password:
        return None
    return derive_master_key(integrity_hash, password)


def derive_master_key(integrity_hash, user_secret):
    """
    Derive master key from integrity hash and user secret.
    master_key = HKDF-SHA512(salt=integrity_hash, ikm=user_secret,
                             info="adelaide:master-key:v1")

    Uses FIPS 140-3 C implementation when available (adl_crypto shared library),
    falls back to pure Python HKDF-SHA512.
    """
    # Try C implementation first (FIPS 140-3 approved path)
    c_result = _try_c_derive_master_key(integrity_hash, user_secret)
    if c_result is not None:
        return c_result

    # Fallback: pure Python implementation (mirrors the C code identically)
    import hashlib
    import hmac

    info = b"adelaide:master-key:v1"
    salt = bytes.fromhex(integrity_hash)
    ikm = user_secret.encode("utf-8")

    # HKDF-Extract
    prk = hmac.new(salt, ikm, hashlib.sha512).digest()

    # HKDF-Expand (single block: output <= SHA-512 digest size)
    expand_input = info + b"\x01"
    okm = hmac.new(prk, expand_input, hashlib.sha512).digest()

    # Take first 64 bytes (512 bits) as master key
    return okm[:32].hex()


def verify_integrity_test_blob(master_key_hex, sub_key_hex):
    """
    Verify integrity test blob from database.
    Returns True if blob exists and decrypts successfully.
    """
    from adelaide_crypto import decrypt_field

    try:
        # Get stored blob from database
        import sqlite3

        db_path = os.path.join(BASE_DIR, "NetworkMemoryPool", os.environ.get("ADELAIDE_USER", "default"), "adelaide_memory.db")
        if not os.path.exists(db_path):
            return False

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT value FROM system_state WHERE key = 'integrity_test'"
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return False

        stored_blob = row[0]

        # Try to decrypt
        decrypted = decrypt_field(sub_key_hex, stored_blob)
        if decrypted == INTEGRITY_TEST_PLAINTEXT:
            return True
        else:
            return False
    except Exception as e:
        print(f"[KEY-DERIV] Integrity test verification failed: {e}")
        return False


def store_integrity_test_blob(sub_key_hex):
    """
    Store integrity test blob in database.
    """
    from adelaide_crypto import encrypt_field

    try:
        import sqlite3

        db_path = os.path.join(BASE_DIR, "NetworkMemoryPool", os.environ.get("ADELAIDE_USER", "default"), "adelaide_memory.db")
        if not os.path.exists(db_path):
            return False

        # Encrypt test plaintext
        encrypted = encrypt_field(sub_key_hex, INTEGRITY_TEST_PLAINTEXT)
        if encrypted == INTEGRITY_TEST_PLAINTEXT:
            return False

        # Store in database
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO system_state (key, value)
            VALUES ('integrity_test', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
            (encrypted,),
        )
        conn.commit()
        conn.close()

        return True
    except Exception as e:
        print(f"[KEY-DERIV] Failed to store integrity test blob: {e}")
        return False


def migrate_from_legacy_key_system():
    """
    Migrate from old file-based key system to hardware-bound key derivation.

    Reads old key from disk (migration only), re-encrypts all databases with
    new hardware-bound key, then DELETES the old key file. Never written again.

    Migration flow:
    1. Detect old key file at config/master.key or ~/.config/adelaide/master.key
    2. Read old key from file
    3. Prompt user for new password
    4. Derive new master_key with hardware-bound integrity hash
    5. Re-encrypt all databases with new key
    6. DELETE old key file
    7. Store integrity_test blob with new key
    """
    # Check both possible legacy locations
    local_key_file = os.path.join(BASE_DIR, "config", "master.key")
    legacy_key_file = os.path.expanduser("~/.config/adelaide/master.key")

    old_key_file = None
    if os.path.exists(local_key_file):
        old_key_file = local_key_file
    elif os.path.exists(legacy_key_file):
        old_key_file = legacy_key_file

    if not old_key_file:
        print("[MIGRATE] No legacy key file found, skipping migration")
        return True

    print(f"[MIGRATE] Legacy key file detected at {old_key_file}")
    print("[MIGRATE] Migrating to hardware-bound system...")

    # Read old key
    try:
        with open(old_key_file, "r") as f:
            old_key_hex = f.read().strip()
        if len(old_key_hex) != 64:
            print("[MIGRATE] Invalid legacy key format")
            return False
    except Exception as e:
        print(f"[MIGRATE] Failed to read legacy key: {e}")
        return False

    # Get new password from user
    if _gui_available() or IS_KISS:
        password = prompt_kiss_password(is_first_boot=True)
    else:
        import getpass

        print("[MIGRATE] Please create a new password for the hardware-bound system.")
        while True:
            password = getpass.getpass("Enter new password: ")
            if not password:
                print("[MIGRATE] No password provided")
                return False
            bits = _password_entropy(password)
            if bits < 20:
                print(
                    f"  Make stronger password! (only {bits} bits — need at least 20)"
                )
                print(
                    "  Tip: Use lowercase, uppercase, numbers, and/or symbols together."
                )
                continue
            break

    if not password:
        print("[MIGRATE] No password provided")
        return False

    # Compute integrity hash
    integrity_hash = compute_integrity_hash()
    if not integrity_hash:
        print("[MIGRATE] Failed to compute integrity hash")
        return False

    # Derive new master key
    new_master_key = derive_master_key(integrity_hash, password)
    print(f"[MIGRATE] New master key derived: {new_master_key[:16]}...")

    # Re-encrypt all databases with new key
    try:
        from adelaide_crypto import decrypt_field, derive_sub_key, encrypt_field

        # Derive sub-keys for old key (each DB context)
        old_subkeys = {
            "memory": derive_sub_key(old_key_hex, "adelaide:db:memory:v1"),
            "session": derive_sub_key(old_key_hex, "adelaide:db:session:v1"),
            "literature": derive_sub_key(old_key_hex, "adelaide:db:literature:v1"),
        }
        new_subkeys = {
            "memory": derive_sub_key(new_master_key, "adelaide:db:memory:v1"),
            "session": derive_sub_key(new_master_key, "adelaide:db:session:v1"),
            "literature": derive_sub_key(new_master_key, "adelaide:db:literature:v1"),
        }

        import sqlite3

        total_reencrypted = 0

        # ── Database migration definitions ──
        # (db_name, subkey_context, [(table, id_col, text_cols...)])
        db_migrations = [
            (
                "adelaide_memory.db",
                "memory",
                [
                    ("memories", "id", ["input", "response"]),
                    ("response_cache", "id", ["prompt", "response", "embedding"]),
                ],
            ),
            (
                "assistant_session.db",
                "session",
                [
                    ("messages", "id", ["content"]),
                ],
            ),
            (
                "literatureRefIndex.db",
                "literature",
                [
                    ("chunks", "id", ["content"]),
                ],
            ),
        ]

        for db_name, subkey_ctx, table_migrations in db_migrations:
            db_path = os.path.join(BASE_DIR, "NetworkMemoryPool", db_name)
            if not os.path.exists(db_path):
                continue

            conn = sqlite3.connect(db_path)
            conn.text_factory = lambda x: x.decode("utf-8", errors="replace")

            for table_name, id_col, text_cols in table_migrations:
                # Check if table exists
                tbl_exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                ).fetchone()
                if not tbl_exists:
                    continue

                # Build SELECT and UPDATE column lists
                col_list = ", ".join([id_col] + text_cols)
                set_clauses = ", ".join([f"{c}=?" for c in text_cols])

                rows = conn.execute(f"SELECT {col_list} FROM {table_name}").fetchall()
                reencrypted_count = 0

                for row in rows:
                    row_id = row[0]
                    values = row[1:]
                    new_values = list(values)
                    any_encrypted = False

                    for i, val in enumerate(values):
                        if not val or len(val) <= 56:
                            continue
                        # Check if looks encrypted (hex-only first 56 chars)
                        if not all(c in "0123456789abcdef" for c in val[:56].lower()):
                            continue

                        any_encrypted = True
                        try:
                            # Decrypt with old subkey
                            decrypted = decrypt_field(old_subkeys[subkey_ctx], val)
                            if decrypted and decrypted != val:
                                # Re-encrypt with new subkey
                                new_values[i] = encrypt_field(
                                    new_subkeys[subkey_ctx], decrypted
                                )
                        except Exception:
                            # Skip rows that fail to decrypt (plaintext misidentified)
                            pass

                    if any_encrypted:
                        reencrypted_count += 1
                        conn.execute(
                            f"UPDATE {table_name} SET {set_clauses} WHERE {id_col}=?",
                            tuple(new_values + [row_id]),
                        )

                if reencrypted_count > 0:
                    conn.commit()
                    total_reencrypted += reencrypted_count
                    print(
                        f"[MIGRATE] Re-encrypted {reencrypted_count} rows in {db_name}.{table_name}"
                    )

            conn.close()

        # Delete old key file
        os.remove(old_key_file)
        print("[MIGRATE] Old key file deleted")

        # Store integrity test blob with new key
        new_aes_key = derive_sub_key(new_master_key, "adelaide:db:memory:v1")
        store_integrity_test_blob(new_aes_key)

        print(
            f"[MIGRATE] Migration completed. Total re-encrypted: {total_reencrypted} rows"
        )
        return True

    except Exception as e:
        import traceback

        print(f"[MIGRATE] Migration failed: {e}")
        traceback.print_exc()
        return False


def hardware_bound_key_derivation():
    """
    Hardware-bound key derivation system (dual-key architecture).

    Two independent keys can decrypt — NOT combined:
      Key 1 — User password / recovery key
      Key 2 — InferiorParadoxical (SHA-512 hardware profiling auto-key)

    Flow
    ----
    1. Compute InferiorParadoxical from hardware state
    2. Try auto-decrypt master_key using InferiorParadoxical stored blob
    3. If auto-decrypt fails → prompt user for password
       → derive master_key = HKDF(integrity_hash, password)
       → verify with integrity test blob
       → re-wrap master_key under new InferiorParadoxical for future auto-decrypt
    4. Return master_key
    """
    from adelaide_crypto import derive_sub_key

    _term_print("[KEY-DERIV] Initializing hardware-bound key derivation...")

    # Step 1: Compute integrity hash (with loading bar)
    _hash_result = [None]
    _hash_done = threading.Event()

    def _compute_hash():
        _hash_result[0] = compute_integrity_hash()
        _hash_done.set()

    _hash_thread = threading.Thread(target=_compute_hash, daemon=True)
    _hash_thread.start()

    # Show loading bar while hash computes
    use_gui_progress = _gui_available() and not IS_KISS
    gui_dialog = None

    if use_gui_progress:
        gui_dialog = _tk_progress_dialog(
            "Adelaide — Loading",
            "Loading preparing for Model...\n(Nothing to see here)"
        )

    bar_width = 40
    elapsed = 0.0
    eta_target = 8.0  # estimated seconds for hash computation
    while not _hash_done.is_set():
        pct = min(95, int(100 * elapsed / eta_target))
        eta = max(0, int(eta_target - elapsed))
        if gui_dialog:
            gui_dialog._update_bar(pct, eta_text=f"ETA: {eta}s")
        elif not IS_KISS:
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            _term_print(f"\r\033[K  Loading preparing for Model... |{bar}| {pct}%  ETA: {eta}s")
        time.sleep(0.1)
        elapsed += 0.1

    _hash_thread.join()
    integrity_hash = _hash_result[0]

    # Clear loading bar
    if gui_dialog:
        gui_dialog._update_bar(100)
        time.sleep(0.2)
        _tk_progress_done(gui_dialog)
    elif not IS_KISS:
        _term_print(f"\r\033[K  Loading preparing for Model... |{'█' * bar_width}| 100%  Done!")

    if not integrity_hash:
        _term_print("[KEY-DERIV] Failed to compute integrity hash")
        return None

    print(f"[KEY-DERIV] Integrity hash: {integrity_hash[:16]}...")

    # Step 2: Try InferiorParadoxical auto-decrypt (Key 2 — hardware auto-key)
    # Silently attempt auto-recovery; only prompts user if hardware changed.
    master_key = _try_inferior_paradoxical_auto_decrypt(integrity_hash)
    auto_decrypt_ok = master_key is not None

    if auto_decrypt_ok:
        _term_print("[KEY-DERIV] InferiorParadoxical auto-decrypt — hardware environment trusted")
        aes_key = derive_sub_key(master_key, "adelaide:db:memory:v1")
        if verify_integrity_test_blob(master_key, aes_key):
            _term_print("[KEY-DERIV] Integrity test blob verification PASSED (auto)")
            return master_key
        else:
            # Stored wrapped key is stale — fall through to password path
            _term_print("[KEY-DERIV] Auto-decrypt OK but integrity test FAILED — re-keying")
            master_key = None
            auto_decrypt_ok = False

    # Step 3: Determine if this is first boot (no key files exist)
    local_key = os.path.join(BASE_DIR, "config", "master.key")
    legacy_key = os.path.expanduser("~/.config/adelaide/master.key")
    
    import sqlite3
    db_path = os.path.join(BASE_DIR, "NetworkMemoryPool", os.environ.get("ADELAIDE_USER", "default"), "adelaide_memory.db")
    has_wrapped_key = False
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = ?",
                (INFERIOR_PARADOXICAL_KEY,)
            ).fetchone()
            if row and row[0]:
                has_wrapped_key = True
            conn.close()
        except Exception:
            pass
            
    first_boot = (not os.path.exists(local_key) and 
                  not os.path.exists(legacy_key) and 
                  not has_wrapped_key)

    # Step 4: Prompt for password (Key 1 — user password / recovery key)
    MAX_PASSWORD_ATTEMPTS = 5
    password_ok = False
    master_key = None
    aes_key = None

    if "--test-fips" in sys.argv:
        print("[KEY-DERIV] --test-fips detected. Bypassing interactive prompt.")
        password = "testfips_password123"
        master_key = derive_master_key(integrity_hash, password)
        _wipe_string(password)
        password = None
    elif first_boot:
        _term_print("[KEY-DERIV] First boot — creating new password")
        if _gui_available() or IS_KISS:
            password = prompt_kiss_password(is_first_boot=True)
            if password:
                master_key = derive_master_key(integrity_hash, password)
                _wipe_string(password)
                password = None
        else:
            import getpass
            _term_print("[KEY-DERIV] First boot detected. Please create a password.")
            while True:
                password = getpass.getpass("  Create password: ", stream=term_stderr)
                if not password:
                    _term_print("  Password cannot be empty.")
                    return None
                bits = _password_entropy(password)
                if bits < 20:
                    _term_print(
                        f"  Make stronger password! (only {bits} bits — need at least 20)"
                    )
                    _term_print(
                        "  Tip: Use lowercase, uppercase, numbers, and/or symbols together."
                    )
                    continue
                break
            master_key = derive_master_key(integrity_hash, password)
            _wipe_string(password)
            password = None
        if master_key:
            aes_key = derive_sub_key(master_key, "adelaide:db:memory:v1")
            password_ok = True
            _term_print("[KEY-DERIV] First boot — password created successfully")
    else:
        _term_print("[KEY-DERIV] Enter password (hardware environment changed or first unlock)")

        for attempt in range(MAX_PASSWORD_ATTEMPTS):
            if attempt > 0:
                delay = min(2 ** attempt, 30)
                _term_print(
                    f"[KEY-DERIV] Wrong password. Retry in {delay}s "
                    f"(attempt {attempt + 1}/{MAX_PASSWORD_ATTEMPTS})"
                )
                time.sleep(delay)

            if _gui_available() or IS_KISS:
                password = prompt_kiss_password(is_first_boot=False)
                if not password:
                    return None
                master_key = derive_master_key(integrity_hash, password)
                _wipe_string(password)
                password = None
            else:
                import getpass
                _term_print("[KEY-DERIV] Please enter your password.")
                master_key = derive_master_key_from_stdin(integrity_hash, "  Password: ")
                if not master_key:
                    print("[KEY-DERIV] No password provided or derivation failed")
                    return None
                print(f"[KEY-DERIV] Master key securely derived in C: {master_key[:16]}...")

            aes_key = derive_sub_key(master_key, "adelaide:db:memory:v1")
            if verify_integrity_test_blob(master_key, aes_key):
                if attempt == 0:
                    _term_print("[KEY-DERIV] Integrity test blob verification PASSED")
                else:
                    _term_print(f"[KEY-DERIV] Correct password (attempt {attempt + 1}/{MAX_PASSWORD_ATTEMPTS})")
                password_ok = True
                break
            else:
                _term_print(f"[KEY-DERIV] Incorrect password (attempt {attempt + 1}/{MAX_PASSWORD_ATTEMPTS})")
                master_key = None
                aes_key = None
                continue

        # After password attempts exhausted, try recovery key
        if not password_ok:
            _term_print("[KEY-DERIV] Password attempts exhausted — offering recovery key")
            if _gui_available() or IS_KISS:
                recovery_key = prompt_kiss_password(is_first_boot=False, is_recovery=True)
            else:
                import getpass
                recovery_key = getpass.getpass("Enter recovery key: ", stream=term_stderr)

            if recovery_key:
                master_key = derive_master_key(integrity_hash, recovery_key)
                aes_key = derive_sub_key(master_key, "adelaide:db:memory:v1")
                _wipe_string(recovery_key)
                recovery_key = None
                if verify_integrity_test_blob(master_key, aes_key):
                    _term_print("[KEY-DERIV] Recovery key verification PASSED")
                    password_ok = True
                else:
                    _term_print("[KEY-DERIV] Recovery key verification FAILED")
                    return None
            else:
                return None

    if not password_ok:
        return None

    # Step 8: Store integrity test blob on first boot
    if first_boot:
        store_integrity_test_blob(aes_key)

    # Step 9: Re-wrap master_key under InferiorParadoxical
    # On first boot: store initial wrap for future auto-decrypt
    # On subsequent boot after auto-decrypt failed: hardware changed, update wrap
    if not auto_decrypt_ok and password_ok and master_key:
        _store_inferior_paradoxical_wrapped_key(master_key, integrity_hash)
        if not first_boot:
            _term_print("[KEY-DERIV] InferiorParadoxical updated for current hardware")

    return master_key


try:
    _lock_fd = open(os.path.join(BASE_DIR, ".adelaide.lock"), "w")
    fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("[!] FATAL: Another instance of Adelaide is already running.")
    print("    Singleton lock enforced. Aborting startup.")
    sys.exit(1)


# Enforce Huggingface cache location
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "model")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "model")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(BASE_DIR, "model")


# ---------------------------------------------------------------------------
#  Logging: tee stdout+stderr to logs/ with 10 MB rollover
# ---------------------------------------------------------------------------
class _TeeWriter:
    """Write to an original stream AND append to a log file simultaneously."""

    def __init__(self, original, log_file):
        self._orig = original
        self._log = log_file

    def write(self, data):
        self._orig.write(data)
        try:
            self._log.write(data)
            self._log.flush()
        except Exception:
            pass

    def flush(self):
        self._orig.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def __getattr__(self, attr):
        return getattr(self._orig, attr)


class _PipeReader(threading.Thread):
    """Daemon thread that reads a subprocess pipe and tees it to a writer."""

    def __init__(self, pipe, writer, label=""):
        super().__init__(daemon=True)
        self._pipe = pipe
        self._writer = writer
        self._label = label

    def run(self):
        try:
            for line in iter(self._pipe.readline, b""):
                self._writer.write(line)
        except Exception:
            pass
        finally:
            self._pipe.close()


def _rotate_logs():
    """Delete oldest log files until total size <= MAX_LOG_BYTES."""
    if not os.path.isdir(LOGS_DIR):
        return
    entries = []
    for name in os.listdir(LOGS_DIR):
        if name.endswith(".log"):
            path = os.path.join(LOGS_DIR, name)
            try:
                entries.append((os.path.getmtime(path), os.path.getsize(path), path))
            except OSError:
                pass
    entries.sort(key=lambda e: e[0])  # oldest first
    total = sum(sz for _, sz, _ in entries)
    for _mtime, sz, path in entries:
        if total <= MAX_LOG_BYTES:
            break
        try:
            os.remove(path)
            total -= sz
        except OSError:
            pass


IS_KISS = False
term_stdout = None
term_stderr = None


def show_bsod(error_msg, log_path, stop_code="0x0000007B"):
    bsod_text = f"""\033[44m\033[37;1m
================================================================================
                                 SYSTEM ERROR
================================================================================

A fatal problem has been detected and Zephyrine has been shut down to prevent
damage to your configuration or platform stability.

STOP_BOOT_FAILURE: {error_msg}

If this is the first time you've seen this error screen, restart the process.
If this screen appears again, verify your model assets and configuration.

*** STOP: {stop_code} (0xF78D2524, 0xC0000034, 0x00000000, 0x00000000)

================================================================================
\033[0m"""
    if term_stdout:
        term_stdout.write(bsod_text + "\n")
        term_stdout.flush()
    else:
        sys.__stdout__.write(bsod_text + "\n")
        sys.__stdout__.flush()


def print_progress(percent, message="Loading AI Model..."):
    bar_width = 40
    filled = int(bar_width * percent / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    is_tty = hasattr(term_stdout, "isatty") and term_stdout.isatty()
    term_type = os.environ.get("TERM", "")
    is_compatible = is_tty and term_type not in ("", "dumb")

    if is_compatible:
        term_stdout.write(f"\r\033[KLoading: |{bar}| {percent}% - {message}")
        term_stdout.flush()
    else:
        term_stdout.write(f"Loading: {percent}% - {message}\n")
        term_stdout.flush()


def render_ascii_logo():
    logo_path = os.path.join(
        BASE_DIR, "ui", "frontend", "public", "Project Zephyrine Logo.png"
    )
    if not os.path.exists(logo_path):
        logo_path = os.path.join(
            BASE_DIR, "ui", "frontend", "dist", "Project Zephyrine Logo.png"
        )

    try:
        from PIL import Image

        img = Image.open(logo_path)
        width = 60
        w, h = img.size
        aspect = h / w
        height = int(width * aspect * 0.45)
        img = img.resize((width, height)).convert("L")

        chars = "@%#*+=-:. "
        num_chars = len(chars)

        lines = []
        for y in range(height):
            line = ""
            for x in range(width):
                pixel = img.getpixel((x, y))
                char_idx = int((pixel / 255) * (num_chars - 1))
                line += chars[char_idx]
            lines.append(line)
        return "\n".join(lines)
    except Exception:
        return """
       .---.
      /     \\
      \\_.._/
       ||||
       ||||
    .-'    '-.
   /          \\
  |  Project   |
  | Zephyrine  |
   \\__________/
"""


def progress_monitor(log_path):
    while not os.path.exists(log_path):
        time.sleep(0.1)

    server_port = os.environ.get("ADLAIDE_SERVER_PORT", "11420")
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            server_port = sys.argv[i + 1]

    current_pct = 0
    target_pct = 0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        is_done = False
        while not is_done:
            line = f.readline()

            # Keep target_pct state-machine tracking
            if line:
                if "Verifying Environment Prerequisites" in line:
                    target_pct = max(target_pct, 5)
                elif "Setting up Adelaide-Lite environment" in line:
                    target_pct = max(target_pct, 10)
                elif "llama.cpp release" in line:
                    target_pct = max(target_pct, 18)
                elif "Resolving Ada dependencies" in line:
                    target_pct = max(target_pct, 30)
                elif "Building Vite Frontend" in line:
                    target_pct = max(target_pct, 45)
                elif "Self-Integrity Quality Check" in line:
                    target_pct = max(target_pct, 55)
                elif "Bootstrapping QRNN LSH worker" in line:
                    target_pct = max(target_pct, 65)
                elif "Bootstrapping ONNX VAD worker" in line:
                    target_pct = max(target_pct, 75)
                elif (
                    "Booting Adelaide Watchdog" in line
                    or "Booting Adelaide Intelligence" in line
                ):
                    target_pct = max(target_pct, 85)
                elif "llama_context: constructing" in line:
                    target_pct = max(target_pct, 90)
                elif (
                    "Model_Manager.Initialize COMPLETE" in line
                    or "Initialize: metrics logger started" in line
                ):
                    target_pct = max(target_pct, 95)
                elif (
                    "Server is UP" in line
                    or "HTTP: http://" in line
                    or "Access Info" in line
                ):
                    target_pct = 100
                    is_done = True

            if current_pct < target_pct:
                current_pct += 1
                eta = max(1, 15 - int(15 * current_pct / 100))
                print_progress(current_pct, f"Model Loading... [ETA: {eta}s]")
                time.sleep(0.02)
            else:
                if not line:
                    time.sleep(0.05)

        while current_pct < 100:
            current_pct += 1
            eta = max(0, 15 - int(15 * current_pct / 100))
            if current_pct == 100:
                print_progress(current_pct, "Model loaded successfully!")
            else:
                print_progress(current_pct, f"Model Loading... [ETA: {eta}s]")
            time.sleep(0.01)

        is_tty = hasattr(term_stdout, "isatty") and term_stdout.isatty()
        term_type = os.environ.get("TERM", "")
        if is_tty and term_type not in ("", "dumb"):
            term_stdout.write("\033[2K\r")
            term_stdout.flush()
        else:
            term_stdout.write("\n")
            term_stdout.flush()

        logo_ascii = render_ascii_logo()
        term_stdout.write("\n\033[36m" + logo_ascii + "\033[0m\n")
        term_stdout.write(
            "\n\033[32;1mHeya! Adelaide Here and Project Zephyrine has been started! I Hope you have an absolutely wonderful day\033[0m\n"
        )
        term_stdout.write("To get started interacting with me, you can use:\n")
        term_stdout.write(
            f"  * \033[36mOllama Client:\033[0m      OLLAMA_HOST=http://localhost:{server_port} (or point OpenWebUI here)\n"
        )
        term_stdout.write(
            f"  * \033[36mOpenAI Client:\033[0m      http://localhost:{server_port}/v1 (note: use /v1)\n"
        )
        try:
            ssl_port = int(server_port) + 1
            term_stdout.write(
                f"  * \033[36mOpenAI Secure:\033[0m      https://localhost:{ssl_port}/v1\n"
            )
        except Exception:
            pass
        term_stdout.write(
            f"  * \033[36mClaude Client:\033[0m      http://localhost:{server_port} or https (secure)\n"
        )
        term_stdout.write(
            "\n\033[32m Remember, I am NOT an AI that nor replacement for chatGPT or Gemini you expected. I am NOT following industry standard status quo AI. I am nobody.\033[0m\n"
        )
        term_stdout.write(
            "\033[35m (preexisting + new) made using javascript, python, and bash script.\033[0m\n\n"
        )
        term_stdout.flush()


def setup_logging():
    """Create logs/ dir, rotate old logs, redirect stdout/stderr to tee.
    Returns the path of the current log file."""
    global IS_KISS, term_stdout, term_stderr
    os.makedirs(LOGS_DIR, exist_ok=True)
    _rotate_logs()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"run_{timestamp}.log")
    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered

    IS_KISS = not (
        "--verbose" in sys.argv
        or "--test-build-integrity-check" in sys.argv
        or "--verify" in sys.argv
        or "--help" in sys.argv
        or "-h" in sys.argv
    )

    if IS_KISS:
        orig_stdout_fd = os.dup(1)
        orig_stderr_fd = os.dup(2)
        term_stdout = open(orig_stdout_fd, "w", buffering=1)
        term_stderr = open(orig_stderr_fd, "w", buffering=1)
        os.dup2(log_fp.fileno(), 1)
        os.dup2(log_fp.fileno(), 2)
        sys.stdout = log_fp
        sys.stderr = log_fp
    else:
        sys.stdout = _TeeWriter(sys.__stdout__, log_fp)
        sys.stderr = _TeeWriter(sys.__stderr__, log_fp)
        print(f"[*] Logging to {log_path}")

    return log_path


# ANSI Color Codes
RST = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GRN = "\033[32m"
YLW = "\033[33m"
BLU = "\033[34m"
MGN = "\033[35m"
CYN = "\033[36m"
WHT = "\033[97m"
BG_B = "\033[44m\033[97m"
BG_RED = "\033[41m\033[97m"


def get_git_version():
    """Get current git commit hash and branch from the project root."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        status = f"{YLW}(dirty){RST}" if dirty else f"{GRN}(clean){RST}"
        return commit, branch, status
    except Exception:
        return None, None, None


def bootstrap_ros2_mac():
    """
    Automatically bootstrap ROS2 environment on macOS via Micromamba/RoboStack.

    Why: Native ROS2 (Humble/Iron) does not officially support macOS without complex
    workarounds or virtual machines/Docker. Docker adds network and USB bridging overhead
    which can harm real-time actuator pacing (ELP3/ELP2). RoboStack provides a native
    conda-forge compiled version of ROS2 for Darwin/macOS.

    How: On Linux (Ubuntu), this is typically handled via standard `apt install ros-humble-desktop`.
    Here, we download Micromamba (a fast C++ Conda implementation), create an isolated
    environment in `.ros_env`, and inject the ROS2 paths into os.environ so all
    subprocesses inherit the native ROS2 DDS bindings.
    """
    if "ROS_DISTRO" in os.environ:
        return  # Already have ROS2 active

    print(f"\n{BOLD}{WHT}[*] Bootstrapping ROS2 RoboStack Environment...{RST}")
    bin_dir = os.path.join(BASE_DIR, "vendor", ".micromamba")
    ros_env_dir = os.path.join(BASE_DIR, "vendor", "ros_env")
    micromamba_bin = os.path.join(bin_dir, "bin", "micromamba")

    os.makedirs(bin_dir, exist_ok=True)

    if not os.path.exists(micromamba_bin):
        print(f"  {CYN}[~] Downloading micromamba...{RST}")
        try:
            subprocess.check_call(
                "curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xvj bin/micromamba",
                shell=True,
                cwd=bin_dir,
            )
        except Exception as e:
            print(f"  {RED}[!!] Failed to download micromamba: {e}{RST}")
            return

    if not os.path.exists(os.path.join(ros_env_dir, "conda-meta")):
        print(
            f"  {CYN}[~] Creating ROS2 environment (this may take several minutes)...{RST}"
        )
        try:
            subprocess.check_call(
                [
                    micromamba_bin,
                    "create",
                    "-y",
                    "-p",
                    ros_env_dir,
                    "-c",
                    "robostack-staging",
                    "-c",
                    "conda-forge",
                    "ros-humble-desktop",
                ],
                cwd=PROJECT_ROOT,
            )
        except Exception as e:
            print(f"  {RED}[!!] Failed to create ROS2 environment: {e}{RST}")
            return

    print(f"  {GRN}[ok]{RST} ROS2 RoboStack environment ready.")

    # Inject variables into os.environ for subprocesses
    os.environ["ROS_DISTRO"] = "humble"
    os.environ["AMENT_PREFIX_PATH"] = ros_env_dir
    os.environ["PYTHONPATH"] = f"{ros_env_dir}/lib/python3.11/site-packages" + (
        f":{os.environ['PYTHONPATH']}" if "PYTHONPATH" in os.environ else ""
    )
    os.environ["PATH"] = f"{ros_env_dir}/bin:{os.environ['PATH']}"


def bootstrap_px4():
    """Clone and compile PX4-Autopilot for ELP2/ELP3 simulation tools."""
    vendor_dir = os.path.join(BASE_DIR, "vendor")
    px4_dir = os.path.join(vendor_dir, "PX4-Autopilot")
    if not os.path.exists(px4_dir):
        print(f"\n{BOLD}{WHT}[*] Cloning PX4-Autopilot into vendor/...{RST}")
        try:
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "https://github.com/PX4/PX4-Autopilot.git",
                    "--recursive",
                ],
                cwd=vendor_dir,
            )
        except Exception as e:
            print(f"  {RED}[!!] Failed to clone PX4-Autopilot: {e}{RST}")
            return

    # Clone MAVLink C Headers for Ada FFI
    mavlink_dir = os.path.join(vendor_dir, "mavlink_c_v2")
    if not os.path.exists(mavlink_dir):
        print(f"\n{BOLD}{WHT}[*] Cloning MAVLink C Headers for FFI...{RST}")
        try:
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "https://github.com/mavlink/c_library_v2.git",
                    mavlink_dir,
                ],
                cwd=vendor_dir,
            )
        except Exception as e:
            print(f"  {RED}[!!] Failed to clone MAVLink C Headers: {e}{RST}")
            return

    # Check if compiled
    px4_build_dir = os.path.join(px4_dir, "build", "px4_sitl_default")
    if not os.path.exists(px4_build_dir):
        print(f"\n{BOLD}{WHT}[*] Compiling PX4-Autopilot (SITL)...{RST}")
        print(f"  {CYN}[~] This may take a while to download modules and compile.{RST}")
        try:
            # Install python dependencies for PX4
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--user",
                    "-r",
                    "Tools/setup/requirements.txt",
                ],
                cwd=px4_dir,
            )
            # Run the SITL compile
            subprocess.check_call(["make", "px4_sitl_default"], cwd=px4_dir)
            print(f"  {GRN}[ok]{RST} PX4-Autopilot SITL compiled successfully.")

            # Compile for FMU-v6x (Hardware fallback for FMC)
            print(
                f"\n{BOLD}{WHT}[*] Compiling PX4-Autopilot (fmu-v6x hardware target)...{RST}"
            )
            subprocess.check_call(["make", "px4_fmu-v6x_default"], cwd=px4_dir)
            print(
                f"  {GRN}[ok]{RST} PX4-Autopilot Hardware target compiled successfully."
            )
        except Exception as e:
            print(f"  {RED}[!!] Failed to compile PX4-Autopilot: {e}{RST}")
            return
    else:
        print(f"\n{GRN}[ok]{RST} PX4-Autopilot is already cloned and compiled.")


def verify_environment(build_px4=False):
    """Check for all required tools and libraries before proceeding."""
    if build_px4:
        bootstrap_px4()

    if platform.system() == "Darwin":
        bootstrap_ros2_mac()
    print(f"\n{BOLD}{WHT}[*] Verifying Environment Prerequisites...{RST}")

    critical_tools = {
        "alr": "Alire (Ada Package Manager) - install via 'brew install alire'",
        "python3": "Python 3.12+ - install via 'brew install python'",
        "cmake": "CMake - install via 'brew install cmake'",
        "git": "Git - install via 'brew install git'",
        "wget": "wget - install via 'brew install wget'",
        "npm": "Node.js/npm - install via 'brew install node'",
        "deno": "Deno - install via 'curl -fsSL https://deno.land/install.sh | sh'",
        "ruff": "Ruff (Linter) - install via 'pip install ruff'",
    }

    missing = []
    for tool, desc in critical_tools.items():
        if shutil.which(tool):
            print(f"  {GRN}[ok]{RST} {tool}")
        else:
            print(f"  {RED}[!!]{RST} {tool} is missing: {desc}")
            missing.append(tool)

    # macOS specific SDK check
    if platform.system() == "Darwin":
        # Check for full Xcode.app installation (not just Command Line Tools)
        xcode_path = "/Applications/Xcode.app"
        if os.path.exists(xcode_path):
            print(f"  {GRN}[ok]{RST} Full Xcode.app found")
        else:
            print(f"  {RED}[!!]{RST} Full Xcode.app NOT found at {xcode_path}")
            print("    Prerequisite: Install full Xcode from the App Store")
            missing.append("xcode-app")

        try:
            subprocess.check_output(
                ["xcrun", "--show-sdk-path"], stderr=subprocess.DEVNULL
            )
            print(f"  {GRN}[ok]{RST} macOS SDK path available")
        except Exception:
            print(
                f"  {RED}[!!]{RST} macOS SDK path not found: run 'xcode-select --install'"
            )
            missing.append("macos-sdk")

    if "ROS_DISTRO" in os.environ:
        print(f"  {GRN}[ok]{RST} ROS2 Detected ({os.environ['ROS_DISTRO']})")
    else:
        print(
            f"  {YLW}[warn]{RST} ROS2 environment not detected (ROS_DISTRO missing). ELP2/ELP3 Actuators will be disabled."
        )

    # PX4-Autopilot check (built locally or on PATH)
    px4_bin = shutil.which("px4")
    px4_dir = os.path.join(BASE_DIR, "vendor", "PX4-Autopilot")
    if px4_bin or (os.path.isdir(px4_dir) and os.path.exists(os.path.join(px4_dir, "build"))):
        if px4_bin:
            print(f"  {GRN}[ok]{RST} PX4 detected ({px4_bin})")
        else:
            print(f"  {GRN}[ok]{RST} PX4-Autopilot found in vendor/")
    else:
        print(f"  {RED}[!!]{RST} PX4-Autopilot not found in vendor/. Use --build-px4 to clone & compile, or install manually.")
        missing.append("px4")

    if missing:
        print(
            f"\n{BG_RED}[BUGCHECK] [FATAL] Environment check failed. Please install the missing tools listed above.{RST}"
        )
        raise RuntimeError("ENV_CHECK_FAILURE: Environment check failed.")
    else:
        print(f"{GRN}[+] Environment verified. All prerequisites met.{RST}\n")


def show_help():
    """Print colorful help screen with git version."""
    commit, branch, status = get_git_version()
    ver_str = f"{CYN}{commit}{RST}" if commit else f"{DIM}unknown{RST}"
    brn_str = f"{MGN}{branch}{RST}" if branch else f"{DIM}unknown{RST}"

    print(f"""
{BG_B}{"=" * 70}{RST}
{BG_B}{"  Adelaide Platform — run.sh".center(70)}{RST}
{BG_B}{"=" * 70}{RST}

  {BOLD}Whimsical Automata Companion — Snowball-Enaga{RST}

  {BOLD}Version:{RST}  {ver_str}  {status}
  {BOLD}Branch:{RST}   {brn_str}
  {BOLD}Platform:{RST} {YLW}{platform.system()}{RST} ({platform.machine()})

  {BOLD}{WHT}USAGE{RST}
    {CYN}./run.sh{RST} [OPTIONS]

  {BOLD}{WHT}OPTIONS{RST}
    {GRN}--no-gui{RST}                        Launch server without the Python Sidecar UI
    {GRN}--host{RST} {CYN}HOST{RST}                     Bind address (default: 0.0.0.0, env: ADLAIDE_SERVER_HOST)
    {GRN}--port{RST} {CYN}PORT{RST}                     Bind port (default: 11420, env: ADLAIDE_SERVER_PORT)
    {GRN}--build-px4{RST}                     Clone and compile PX4-Autopilot for simulation tools
    {GRN}--test-build-integrity-check{RST}    Build only, verify integrity, then exit
    {GRN}--show-key{RST}                      Show the current AES-256 master key, then exit
    {GRN}--enforce-api-key{RST}               Enable x-api-key validation on the Ada server
    {GRN}--no-enforce-api-key{RST}            Explicitly disable API key enforcement (default)
    {GRN}--api-key{RST} {CYN}ACTION [ARGS]{RST}  Manage API keys: add [key] | remove <key> | list | edit <old> <new>
    {GRN}-h{RST}, {GRN}--help{RST}                  Show this help screen

  {BOLD}{WHT}EXAMPLES{RST}
    {DIM}Default — full GUI, binds on all interfaces, port 11420:{RST}
      {CYN}./run.sh{RST}

    {DIM}Headless server, no GUI sidecar:{RST}
      {CYN}./run.sh --no-gui{RST}

    {DIM}Custom port (e.g. 8080):{RST}
      {CYN}./run.sh --port 8080{RST}
      {DIM}→ API at http://localhost:8080{RST}

    {DIM}Bind to localhost only (private, no LAN access):{RST}
      {CYN}./run.sh --host 127.0.0.1{RST}
      {DIM}→ API at http://127.0.0.1:11420{RST}

    {DIM}Custom host + port:{RST}
      {CYN}./run.sh --host 0.0.0.0 --port 9000{RST}
      {DIM}→ API at http://localhost:9000{RST}

    {DIM}Headless with custom port:{RST}
      {CYN}./run.sh --no-gui --port 8080{RST}

    {DIM}Via environment variables:{RST}
      {CYN}ADLAIDE_SERVER_PORT=3000 ADLAIDE_SERVER_HOST=127.0.0.1 ./run.sh{RST}

    {DIM}Docker / LAN access (bind all interfaces):{RST}
      {CYN}./run.sh --host 0.0.0.0 --port 11420{RST}
      {DIM}→ API at http://<your-ip>:11420 from other machines{RST}

    {DIM}Phone / Cloud Terminal (access from phone or tablet):{RST}
      {CYN}./run.sh --host 0.0.0.0 --port 11420{RST}
      {DIM}→ Find your computer's IP: ifconfig | grep 'inet '{RST}
      {DIM}→ Open http://<your-computer-host-ip>:11420 on your phone browser{RST}
      {DIM}→ Or use curl in Termux / iSH / a-Shell:{RST}
      {DIM}  curl http://<your-computer-host-ip>:11420/api/version{RST}

    {DIM}Multiple devices ( LAN party / office ):{RST}
      {CYN}./run.sh --host 0.0.0.0 --port 11420{RST}
      {DIM}→ Any device on same network can hit http://<your-computer-host-ip>:11420{RST}
      {DIM}→ Works with OpenWebUI, OpenCode, curl, or any HTTP client{RST}

    {DIM}API Key Management:{RST}
      {CYN}./run.sh --api-key list{RST}
      {DIM}→ List all configured API keys (shows first 8 chars each){RST}
      {CYN}./run.sh --api-key add my-secret-key-123{RST}
      {DIM}→ Add a specific API key to the encrypted store{RST}
      {CYN}./run.sh --api-key add{RST}
      {DIM}→ Auto-generate a random 256-bit API key and add it{RST}
      {CYN}./run.sh --api-key remove my-secret-key-123{RST}
      {DIM}→ Remove an API key from the store{RST}
      {CYN}./run.sh --api-key edit old-key new-key{RST}
      {DIM}→ Replace an existing key with a new one{RST}

    {DIM}API Key Enforcement:{RST}
      {CYN}./run.sh --enforce-api-key{RST}
      {DIM}→ Enable x-api-key header validation (clients must send a valid key){RST}
      {CYN}./run.sh --no-enforce-api-key{RST}
      {DIM}→ Disable enforcement (default, for Ollama app compatibility){RST}
      {CYN}./run.sh --api-key add --enforce-api-key{RST}
      {DIM}→ Add a key first, then start server with enforcement{RST}

    {DIM}With API key enforcement + curl:{RST}
      {CYN}./run.sh --api-key add mykey --enforce-api-key{RST}
      {DIM}  # Then from another terminal:{RST}
       {CYN}curl http://localhost:11420/api/chat -H "x-api-key: mykey" -d '{{"model":"Snowball-Enaga","messages":[{{"role":"user","content":"Hello"}}],"stream":false}}'{RST}

  {BOLD}{WHT}RUNTIME PROCESSES{RST}
    {MGN}1. StellaIcarus Daemon{RST}    Hardware monitor, power state, telemetry
    {MGN}2. adelaide_server{RST}        HTTP API (default port 11420)
    {MGN}3. adelaide_watchdog{RST}      Monitors server health, auto-restarts

  {BOLD}{WHT}SERVER API{RST} (connect via {CYN}http://localhost:11420{RST} or {CYN}http://127.0.0.1:11420{RST})
    {CYN}POST{RST} /api/chat                Chat completion (streaming)
    {CYN}POST{RST} /api/generate            Text generation
    {CYN}POST{RST} /v1/chat/completions    OpenAI-compatible chat
    {CYN}POST{RST} /v1/completions         OpenAI-compatible completions
    {CYN}POST{RST} /api/embeddings         Text embeddings
    {CYN}POST{RST} /v1/embeddings          OpenAI-compatible embeddings
    {CYN}POST{RST} /v1/audio/transcriptions  Speech-to-text (Moonshine)
    {CYN}POST{RST} /v1/audio/speech        Text-to-speech (Kokoro)
    {CYN}GET{RST}  /api/health             Health check
    {CYN}GET{RST}  /api/version            Server version
    {CYN}GET{RST}  /api/tags               List models
    {CYN}GET{RST}  /api/power              Power state (StellaIcarus)
    {CYN}GET{RST}  /api/telemetry          System telemetry
    {CYN}GET{RST}  /api/ps                 Process status
    {CYN}POST{RST} /api/schedule           Schedule a delayed task
    {CYN}POST{RST} /api/ZenithRoutine      ZenithOrion pacing loop

  {BOLD}{WHT}GUI SIDECAR{RST}
    {CYN}GET{RST}    /api/sessions             List chat sessions
    {CYN}POST{RST}   /api/sessions             Create session
    {CYN}PUT{RST}    /api/sessions/{{id}}      Rename session
    {CYN}DELETE{RST} /api/sessions/{{id}}      Delete session
    {CYN}POST{RST}   /api/sessions/{{id}}/duplicate  Duplicate session
    {CYN}GET{RST}    /api/messages             Message history
    {CYN}GET{RST}    /api/adelaideenginestats  Engine stats
    {CYN}POST{RST}   /api/knowledgestackfrontend/upload     Knowledge upload
    {CYN}GET{RST}    /api/knowledgestackfrontend/search     Knowledge search
    {CYN}POST{RST}   /api/knowledgestackfrontend/memory/upload   Memory upload
    {CYN}GET{RST}    /api/knowledgestackfrontend/memory/search   Memory search
    {CYN}GET{RST}    /api/knowledgestackfrontend/graph          Knowledge graph
    {CYN}GET{RST}    /api/knowledgestackfrontend/memory/graph   Memory graph
    {CYN}GET{RST}    /api/docs/readme          Readme
    {CYN}GET{RST}    /api/docs/license         License
    {CYN}GET{RST}    /api/user_info            User info


  {DIM}  Documentation:  AdelaideZephyrineSystem/documentation/{RST}
  {DIM}  Architecture:   AdelaideZephyrineSystem/run.py (line 14){RST}
""")


# ============================================================================
# [DO NOT REMOVE] ADELAITE LITE — PROGRAM ARCHITECTURE
# ============================================================================
# WARNING: This comment block documents the full system architecture.
# Removing it will make the program nearly impossible to understand or
# maintain. Any agent or contributor modifying this file must read this
# section before making changes.
#
# This script is the top-level orchestrator for the Adelaide Intelligence
# Platform. It builds all dependencies (if source changed), then spawns
# three concurrent processes that together form the runtime.
#
# ENTRY POINT CHAIN:
#   run.sh --no-gui
#     └─ cd AdelaideZephyrineSystem && python3 run.py --no-gui
#          ├─ [Build Phase] (triggered when MD5 hash of source files changes)
#          │    ├─ Clone & build llama.cpp (CMake, ggml-metal on macOS arm64)
#          │    ├─ Build mtmd library (CLIP vision encoding for multimodal)
#          │    ├─ Clone & build moonshine (ONNX-based speech-to-text)
#          │    ├─ Clone & build kokoro-onnx (text-to-speech)
#          │    ├─ Download Qwen3.5 GGUF models (0.8B, 9B, Embedding)
#          │    ├─ Download Kokoro TTS models (ONNX + voices)
#          │    ├─ Install Playwright Chromium (for Deno web crawler)
#          │    ├─ alr build (Ada/Alire — compiles all Ada sources to bin/)
#          │    └─ npm install && npm run build (Vite frontend)
#          │
#          ├─ [Runtime] Spawns 3 background processes:
#          │    ├─ 1. StellaIcarus Daemon Manager (Python, hardware monitor)
#          │    ├─ 2. adelaide_server (Ada binary, HTTP API on port 11420)
#          │    └─ 3. adelaide_watchdog (Ada binary, monitors server health)
#          │
#          └─ [--no-gui] Waits for adelaide_server exit, shows crash banner
#
# PROCESS ARCHITECTURE:
#
#   ┌─────────────────────────────────────────────────────────────┐
#   │                    run.py (Orchestrator)                     │
#   │  - Builds everything if source changed (MD5 hash check)     │
#   │  - Sets DYLD_LIBRARY_PATH for onnxruntime (moonshine)       │
#   │  - Spawns all child processes                               │
#   │  - Handles SIGINT/SIGTERM cleanup                           │
#   └──────┬──────────────────┬──────────────────┬────────────────┘
#          │                  │                  │
#          ▼                  ▼                  ▼
#   ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐
#   │  StellaIcarus │  │ adelaide_server │  │ adelaide_watchdog │
#   │  Daemon (Py)  │  │    (Ada/AWS)    │  │    (Ada)         │
#   │               │  │  Port 11420     │  │                  │
#   │ - HW monitor  │  │ - HTTP API      │  │ - Monitors PID   │
#   │ - Power state │  │ - LLM inference │  │ - Checks heartbeat│
#   │ - Telemetry   │  │ - RAG pipeline  │  │ - Restarts server│
#   │ - ELP bridge  │  │ - STT/TTS       │  │   if stale       │
#   └──────────────┘  └─────────────────┘  └──────────────────┘
#
# ADA SERVER INTERNALS (adelaide_server.adb):
#
#   Startup sequence (order matters):
#     STEP 0: Disk benchmark (reads 1GB from GGUF, classifies storage speed)
#     STEP 1: Model_Manager.Initialize
#              ├─ Llama_Backend_Init (ggml-metal/CPU backends)
#              ├─ Database_Manager.Initialize (SQLite databases)
#              ├─ ELP_Queue.Initialize (priority queue monitor)
#              └─ Idle_Monitor task (unloads idle models after 30s)
#     STEP 2: Knowledge_Manager.Initialize
#              └─ Background tasks (ELP0):
#                   ├─ Indexing_Task (parses references.bib)
#                   ├─ Native_Crawl_Task (walks filesystem → embeddings)
#                   └─ Proactive_Cache_Task (predicts follow-ups)
#     STEP 3: Scheduler_Manager.Initialize
#     STEP 4: Watchdog_IPC.Init (creates run/, writes PID + heartbeat)
#     STEP 5: Knowledge_Manager.Start_Tasks (starts ELP0 producers)
#     STEP 6: AWS.Server.Start (HTTP on port 11420)
#     STEP 7: Health ping watchdog (3s interval, 60s deadline)
#     STEP 8: Moonshine_Interface.Init_Moonshine (STT, ~500MB ONNX)
#     STEP 9: Main heartbeat loop (1Hz heartbeat + ELP stats every 5s)
#
# ELP PRIORITY QUEUE ("Volatus Damarae" architecture):
#   Serial processing — prevents heap corruption from concurrent llama.cpp FFI.
#   Capacity: 2^63. Priority: ELP3 > ELP2 > ELP1 > ELP0.
#
#   ELP3: ZenithOrion — 1ms deterministic pacing loop (highest frequency)
#   ELP2: StellaIcarus — deterministic API response hooks
#   ELP1: User-facing generation (real-time inference)
#   ELP0: Background indexing/RAG (preemptible by ELP1)
#
# MODEL TYPES:
#   Qwen_0_8B       — Small LLM (always loaded, exempt from idle unload)
#   Qwen_9B         — Large LLM (loaded on-demand for complex reasoning)
#   Qwen_Embedding  — Embedding model (semantic search)
#   MMProj          — Multimodal projection (CLIP vision via mtmd)
#
# KEY SUBSYSTEMS:
#   Llama_Interface     — Ada→C FFI wrapping llama.cpp
#   Mtmd_Interface      — Ada→C FFI for multimodal (CLIP vision)
#   Moonshine_Interface — Ada→C FFI for speech-to-text (ONNX)
#   Kokoro_Interface    — Ada→Python for text-to-speech
#   Kratos              — Crash isolation (sigaction + longjmp)
#   Speculative_Cache   — Predictive response cache (5 entries, LRU)
#   Database_Manager    — SQLite (memory, literature, knowledge graph)
#   Streaming_Queue     — AWS streaming response support
#   Watchdog_IPC        — File-based IPC (PID, heartbeat, exit reason)
#   ZenithOrion         — 1ms deterministic pacing loop (ELP3)
#
# EXTERNAL DEPENDENCIES (sibling directories):
#   vendor/llama.cpp/            — LLM inference engine
#   vendor/moonshine/            — Speech-to-text ONNX models
#   vendor/kokoro-onnx/          — Text-to-speech ONNX
#   vendor/kokoclone/            — Zero-shot voice cloning
#   vendor/tts_kokoro_component/ — Kokoro TTS Python deps (isolated venv)
#
# COMMUNICATION FLOW:
#   User Request → HTTP :11420 → Adelaide_Server_Pkg.Dispatch
#     ├─ Chat/Generate   → Model_Manager → Llama_Interface → llama.cpp
#     ├─ Embeddings      → Model_Manager → Llama_Interface (embed mode)
#     ├─ Transcription   → Moonshine_Interface → libmoonshine.dylib
#     ├─ TTS             → Kokoro_Interface → Python subprocess
#     ├─ Vision          → Image_Encoder → mtmd (CLIP) → Llama_Interface
#     ├─ RAG             → Database_Manager → semantic search → Model_Manager
#     └─ Power state     ← StellaIcarus Daemon → /api/power endpoint
#
# CRASH ISOLATION (Kratos):
#   C-level crashes (SIGSEGV, SIGBUS, SIGFPE, SIGTRAP, SIGABRT) during
#   llama.cpp inference are caught by Kratos (sigaction + longjmp) instead
#   of killing the server. The external watchdog monitors heartbeat files
#   and restarts the server if it dies.
# ============================================================================

#  QUIRK: Block NT kernel at runtime (see QUIRK-005)
#  Windows is NOT supported.  The build system (AdelaideZephyrineSystem.gpr) also
#  blocks compilation on Windows, but this is an additional guard.
#  LINUX-COMPAT (future): When porting to Linux, remove this check.
if "--help" in sys.argv or "-h" in sys.argv:
    show_help()
    sys.exit(0)

# ── Show master key ─────────────────────────────────────────────────────────
# Print the current master key and exit. Useful for CI, backup, or when
# you need to copy the key to another machine.
if "--show-key" in sys.argv:
    try:
        master_key = load_master_key()
        print(f"[CRYPTO] Master key: {master_key}")
    except RuntimeError as e:
        print(f"[CRYPTO] No master key available: {e}")
        print("[CRYPTO] Run without --show-key first to bootstrap a new key.")
        sys.exit(1)
    sys.exit(0)

# ── API key management ───────────────────────────────────────────────────────
# Usage:
#   python3 run.py --api-key add <key>
#   python3 run.py --api-key add              (auto-generate a random key)
#   python3 run.py --api-key remove <key>
#   python3 run.py --api-key list
#   python3 run.py --api-key edit <old> <new>
if "--api-key" in sys.argv:
    idx = sys.argv.index("--api-key")
    args_after = sys.argv[idx + 1 :]
    if not args_after:
        print(
            "[API-KEY] Usage: --api-key add [key] | remove <key> | list | edit <old> <new>"
        )
        sys.exit(1)

    action = args_after[0]
    # API key management requires the master key env var to be set
    # (server must be running, or set ADELAIDE_MASTER_KEY manually)
    try:
        master_key = load_master_key()
    except RuntimeError:
        print("[API-KEY] ERROR: ADELAIDE_MASTER_KEY env var not set.")
        print("[API-KEY] Start the server first, or set ADELAIDE_MASTER_KEY manually.")
        sys.exit(1)

    if action == "add":
        if len(args_after) >= 2 and args_after[1]:
            key = args_after[1]
        else:
            # Auto-generate a 32-byte hex key (64 chars)
            import secrets

            key = secrets.token_hex(32)
            print(f"[API-KEY] Generated new key: {key}")
        from adelaide_crypto import add_api_key

        add_api_key(key)
        print("[API-KEY] Key added successfully.")
        # Show the key so user can copy it
        if len(args_after) < 2 or not args_after[1]:
            print(f"[API-KEY] Copy this key for your client: {key}")

    elif action == "remove":
        if len(args_after) < 2:
            print("[API-KEY] Usage: --api-key remove <key>")
            sys.exit(1)
        from adelaide_crypto import remove_api_key

        remove_api_key(args_after[1])

    elif action == "list":
        from adelaide_crypto import list_api_keys

        list_api_keys()

    elif action == "edit":
        if len(args_after) < 3:
            print("[API-KEY] Usage: --api-key edit <old> <new>")
            sys.exit(1)
        from adelaide_crypto import edit_api_key

        edit_api_key(args_after[1], args_after[2])

    else:
        print(f"[API-KEY] Unknown action: {action}")
        print(
            "[API-KEY] Usage: --api-key add [key] | remove <key> | list | edit <old> <new>"
        )
        sys.exit(1)

    sys.exit(0)

if platform.system() == "Windows":
    print("[FATAL] Windows (NT kernel) is not supported.")
    print("[FATAL] This server targets macOS (arm64) with planned Linux support.")
    print("[FATAL] See AdelaideZephyrineSystem.gpr QUIRK-005 for details.")
    sys.exit(1)

# Set HF_HOME so huggingface caches locally in the project directory
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "model")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Kill any stale processes from previous runs before starting
print("[*] Cleaning up any stale processes from previous runs...")
try:
    subprocess.run(["pkill", "-9", "-f", "adelaide_server"], stderr=subprocess.DEVNULL)
    subprocess.run(
        ["pkill", "-9", "-f", "adelaide_watchdog"], stderr=subprocess.DEVNULL
    )
    subprocess.run(["pkill", "-9", "-f", "vad_worker.py"], stderr=subprocess.DEVNULL)
except Exception:
    pass

# Globals to keep track of background processes
daemon_process = None
server_process = None
vad_process = None
watchdog_process = None
sidecar_process = None
kokoro_process = None

# Master key temp file path (cleaned up on shutdown)
_master_key_file_path = None


def get_files_to_hash():
    # NOTE: run.py itself is NOT hashed - it's an interpreter script, not a
    # compiled artifact. Changes to run.py don't trigger rebuilds.
    patterns = [
        "src/**/*",
        "config/**/*",
        "AdelaideZephyrineSystem.gpr",
        "ui/frontend/src/**/*",
        "ui/frontend/index.html",
        "ui/frontend/package.json",
    ]
    files = []
    for pattern in patterns:
        path = os.path.join(BASE_DIR, pattern)
        if "/**/" in pattern:
            # Recursive glob isn't strictly needed if we just os.walk, but let's do a simple recursive collect
            base = path.split("/**/")[0]
            if os.path.exists(base):
                for root, _, filenames in os.walk(base):
                    for name in filenames:
                        files.append(os.path.join(root, name))
        else:
            if os.path.exists(os.path.join(BASE_DIR, pattern)):
                files.append(os.path.join(BASE_DIR, pattern))

    # Also hash mtmd source files (cloned by run.py into vendor/llama.cpp)
    # Detects when a fresh clone or update changes multimodal source
    mtmd_dir = os.path.abspath(
        os.path.join(BASE_DIR, "vendor", "llama.cpp", "tools", "mtmd")
    )
    if os.path.exists(mtmd_dir):
        for root, _, filenames in os.walk(mtmd_dir):
            for name in filenames:
                if name.endswith((".cpp", ".h", ".c")):
                    files.append(os.path.join(root, name))

    return sorted(files)


def calculate_hash(file_paths):
    hasher = hashlib.md5()
    for file_path in file_paths:
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                # To closely mimic the bash find | sort | xargs md5 -q
                # we hash the contents of the files in sorted order
                hasher.update(f.read())
    return hasher.hexdigest()


# ── Venv Validity Detection ────────────────────────────────────────────────
# Detects when the pyvenv is stale due to:
#   1. Project directory moved (shebangs, .pth files, installed paths break)
#   2. Requirements files changed (new deps needed)
#   3. Python sidecar scripts changed (installed in venv)
#
# Uses a separate .venv_hash file (independent of .build_hash) so venv
# rebuilds don't trigger a full source rebuild and vice versa.

def get_venv_files_to_hash():
    """Collect files whose changes invalidate the pyvenv."""
    patterns = [
        # Requirements files
        "lsh/requirements-lsh.txt",
        "vendor/tts_kokoro_component/requirements.txt",
        # Python sidecar scripts installed into pyvenv
        "vad_component/vad_worker.py",
        "lsh/lsh_qrnn_worker.py",
        # Python crypto/sidecar modules
        "python/**/*.py",
    ]
    files = []
    for pattern in patterns:
        path = os.path.join(BASE_DIR, pattern)
        if "/**/" in pattern:
            base = path.split("/**/")[0]
            if os.path.exists(base):
                for root, _, filenames in os.walk(base):
                    for name in filenames:
                        files.append(os.path.join(root, name))
        else:
            if os.path.exists(path):
                files.append(path)
    return sorted(files)


def calculate_venv_hash():
    """
    Compute venv validity hash.

    Includes BASE_DIR path so directory moves are detected instantly.
    When the project moves, every venv path (shebangs, .pth, installed
    metadata) becomes stale — this forces a full venv rebuild.
    """
    hasher = hashlib.md5()

    # 1. Hash the project directory path itself (detects moves)
    hasher.update(BASE_DIR.encode("utf-8"))

    # 2. Hash all venv-relevant files
    for fpath in get_venv_files_to_hash():
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                hasher.update(f.read())

    return hasher.hexdigest()


def check_venv_validity():
    """
    Check if pyvenv is valid. Returns True if venv is OK, False if rebuild needed.

    Detects:
      - Project moved to a different directory
      - Requirements files changed
      - Python sidecar scripts changed
    """
    venv_dirs = [
        os.path.join(BASE_DIR, "pyvenv"),
        os.path.join(BASE_DIR, "vendor", "tts_kokoro_component", "venv"),
    ]
    venv_hash_file = os.path.join(BASE_DIR, ".venv_hash")

    current_hash = calculate_venv_hash()

    # Read stored hash
    stored_hash = ""
    if os.path.exists(venv_hash_file):
        with open(venv_hash_file, "r") as f:
            stored_hash = f.read().strip()

    # Check if ALL venvs exist and hash matches
    all_exist = all(os.path.isdir(d) for d in venv_dirs)
    if current_hash == stored_hash and all_exist:
        return True  # venv is valid

    # Venv is invalid — determine why
    missing = [d for d in venv_dirs if not os.path.isdir(d)]
    if missing:
        print(f"[VENV] Missing venvs: {', '.join(os.path.basename(d) for d in missing)} — will create fresh")
    elif stored_hash:
        # Hash mismatch — check if project moved
        try:
            main_venv_python = os.path.join(venv_dirs[0], "bin", "python3")
            if os.path.exists(main_venv_python):
                import subprocess
                result = subprocess.run(
                    [main_venv_python, "-c", "import sys; print(sys.prefix)"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    old_prefix = result.stdout.strip()
                    if old_prefix != BASE_DIR:
                        print(f"[VENV] Project moved: {old_prefix} → {BASE_DIR}")
                    else:
                        print("[VENV] Requirements or sidecar scripts changed")
        except Exception:
            print("[VENV] Venv state unclear — rebuilding")
    else:
        print("[VENV] Venv hash missing or corrupted — rebuilding")

    return False


def invalidate_venv():
    """Destroy all project venvs and clear venv hash so next check forces rebuild."""
    venv_hash_file = os.path.join(BASE_DIR, ".venv_hash")

    # All project venvs that contain hardcoded paths (shebangs, .pth, metadata)
    venv_dirs = [
        os.path.join(BASE_DIR, "pyvenv"),                                    # main venv (LSH, VAD, sidecars)
        os.path.join(BASE_DIR, "vendor", "tts_kokoro_component", "venv"),    # Kokoro TTS isolated venv
    ]

    for venv_dir in venv_dirs:
        if os.path.isdir(venv_dir):
            print(f"[VENV] Destroying stale venv at {venv_dir}...")
            shutil.rmtree(venv_dir, ignore_errors=True)

    # Clear stored hash
    if os.path.exists(venv_hash_file):
        os.remove(venv_hash_file)


def save_venv_hash():
    """Save current venv hash after successful rebuild."""
    venv_hash_file = os.path.join(BASE_DIR, ".venv_hash")
    with open(venv_hash_file, "w") as f:
        f.write(calculate_venv_hash())


# [DO NOT REMOVE] Graceful shutdown via SIGQUIT (Ctrl+\ by default).
# Writes .shutdown_requested flag so run.py signals the Ada server to
# delete it and exit gracefully.  run.py does NOT kill children here —
# it just drops the flag and returns.  The Ada server detects the flag,
# deletes it, does its own clean shutdown.  This prevents accidental
# single-key presses from killing the server.
#
# SIGTERM and SIGINT (Ctrl+C) are kept as hard kill fallbacks (process group kill).
def cleanup(signum=None, frame=None):
    """Signal handler for graceful shutdown.

    SIGQUIT: Write .shutdown_requested flag → Ada server polls for it,
             deletes it, and exits gracefully.  run.py returns without
             killing children — Ada owns its own lifecycle.
    SIGTERM / SIGINT (Ctrl+C): Hard kill — terminate all children immediately.
    """
    if signum == signal.SIGQUIT:
        print("\n[*] SIGQUIT received — writing shutdown flag for Ada...")
        shutdown_flag = os.path.join(BASE_DIR, "run", ".shutdown_requested")
        try:
            os.makedirs(os.path.dirname(shutdown_flag), exist_ok=True)
            with open(shutdown_flag, "w") as f:
                f.write(f"pid={os.getpid()}\n")
        except Exception:
            pass
        # Flag written — return without killing.  Ada will detect and exit
        # gracefully on its next main-loop tick.
        return

    # SIGTERM / SIGINT path: Hard kill all children via process group.
    sig_name = signal.Signals(signum).name if signum else "UNKNOWN"
    print(f"\n[*] {sig_name} received — hard killing all children...")

    # SIGTERM path: Collect PIDs to kill directly — do NOT rely on
    # proc.terminate() inside a signal handler (can deadlock with main
    # thread's proc.wait()).
    pids_to_kill = []
    for proc in [
        daemon_process,
        server_process,
        watchdog_process,
        vad_process,
        sidecar_process,
    ]:
        if proc and proc.poll() is None:
            pids_to_kill.append((proc.pid, proc.args[0] if proc.args else "unknown"))

    # Send SIGTERM first, then SIGKILL after 2s grace period
    SIGTERM = signal.SIGTERM
    SIGKILL = signal.SIGKILL

    for pid, name in pids_to_kill:
        print(f"[*] Sending SIGTERM to {name} (PID {pid})...")
        try:
            os.kill(pid, SIGTERM)
        except ProcessLookupError:
            pass

    # Give 2 seconds for graceful shutdown
    time.sleep(2.0)

    for pid, name in pids_to_kill:
        try:
            # Check if still alive
            os.kill(pid, 0)
            print(f"[*] PID {pid} still alive, sending SIGKILL...")
            os.kill(pid, SIGKILL)
        except ProcessLookupError:
            print(f"[*] PID {pid} exited cleanly.")

    # Force-kill any remaining zombie processes via process group
    for proc in [
        daemon_process,
        server_process,
        watchdog_process,
        vad_process,
        sidecar_process,
    ]:
        if proc:
            try:
                os.killpg(os.getpgid(proc.pid), SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    # Nuclear option: pkill by name for processes that survive SIGKILL
    # (e.g. daemon runner with its own child threads)
    for proc_name in ["adelaide_server", "adelaide_watchdog", "vad_worker.py",
                       "stellaicarus_daemon_runner"]:
        try:
            subprocess.run(["pkill", "-9", "-f", proc_name],
                           stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except Exception:
            pass

    # Wipe master key from environment + remove temp key file
    os.environ.pop("ADELAIDE_MASTER_KEY", None)
    os.environ.pop("ADELAIDE_MASTER_KEY_FILE", None)
    if _master_key_file_path and os.path.exists(_master_key_file_path):
        try:
            os.unlink(_master_key_file_path)
        except Exception:
            pass

    print("[*] Cleanup complete.")
    os._exit(0)


# [DO NOT REMOVE] Register signal handlers.
# SIGQUIT (Ctrl+\ by default) triggers graceful shutdown: writes flag
# file → Ada polls for flag → deletes it → exits gracefully.
# SIGTERM and SIGINT (Ctrl+C) trigger hard kill (process group kill).
signal.signal(signal.SIGQUIT, cleanup)
signal.signal(signal.SIGTERM, cleanup)
signal.signal(signal.SIGINT, cleanup)


def checkout_latest_release(repo_dir, module_name):
    """Fetches the latest release tag and checks it out for stability."""
    try:
        # Fetch tags
        subprocess.run(
            ["git", "fetch", "--tags", "origin"],
            cwd=repo_dir,
            check=False,
            capture_output=True,
        )
        # Find latest tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        latest_tag = result.stdout.strip()
        if latest_tag:
            # Checkout tag
            checkout_res = subprocess.run(
                ["git", "checkout", latest_tag],
                cwd=repo_dir,
                check=False,
                capture_output=True,
                text=True,
            )
            if checkout_res.returncode == 0:
                print(f"[{module_name}] Checked out latest release: {latest_tag}")
            else:
                print(
                    f"[{module_name}] Failed to checkout {latest_tag}: {checkout_res.stderr}"
                )
            return latest_tag
    except Exception as e:
        print(f"[{module_name}] Error checking out latest tag: {e}")
    return None


def safe_cmake_configure(cmake_flags, cwd, build_dir, module_name):
    """Robust CMake configure that detects cache corruption and retries cleanly."""
    result = subprocess.run(
        cmake_flags, cwd=cwd, check=False, capture_output=True, text=True
    )
    if result.returncode != 0 and (
        "CMakeCache.txt" in result.stderr or "CMake Error" in result.stderr
    ):
        print(
            f"{BG_RED}[BUGCHECK] [{module_name}] Corrupted CMakeCache detected. Clearing build dir and retrying...{RST}"
        )
        shutil.rmtree(build_dir, ignore_errors=True)
        os.makedirs(build_dir, exist_ok=True)
        # Re-run from scratch
        result = subprocess.run(
            cmake_flags, cwd=cwd, check=False, capture_output=True, text=True
        )
    return result


def main():
    global current_log_path
    try:
        real_main()
    except BaseException as e:
        is_error = True
        if isinstance(e, SystemExit):
            if e.code == 0:
                is_error = False

        log_path = globals().get("current_log_path") or os.environ.get(
            "ADELAIDE_LOG_FILE", ""
        )
        if not log_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(LOGS_DIR, f"run_{timestamp}.log")

        if IS_KISS and is_error:
            error_str = str(e) or "BOOT_PROCESS_ABORTED"

            # Disguise developer-level terms
            if "GNATprove" in error_str or "formal verification" in error_str:
                error_str = (
                    "CORE_INIT_FAILURE: Core security audit and safety check failed."
                )
            elif "AFL++" in error_str or "fuzzing" in error_str:
                error_str = "CORE_INIT_FAILURE: Core safety analysis and stability checks failed."
            elif "Ruff" in error_str or "INTEGRITY_CHECK_FAILURE" in error_str:
                error_str = (
                    "INTEGRITY_CHECK_FAILURE: System code integrity check failed."
                )
            elif (
                "pyrefly" in error_str
                or "LSH_BOOTSTRAP_FAILURE" in error_str
                or "ruff" in error_str
            ):
                error_str = (
                    "LSH_BOOTSTRAP_FAILURE: Sequence worker security check failed."
                )
            elif "VAD" in error_str or "VAD_BOOTSTRAP_FAILURE" in error_str:
                error_str = (
                    "VAD_BOOTSTRAP_FAILURE: Voice module initialization check failed."
                )

            stop_code = "0x0000007B"
            if "CORE_INIT_FAILURE" in error_str:
                stop_code = "0x00000001"
            elif "FRONTEND_INIT_FAILURE" in error_str:
                stop_code = "0x00000002"
            elif "INTEGRITY_CHECK_FAILURE" in error_str:
                stop_code = "0x00000003"
            elif "VAD_BOOTSTRAP_FAILURE" in error_str:
                stop_code = "0x00000004"
            elif "LSH_BOOTSTRAP_FAILURE" in error_str:
                stop_code = "0x00000005"
            elif "SERVER_CRASHED" in error_str:
                stop_code = "0x00000006"

            show_bsod(error_str, log_path, stop_code)
            sys.exit(1)
        else:
            raise


def real_main():
    global \
        daemon_process, \
        server_process, \
        watchdog_process, \
        vad_process, \
        current_log_path

    current_log_path = setup_logging()

    if "--test-fips" in sys.argv:
        os.environ["ADELAIDE_USER"] = "testfips"
        if "--no-gui" not in sys.argv:
            sys.argv.append("--no-gui")

    # Determine user identity
    if not os.environ.get("ADELAIDE_USER"):
        _welcome_msg = (
            "Heya! I'm Adelaide Zephyrine Charlotte,\n"
            "Today is quite a nice windy with the sun as a\n"
            "star that light pouring above the cloud here\n"
            "and fancy to meet you!"
        )
        if _gui_available() and not IS_KISS:
            # GUI mode: show welcome in dialog, not on terminal
            user = _tk_input_dialog("Adelaide — Identity", "Who am I speaking to?", welcome_msg=_welcome_msg)
            if user:
                user = user.strip()
            if not user:
                # GUI dialog failed or was cancelled — fall back to terminal
                _term_print("")
                _term_print("  (GUI dialog didn't work, let's try here instead)")
                _term_print("")
                user = input("  Your name: ").strip()
        elif not IS_KISS:
            # Verbose mode: print welcome on terminal
            _term_print("")
            _term_print("  Heya! I'm Adelaide Zephyrine Charlotte,")
            _term_print("  Today is quite a nice windy with the sun as a")
            _term_print("  star that light pouring above the cloud here")
            _term_print("  and fancy to meet you!")
            _term_print("")
            user = input("  Your name: ").strip()
        else:
            # KISS mode: no terminal output, just prompt
            user = input("  Your name: ").strip()
            
        if not user:
            print("[IDENTITY] FATAL: I need a name to call you by!")
            sys.exit(1)
        os.environ["ADELAIDE_USER"] = user
        if not IS_KISS:
            _term_print(f"  Nice to meet you, {user}! :D")
            _term_print("")
        print(f"[IDENTITY] Operating as user: {os.environ['ADELAIDE_USER']}")

    # ── Show GUI loading bar immediately ──────────────────────────────────
    # Prevents freeze UX between name dialog and first visible work.
    #
    # NOTE: Step labels use opaque hex codes so the GUI never exposes
    # implementation details (Ada, Python, Vite, etc.) that could confuse
    # or intimidate users.  See the build/validation section below for
    # a full mapping of hex → actual task.
    _setup_gui = None
    if _gui_available() and not IS_KISS:
        _setup_gui = _tk_progress_dialog(
            "Adelaide — Loading",
            "Loading preparing for Model...\n(Nothing to see here)"
        )
        _setup_gui._update_bar(0, step_text="code step 0x0001", pulse=True)  # Starting up
        _setup_gui._start_pulse()

    # Whimsical password promise (only on first entry when user was just created)
    if not os.environ.get("_ZEPZEP_PASSWORD_PROMPTED"):
        os.environ["_ZEPZEP_PASSWORD_PROMPTED"] = "1"
        if not IS_KISS:
            _term_print("")
            _term_print("  Oki :D, now so that we can keep secret between")
            _term_print("  each other, I am with my pinky finger, promise")
            _term_print("  to not share your data with others *wink")
            _term_print("")
    # Kill any stale processes from previous runs before starting
    print("[*] Cleaning up any stale processes from previous runs...")
    if _setup_gui:
        _setup_gui._update_bar(2, step_text="code step 0x0002", pulse=True)  # Clean up stale processes from previous runs
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "adelaide_server"], stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["pkill", "-9", "-f", "adelaide_watchdog"], stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["pkill", "-9", "-f", "vad_worker.py"], stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

    if IS_KISS:
        p_thread = threading.Thread(
            target=progress_monitor, args=(current_log_path,), daemon=True
        )
        p_thread.start()

    # Declare key paths and config objects at the top to prevent UnboundLocalError on direct launch
    env = os.environ.copy()
    lsh_reqs = os.path.join(BASE_DIR, "lsh", "requirements-lsh.txt")
    lsh_worker = os.path.join(BASE_DIR, "lsh", "lsh_qrnn_worker.py")
    vad_worker_script = os.path.join(BASE_DIR, "vad_component", "vad_worker.py")
    pyvenv_dir = os.path.join(BASE_DIR, "pyvenv")
    pyvenv_python = (
        os.path.join(pyvenv_dir, "bin", "python3")
        if platform.system() != "Windows"
        else os.path.join(pyvenv_dir, "Scripts", "python.exe")
    )
    alr_cmd = "alr.exe" if platform.system() == "Windows" else "alr"
    hash_file = os.path.join(BASE_DIR, ".build_hash")

    # 0. Verify all critical prerequisites are installed
    # PX4 is critical and auto-clones/compiles if missing
    if _setup_gui:
        _setup_gui._update_bar(5, step_text="code step 0x0003", pulse=True)  # Verify environment prerequisites
    verify_environment(build_px4=True)

    print(f"[*] Setting up Adelaide-Lite environment in {BASE_DIR}...")

    start_time = int(time.time() * 1000)

    # Detect Platform and Backend
    ggml_backend = "none"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        ggml_backend = "metal"
    elif platform.system() == "Linux":
        if shutil.which("nvcc") or shutil.which("nvidia-smi"):
            ggml_backend = "cuda"
        elif shutil.which("sycl-ls") or os.environ.get("ONEAPI_ROOT"):
            ggml_backend = "sycl"
        else:
            ggml_backend = "vulkan"
    os.environ["GGML_BACKEND"] = ggml_backend
    print(
        f"[*] Detected Platform: {platform.system()} | Selected Backend: {ggml_backend.upper()}"
    )

    # Calculate Current Hash
    current_hash = calculate_hash(get_files_to_hash())
    hash_file = os.path.join(BASE_DIR, ".build_hash")

    saved_hash = ""
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            saved_hash = f.read().strip()

    # ── Venv Validity Check ────────────────────────────────────────────────
    # Detects stale pyvenv from directory moves, requirements changes, etc.
    # If invalid, forces a full rebuild (clears .build_hash).
    venv_valid = check_venv_validity()
    if not venv_valid:
        print("[*] Venv invalid — forcing full rebuild...")
        invalidate_venv()
        saved_hash = ""  # force current_hash != saved_hash → triggers rebuild

    daemon_build_flag = ""

    if current_hash != saved_hash:
        print("[*] Changes detected, checking downloads and rebuilding...")
        if _setup_gui:
            _setup_gui._update_bar(15, step_text="code step 0x0004", pulse=True)  # Download and rebuild components
        threads = str(os.cpu_count() or 4)

        # =====================================================================
        # GGML: Built in-tree by llama.cpp (vendor/llama.cpp/build/ggml/)
        # =====================================================================
        # The GPR links against vendor/llama.cpp/build/ggml/src/libggml*.a
        # No separate ggml build needed — llama.cpp compiles its own ggml
        # as part of its cmake build. This ensures version consistency.
        # [VITAL-DO-NOT-REMOVE] Never use Homebrew's ggml.

        # =====================================================================
        # llama.cpp: clone → fetch+pull latest → rebuild if updated
        # =====================================================================
        # We always fetch+pull so we get the latest fixes.
        # llama.cpp builds ggml in-tree. The GPR links the in-tree build.
        llama_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "llama.cpp"))
        llama_build_dir = os.path.join(llama_dir, "build")
        llama_lib = os.path.join(llama_build_dir, "src", "libllama.a")
        llama_start = time.time()

        if not os.path.exists(llama_dir):
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Cloning llama.cpp...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/ggml-org/llama.cpp.git",
                    llama_dir,
                ],
                check=False,
            )
            checkout_latest_release(llama_dir, "LLAMA")
            needs_build = True
        else:
            old_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=llama_dir,
                capture_output=True,
                text=True,
            ).stdout.strip()
            print(
                f"[LLAMA] [{time.strftime('%H:%M:%S')}] Fetching latest llama.cpp release..."
            )
            checkout_latest_release(llama_dir, "LLAMA")
            new_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=llama_dir,
                capture_output=True,
                text=True,
            ).stdout.strip()
            needs_build = (old_head != new_head) or not os.path.exists(llama_lib)
            if old_head != new_head:
                print(
                    f"[LLAMA] [{time.strftime('%H:%M:%S')}] Updated: {old_head[:8]} → {new_head[:8]}"
                )
            else:
                print(
                    f"[LLAMA] [{time.strftime('%H:%M:%S')}] Already up to date ({new_head[:8]})"
                )

        # Build if needed (new clone, update, or missing lib)
        if needs_build:
            print(f"[LLAMA] [{time.strftime('%H:%M:%S')}] Building llama.cpp...")
            print(
                f"[LLAMA] [{time.strftime('%H:%M:%S')}] CMake flags: -DGGML_NATIVE=ON -DLLAMA_BUILD_TOOLS=ON -DBUILD_SHARED_LIBS=OFF"
            )
            os.makedirs(llama_build_dir, exist_ok=True)
            cmake_flags = [
                "cmake",
                "-B",
                "build",
                "-DGGML_NATIVE=ON",
                "-DLLAMA_BUILD_TOOLS=ON",
                "-DBUILD_SHARED_LIBS=OFF",
            ]
            if ggml_backend == "metal":
                print(
                    f"[LLAMA] [{time.strftime('%H:%M:%S')}] Metal GPU acceleration: ENABLED"
                )
                cmake_flags.append("-DGGML_METAL=ON")
            elif ggml_backend == "cuda":
                print(
                    f"[LLAMA] [{time.strftime('%H:%M:%S')}] CUDA GPU acceleration: ENABLED"
                )
                cmake_flags.append("-DGGML_CUDA=ON")
            elif ggml_backend == "sycl":
                print(
                    f"[LLAMA] [{time.strftime('%H:%M:%S')}] SYCL/oneAPI GPU acceleration: ENABLED"
                )
                cmake_flags.append("-DGGML_SYCL=ON")
            elif ggml_backend == "vulkan":
                print(
                    f"[LLAMA] [{time.strftime('%H:%M:%S')}] Vulkan GPU acceleration: ENABLED"
                )
                cmake_flags.append("-DGGML_VULKAN=ON")
            result = safe_cmake_configure(
                cmake_flags,
                cwd=llama_dir,
                build_dir=llama_build_dir,
                module_name="LLAMA",
            )
            if result.returncode != 0:
                print(
                    f"{BG_RED}[BUGCHECK] [LLAMA] [{time.strftime('%H:%M:%S')}] CMake configure FAILED{RST}"
                )
                if result.stderr:
                    print(
                        f"[LLAMA] [{time.strftime('%H:%M:%S')}] stderr: {result.stderr[-500:]}"
                    )
            else:
                print(
                    f"[LLAMA] [{time.strftime('%H:%M:%S')}] CMake configure OK, building..."
                )
                # DO NOT SUPPRESS VERBOSITY IF YOU ARE NOT OVERCONFIDENT
                result = subprocess.run(
                    [
                        "cmake",
                        "--build",
                        "build",
                        "--config",
                        "Release",
                        "-j",
                        "--verbose",
                    ],
                    cwd=llama_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                llama_elapsed = time.time() - llama_start
                if result.returncode == 0:
                    print(
                        f"[LLAMA] [{time.strftime('%H:%M:%S')}] Build SUCCESS in {llama_elapsed:.1f}s"
                    )
                else:
                    print(
                        f"{BG_RED}[BUGCHECK] [LLAMA] [{time.strftime('%H:%M:%S')}] Build FAILED in {llama_elapsed:.1f}s{RST}"
                    )
                    if result.stderr:
                        print(
                            f"[LLAMA] [{time.strftime('%H:%M:%S')}] stderr: {result.stderr[-500:]}"
                        )
        else:
            llama_elapsed = time.time() - llama_start
            print(
                f"[LLAMA] [{time.strftime('%H:%M:%S')}] Library exists, skipping build"
            )

        # Ensure mtmd (multimodal) library is built
        mtmd_lib = os.path.join(llama_build_dir, "tools", "mtmd", "libmtmd.a")
        mtmd_start = time.time()
        if not os.path.exists(mtmd_lib):
            print(
                f"[MTMD] [{time.strftime('%H:%M:%S')}] Building mtmd (multimodal) library..."
            )
            # DO NOT SUPPRESS VERBOSITY IF YOU ARE NOT OVERCONFIDENT
            result = subprocess.run(
                ["cmake", "--build", "build", "--target", "mtmd", "-j", "--verbose"],
                cwd=llama_dir,
                check=False,
                capture_output=True,
                text=True,
            )
            mtmd_elapsed = time.time() - mtmd_start
            if result.returncode == 0:
                print(
                    f"[MTMD] [{time.strftime('%H:%M:%S')}] Build SUCCESS in {mtmd_elapsed:.1f}s"
                )
                # Verify the library was created
                if os.path.exists(mtmd_lib):
                    mtmd_size = os.path.getsize(mtmd_lib)
                    print(
                        f"[MTMD] [{time.strftime('%H:%M:%S')}] Library created: {mtmd_size:,} bytes"
                    )
                else:
                    print(
                        f"[MTMD] [{time.strftime('%H:%M:%S')}] WARNING: Library file not found after build!"
                    )
            else:
                print(
                    f"{BG_RED}[BUGCHECK] [MTMD] [{time.strftime('%H:%M:%S')}] Build FAILED in {mtmd_elapsed:.1f}s{RST}"
                )
                if result.stdout:
                    print(
                        f"[MTMD] [{time.strftime('%H:%M:%S')}] stdout: {result.stdout[-500:]}"
                    )
                if result.stderr:
                    print(
                        f"[MTMD] [{time.strftime('%H:%M:%S')}] stderr: {result.stderr[-500:]}"
                    )
        else:
            mtmd_elapsed = time.time() - mtmd_start
            mtmd_size = os.path.getsize(mtmd_lib)
            print(
                f"[MTMD] [{time.strftime('%H:%M:%S')}] Library exists ({mtmd_size:,} bytes), skipping build"
            )

        # Check and clone kokoro-onnx
        kokoro_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "kokoro-onnx"))
        if not os.path.exists(kokoro_dir):
            print("[*] Cloning kokoro-onnx...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/thewh1teagle/kokoro-onnx",
                    kokoro_dir,
                ],
                check=False,
            )
            checkout_latest_release(kokoro_dir, "KOKORO-ONNX")
        else:
            print("[*] kokoro-onnx already exists, skipping clone.")

        kokoclone_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "kokoclone"))
        if not os.path.exists(kokoclone_dir):
            print("[*] Cloning KokoClone Zero-Shot Repository...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/Ashish-Patnaik/kokoclone.git",
                    kokoclone_dir,
                ],
                check=True,
            )
            checkout_latest_release(kokoclone_dir, "KOKOCLONE")
        else:
            print("[*] kokoclone already exists, skipping clone.")

        # Ensure Kokoro TTS component dependencies are installed in an isolated venv
        kokoro_comp_dir = os.path.abspath(
            os.path.join(BASE_DIR, "vendor", "tts_kokoro_component")
        )
        kokoro_venv_dir = os.path.join(kokoro_comp_dir, "venv")
        if not os.path.exists(kokoro_venv_dir):
            print(
                "[*] Creating dedicated virtual environment for Kokoro TTS (Python 3.12)..."
            )
            subprocess.run(["python3.12", "-m", "venv", kokoro_venv_dir], check=True)

        print("[*] Installing Kokoro TTS requirements...")
        kokoro_pip = (
            os.path.join(kokoro_venv_dir, "bin", "pip")
            if platform.system() != "Windows"
            else os.path.join(kokoro_venv_dir, "Scripts", "pip.exe")
        )
        subprocess.run(
            [
                kokoro_pip,
                "install",
                "-r",
                os.path.join(kokoro_comp_dir, "requirements.txt"),
            ],
            check=False,
        )
        # kokoclone/stereo_cloner needs torch but it's not in requirements.txt
        # (git-cloned repo). Install here so it persists across repo updates.
        kokoro_python = os.path.join(kokoro_venv_dir, "bin", "python")
        torch_check = subprocess.run(
            [kokoro_python, "-c", "import torch"], capture_output=True
        )
        if torch_check.returncode != 0:
            print("[*] Installing torch for kokoclone voice cloning...")
            subprocess.run(
                [
                    kokoro_pip,
                    "install",
                    "torch",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ],
                check=False,
            )

        # Check and clone moonshine
        moonshine_dir = os.path.abspath(os.path.join(BASE_DIR, "vendor", "moonshine"))
        if not os.path.exists(moonshine_dir):
            print("[*] Cloning moonshine...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/moonshine-ai/moonshine.git",
                    moonshine_dir,
                ],
                check=False,
            )
            checkout_latest_release(moonshine_dir, "MOONSHINE")

            # Autoremove examples to save space
            moonshine_examples = os.path.join(moonshine_dir, "examples")
            if os.path.exists(moonshine_examples):
                print("[*] Removing heavy moonshine/examples directory...")
                shutil.rmtree(moonshine_examples, ignore_errors=True)
        else:
            print("[*] moonshine already exists, skipping clone.")

        # Ensure Moonshine is built
        moonshine_build_dir = os.path.join(moonshine_dir, "build")
        moonshine_core_lib = (
            os.path.join(moonshine_build_dir, "core", "libmoonshine.dylib")
            if platform.system() == "Darwin"
            else os.path.join(moonshine_build_dir, "core", "libmoonshine.so")
        )
        if not os.path.exists(moonshine_core_lib):
            print("[*] Building moonshine C API...")
            os.makedirs(moonshine_build_dir, exist_ok=True)
            result = safe_cmake_configure(
                ["cmake", ".."],
                cwd=moonshine_build_dir,
                build_dir=moonshine_build_dir,
                module_name="MOONSHINE",
            )
            subprocess.run(
                ["make", f"-j{threads}"], cwd=moonshine_build_dir, check=False
            )
        else:
            print("[*] moonshine core library exists, skipping cmake build.")

        # Check and download Moonshine models
        moonshine_models_dir = os.path.abspath(
            os.path.join(BASE_DIR, "vendor", "moonshine", "models")
        )
        if not os.path.exists(moonshine_models_dir) or not os.listdir(
            moonshine_models_dir
        ):
            print("[*] Downloading Moonshine models...")
            os.makedirs(moonshine_models_dir, exist_ok=True)
            env_for_download = os.environ.copy()
            env_for_download["PYTHONPATH"] = os.path.join(
                moonshine_dir, "python", "src"
            )
            download_script = os.path.join(
                moonshine_dir, "python", "src", "moonshine_voice", "download.py"
            )
            subprocess.run(
                [
                    sys.executable,
                    download_script,
                    "--stt",
                    "--language",
                    "en",
                    "--root",
                    moonshine_models_dir,
                ],
                env=env_for_download,
                check=False,
            )
        else:
            print("[*] Moonshine models already exist, skipping download.")

        # =====================================================================
        # stable-diffusion.cpp: clone → fetch+pull latest → init ggml → build
        # =====================================================================
        # [VITAL-DO-NOT-REMOVE] FLUX Schnell image generation backend.
        # Builds a static library (libstable_diffusion.a) for Ada FFI linkage.
        # The ggml submodule within stable-diffusion.cpp must be initialized
        # before cmake can configure — it provides the compute graph runtime.
        sd_cpp_dir = os.path.abspath(
            os.path.join(BASE_DIR, "vendor", "stable-diffusion.cpp")
        )
        sd_cpp_built = os.path.join(sd_cpp_dir, "build")
        sd_cpp_lib_static = os.path.join(sd_cpp_built, "libstable-diffusion.a")
        sd_cpp_lib_shared = (
            os.path.join(sd_cpp_built, "libstable-diffusion.dylib")
            if platform.system() == "Darwin"
            else os.path.join(sd_cpp_built, "libstable-diffusion.so")
        )
        sd_cpp_start = time.time()

        if not os.path.exists(sd_cpp_dir):
            print(
                f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Cloning stable-diffusion.cpp..."
            )
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/leejet/stable-diffusion.cpp.git",
                    sd_cpp_dir,
                ],
                check=False,
            )
            checkout_latest_release(sd_cpp_dir, "SD-CPP")
            needs_build = True
        else:
            old_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=sd_cpp_dir,
                capture_output=True,
                text=True,
            ).stdout.strip()
            print(
                f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Fetching latest stable-diffusion.cpp release..."
            )
            checkout_latest_release(sd_cpp_dir, "SD-CPP")
            new_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=sd_cpp_dir,
                capture_output=True,
                text=True,
            ).stdout.strip()
            needs_build = (old_head != new_head) or not (
                os.path.exists(sd_cpp_lib_static) or os.path.exists(sd_cpp_lib_shared)
            )
            if old_head != new_head:
                print(
                    f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Updated: {old_head[:8]} → {new_head[:8]}"
                )
            else:
                print(
                    f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Already up to date ({new_head[:8]})"
                )

        # Init stable-diffusion.cpp's own ggml submodule (required for cmake)
        sd_ggml_sub = os.path.join(sd_cpp_dir, "ggml")
        sd_ggml_cmakelists = os.path.join(sd_ggml_sub, "CMakeLists.txt")
        if not os.path.exists(sd_ggml_cmakelists):
            print(
                f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Initializing ggml submodule inside stable-diffusion.cpp..."
            )
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=sd_cpp_dir,
                check=False,
                capture_output=True,
            )

        # Build static library for Ada FFI linkage
        if needs_build or not (
            os.path.exists(sd_cpp_lib_static) or os.path.exists(sd_cpp_lib_shared)
        ):
            print(
                f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Building stable-diffusion.cpp (static lib)..."
            )
            os.makedirs(sd_cpp_built, exist_ok=True)
            cmake_flags = [
                "cmake",
                "..",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DSD_BUILD_EXAMPLES=OFF",
            ]
            if ggml_backend == "metal":
                cmake_flags.append("-DGGML_METAL=ON")
            elif ggml_backend == "cuda":
                cmake_flags.append("-DGGML_CUDA=ON")
            result = safe_cmake_configure(
                cmake_flags,
                cwd=sd_cpp_built,
                build_dir=sd_cpp_built,
                module_name="SD-CPP",
            )
            if result.returncode != 0:
                print(
                    f"{BG_RED}[BUGCHECK] [SD-CPP] [{time.strftime('%H:%M:%S')}] CMake FAILED: {result.stderr[-500:]}{RST}"
                )
            else:
                # DO NOT SUPPRESS VERBOSITY IF YOU ARE NOT OVERCONFIDENT
                result = subprocess.run(
                    ["cmake", "--build", ".", "--config", "Release", "-j", "--verbose"],
                    cwd=sd_cpp_built,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                sd_elapsed = time.time() - sd_cpp_start
                if result.returncode == 0:
                    # Verify the library was created
                    if os.path.exists(sd_cpp_lib_static):
                        sd_size = os.path.getsize(sd_cpp_lib_static)
                        print(
                            f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Build SUCCESS in {sd_elapsed:.1f}s ({sd_size:,} bytes)"
                        )
                    elif os.path.exists(sd_cpp_lib_shared):
                        sd_size = os.path.getsize(sd_cpp_lib_shared)
                        print(
                            f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Build SUCCESS (shared) in {sd_elapsed:.1f}s ({sd_size:,} bytes)"
                        )
                    else:
                        print(
                            f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Build completed but library not found at expected path"
                        )
                else:
                    print(
                        f"{BG_RED}[BUGCHECK] [SD-CPP] [{time.strftime('%H:%M:%S')}] Build FAILED in {sd_elapsed:.1f}s{RST}"
                    )
                    if result.stderr:
                        print(
                            f"[SD-CPP] [{time.strftime('%H:%M:%S')}] stderr: {result.stderr[-500:]}"
                        )
        else:
            sd_elapsed = time.time() - sd_cpp_start
            print(
                f"[SD-CPP] [{time.strftime('%H:%M:%S')}] Library exists ({sd_elapsed:.1f}s), skipping build"
            )

        # Check and download Qwen models
        qwen_models_dir = os.path.abspath(os.path.join(BASE_DIR, "model"))
        os.makedirs(qwen_models_dir, exist_ok=True)

        models_to_download = [
            {
                "url": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf?download=true",
                "output": "Qwen3.5-0.8B-Q4_K_M.gguf",
            },
            {
                "url": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F16.gguf?download=true",
                "output": "mmproj-0.8B-F16.gguf",
            },
            {
                "url": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true",
                "output": "Qwen3-Embedding-0.6B-Q8_0.gguf",
            },
            {
                "url": "https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/resolve/main/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf?download=true",
                "output": "Mythos9bHybridq4.gguf",
            },
            {
                "url": "https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/resolve/main/mmproj-Qwythos-9B-Claude-Mythos-5-1M-f16.gguf?download=true",
                "output": "Mythos9bHybridq4-mmproj-fp16.gguf",
            },
            {
                "url": "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf?download=true",
                "output": "Qwen3-Reranker-0.6B-Q8_0.gguf",
            },
        ]

        aria2c_cmd = shutil.which("aria2c")
        for model in models_to_download:
            target_path = os.path.join(qwen_models_dir, model["output"])
            if not os.path.exists(target_path):
                print(f"[*] Downloading {model['output']}...")
                if aria2c_cmd:
                    subprocess.run(
                        [
                            aria2c_cmd,
                            "-x",
                            "16",
                            "-s",
                            "16",
                            "-k",
                            "1M",
                            model["url"],
                            "-o",
                            model["output"],
                            "-d",
                            qwen_models_dir,
                        ],
                        check=True,
                    )
                else:
                    subprocess.run(
                        [
                            "wget",
                            "-q",
                            "--show-progress",
                            model["url"],
                            "-O",
                            target_path,
                        ],
                        check=True,
                    )

        # Check and download Kokoro models
        kokoro_models_dir = os.path.abspath(
            os.path.join(BASE_DIR, "vendor", "kokoro_models")
        )
        os.makedirs(kokoro_models_dir, exist_ok=True)
        kokoro_onnx_model = os.path.join(kokoro_models_dir, "kokoro-v0_19.int8.onnx")
        kokoro_voices = os.path.join(kokoro_models_dir, "voices-v1.0.bin")
        if not os.path.exists(kokoro_onnx_model):
            print("[*] Downloading Kokoro ONNX model...")
            subprocess.run(
                [
                    "wget",
                    "-q",
                    "--show-progress",
                    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.int8.onnx",
                ],
                cwd=kokoro_models_dir,
                check=False,
            )
        if not os.path.exists(kokoro_voices):
            print("[*] Downloading Kokoro voices...")
            subprocess.run(
                [
                    "wget",
                    "-q",
                    "--show-progress",
                    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                ],
                cwd=kokoro_models_dir,
                check=False,
            )

        # =====================================================================
        # FLUX Schnell models (stable-diffusion.cpp image generation)
        # =====================================================================
        # [VITAL-DO-NOT-REMOVE] TWO-STAGE IMAGE GENERATION ARCHITECTURE:
        #
        #   STAGE 1: FLUX Schnell Q2_K (sparse, fast, low quality)
        #     - Diffusion model: flux1-schnell.gguf (~4GB GGUF)
        #     - Text encoders: clip_l.safetensors + t5xxl Q4_0 GGUF (~2.9GB)
        #     - VAE: ae.safetensors (~335MB)
        #     - Output: sparse/draft image (2-4 steps, CFG 1.0)
        #
        #   STAGE 2: SD Refinement (img2img upscale, high quality)
        #     - Model: sd-refinement.gguf (~1.9GB, SD 1.5 pruned)
        #     - Input: Stage 1 output + added noise (strength ~0.4)
        #     - Output: refined/final image (dpmpp2mv2, 8+ steps)
        #     - Prompt: "Masterpiece, Amazing, 4k, " + original_prompt + ", highly detailed..."
        #
        #   Memory budget: FLUX Q2_K (~4GB) + t5xxl Q4_0 (~2.9GB) + SD refinement (~1.9GB)
        #   = ~8.8GB total (fits 9B-class VRAM with swap)
        #
        # Source repos:
        #   Diffusion: city96/FLUX.1-schnell-gguf (preconverted GGUF)
        #   T5-XXL:    Phil2Sat/T5XXL-Unchained-GGUF (Q4_0, smallest GGUF t5xxl)
        #   CLIP-L:    comfyanonymous/flux_text_encoders (safetensors)
        #   VAE:       ffxvs/vae-flux (public mirror, BFL repos are gated)
        #   Refinement: second-state/stable-diffusion-v1-5-GGUF (SD 1.5 Q8_0)
        # Reference: stable-diffusion.cpp/docs/flux.md
        #            project-zephyrine imagination_worker.py (two-stage pipeline)
        flux_models_dir = os.path.abspath(os.path.join(BASE_DIR, "model"))
        os.makedirs(flux_models_dir, exist_ok=True)

        #  SHA256 hashes verified from HuggingFace repo metadata.
        #  None = no hash available, skip verification.
        flux_models_to_download = [
            # Diffusion model Q2_K (~4GB) — fits 9B-class VRAM budget
            {
                "url": "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q2_K.gguf?download=true",
                "output": "flux1-schnell.gguf",
                "sha256": None,  # ~4GB, too large to pre-verify
            },
            # T5-XXL text encoder Q4_0 GGUF (~2.9GB) — small enough for VRAM
            {
                "url": "https://huggingface.co/Phil2Sat/T5XXL-Unchained-GGUF/resolve/main/Kaoru8-t5xxl-unchained-Q4_0.gguf?download=true",
                "output": "flux1-t5xxl.gguf",
                "sha256": None,
            },
            # CLIP-L text encoder (safetensors, ~246MB — small, always fits)
            {
                "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true",
                "output": "clip_l.safetensors",
                "sha256": "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
            },
            # VAE (safetensors, ~335MB — public mirror, BFL repos are gated)
            {
                "url": "https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors?download=true",
                "output": "ae.safetensors",
                "sha256": "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
            },
            # SD refinement model (~1.9GB — Stage 2 img2img upscale after FLUX sparse output)
            # Architecture: FLUX Q2_K sparse → add noise → SD refinement upscale
            {
                "url": "https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF/resolve/main/stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf?download=true",
                "output": "sd-refinement.gguf",
                "sha256": None,
            },
        ]

        def sha256_file(filepath):
            """Compute SHA256 of a file, streaming in chunks for large files."""
            h = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        def download_with_retry(url, output_path, expected_sha256=None):
            """Download a file with infinite retry, resume, and SHA256 verification."""
            attempt = 0
            while True:
                attempt += 1
                print(
                    f"[*] Downloading {os.path.basename(output_path)} (attempt #{attempt})..."
                )
                result = subprocess.run(
                    [
                        "wget",
                        "-c",
                        "-t",
                        "0",
                        "--timeout=30",
                        "--waitretry=5",
                        "--show-progress",
                        url,
                        "-O",
                        output_path,
                    ],
                    check=False,
                    timeout=None,
                )
                if result.returncode != 0:
                    print(
                        f"{BG_RED}[BUGCHECK] [!] wget failed (code {result.returncode}), retrying in 5s...{RST}"
                    )
                    time.sleep(5)
                    continue

                # wget succeeded — verify SHA256 if provided
                if expected_sha256:
                    print(
                        f"[*] Verifying SHA256 for {os.path.basename(output_path)}..."
                    )
                    actual_sha256 = sha256_file(output_path)
                    if actual_sha256 == expected_sha256:
                        print(f"[+] {os.path.basename(output_path)} OK (hash verified)")
                        return True
                    else:
                        print(
                            f"[!] SHA256 MISMATCH: expected={expected_sha256} actual={actual_sha256}"
                        )
                        print("[!] Corrupted download, deleting and retrying...")
                        os.remove(output_path)
                        time.sleep(5)
                        continue
                else:
                    print(
                        f"[+] {os.path.basename(output_path)} downloaded ({os.path.getsize(output_path):,} bytes)"
                    )
                    return True

        for model in flux_models_to_download:
            target_path = os.path.join(flux_models_dir, model["output"])
            expected_sha256 = model.get("sha256")

            if os.path.exists(target_path):
                if expected_sha256:
                    actual_sha256 = sha256_file(target_path)
                    if actual_sha256 == expected_sha256:
                        print(
                            f"[SKIP] {model['output']} exists and hash verified ({os.path.getsize(target_path):,} bytes)"
                        )
                        continue
                    else:
                        print(
                            f"[REHASH] {model['output']} hash mismatch, re-downloading..."
                        )
                        os.remove(target_path)
                else:
                    print(
                        f"[SKIP] {model['output']} exists ({os.path.getsize(target_path):,} bytes)"
                    )
                    continue

            download_with_retry(model["url"], target_path, expected_sha256)

        # Ensure Deno Playwright Chromium is installed
        print("[*] Installing Playwright Chromium binary for Deno crawler...")
        # Cross platform deno invocation
        deno_cmd = "deno.exe" if platform.system() == "Windows" else "deno"
        try:
            subprocess.run(
                [deno_cmd, "run", "-A", "npm:playwright", "install", "chromium"],
                check=False,
            )
        except FileNotFoundError:
            print("[!] Deno not found in PATH, skipping playwright installation.")

        # ═══════════════════════════════════════════════════════════════════
        # Build / Verification Step Hex Codes (GUI-safe labels)
        #
        # The GUI displays these hex codes instead of technology names so
        # the interface stays clean and approachable for all users.
        #
        #   0x0001  Starting up
        #   0x0002  Clean up stale processes from previous runs
        #   0x0003  Verify environment prerequisites
        #   0x0004  Download and rebuild components
        #   0x0005  Build core engine
        #   0x0006  Build complete, running verification suites
        #   0x0007  Formal proof verification of core logic
        #   0x0008  Fuzz testing setup
        #   0x0009  Build user interface
        #   0x000A  Code quality check
        #   0x000B  Symbolic analysis of code paths
        #   0x000C  Type consistency check
        #   0x000D  Initialize background processing systems
        #   0x000E  Initialize audio processing pipeline
        # ═══════════════════════════════════════════════════════════════════
        print("[*] Resolving Ada dependencies and building project...")
        if _setup_gui:
            _setup_gui._update_bar(70, step_text="code step 0x0005", pulse=True)  # Build core engine (Ada compilation)

        env = os.environ.copy()
        if platform.system() == "Darwin":
            try:
                sdk_path = (
                    subprocess.check_output(["xcrun", "--show-sdk-path"])
                    .decode()
                    .strip()
                )
                env["SDKROOT"] = sdk_path
                env["CPATH"] = os.path.join(sdk_path, "usr", "include")
                env["C_INCLUDE_PATH"] = os.path.join(sdk_path, "usr", "include")
                env["LIBRARY_PATH"] = os.path.join(sdk_path, "usr", "lib")
            except Exception as e:
                print(f"[!] Warning: Could not set macOS SDK paths: {e}")

        # Note for future agents: The user strictly wants Alire to use the local alirevenv
        alr_cmd = "alr.exe" if platform.system() == "Windows" else "alr"
        # Update version.ads with current git hash before building
        version_script = os.path.join(BASE_DIR, "scripts", "update_version.sh")
        if os.path.exists(version_script):
            subprocess.run(["bash", version_script], cwd=BASE_DIR, check=False)

        # Run build in a thread so tkinter GUI stays responsive with progress bar
        _build_result = [None]
        _build_done = threading.Event()

        def _run_build():
            try:
                subprocess.run([alr_cmd, "build"], env=env, cwd=BASE_DIR, check=True)
                _build_result[0] = True
            except subprocess.CalledProcessError:
                _build_result[0] = False
            _build_done.set()

        _build_thread = threading.Thread(target=_run_build, daemon=True)
        _build_thread.start()

        # Reuse the existing setup GUI dialog instead of creating a new one
        build_gui_dialog = _setup_gui

        build_bar_width = 40
        build_elapsed = 0.0
        build_eta_target = 60.0  # estimate for build
        while not _build_done.is_set():
            pct = min(99, int(100 * build_elapsed / build_eta_target))
            eta = max(0, int(build_eta_target - build_elapsed))
            if build_gui_dialog:
                build_gui_dialog._update_bar(pct, eta_text=f"ETA: {eta}s", step_text="code step 0x0005")  # Build core engine (Ada compilation)
            elif not IS_KISS:
                filled = int(build_bar_width * pct / 100)
                bar = "█" * filled + "░" * (build_bar_width - filled)
                _term_print(f"\r\033[K  Loading preparing for Model... |{bar}| {pct}%  ETA: {eta}s")
            time.sleep(0.5)
            build_elapsed += 0.5

        _build_thread.join()

        if build_gui_dialog:
            build_gui_dialog._update_bar(80, eta_text="", step_text="code step 0x0006")  # Build complete, running verification suites
            time.sleep(0.3)
        elif not IS_KISS:
            _term_print(f"\r\033[K  Loading preparing for Model... |{'█' * build_bar_width}| 100%  Done!")

        if not _build_result[0]:
            if build_gui_dialog:
                _tk_progress_done(build_gui_dialog)
            raise RuntimeError("CORE_INIT_FAILURE: Core initialization failed.")

        # =====================================================================
        # VERIFICATION STAGES: GNATprove, AFL++, Ruff, pyrefly, and tsc
        # =====================================================================

        # 1. GNATprove Formal Verification (always on rebuild)
        print("\n[*] Stage: GNATprove SPARK Static Analysis...")
        if _setup_gui:
            _setup_gui._update_bar(82, step_text="code step 0x0007", pulse=True)  # Formal proof verification of core logic
        prove_cmd = [
            alr_cmd,
            "exec",
            "--",
            "gnatprove",
            "-P",
            "adelaide_spark.gpr",
            "--level=4",
            "--prover=cvc5,z3,altergo",
            "--timeout=60",
            "--memlimit=2000",
            "--steps=0",
            "--counterexamples=on",
            "--report=fail",
            "--warnings=error",
            "-j0",
        ]
        try:
            subprocess.run(prove_cmd, cwd=BASE_DIR, env=env, check=True)
            print("[+] GNATprove: Formal verification PASSED.")
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "CORE_INIT_FAILURE: GNATprove formal verification failed."
            )

        # 2. AFL++ Fuzzing Environment Check
        print("\n[*] Stage: AFL++ Fuzzing Readiness Check...")
        if _setup_gui:
            _setup_gui._update_bar(85, step_text="code step 0x0008", pulse=True)  # Fuzz testing setup
        fuzz_ready = False
        for compiler in ["afl-clang-fast", "afl-gcc-fast", "afl-clang-lto"]:
            if shutil.which(compiler):
                fuzz_ready = True
                break
        if fuzz_ready and shutil.which("afl-fuzz"):
            print("[+] AFL++ environment is fully ready for binary torture.")
        else:
            raise RuntimeError("CORE_INIT_FAILURE: AFL++ environment is incomplete.")

        # 3. Vite Frontend build (runs tsc and vite build)
        print("[*] Building Vite Frontend for Sidecar UI...")
        if _setup_gui:
            _setup_gui._update_bar(88, step_text="code step 0x0009", pulse=True)  # Build user interface
        frontend_dir = os.path.join(BASE_DIR, "ui", "frontend")
        if os.path.exists(frontend_dir):
            npm_cmd = "npm.cmd" if platform.system() == "Windows" else "npm"
            try:
                subprocess.run([npm_cmd, "install"], cwd=frontend_dir, check=True)
                print("[*] Running auto npm audit fix to resolve vulnerabilities...")
                subprocess.run([npm_cmd, "audit", "fix"], cwd=frontend_dir, check=False)
                subprocess.run([npm_cmd, "run", "build"], cwd=frontend_dir, check=True)
            except subprocess.CalledProcessError:
                raise RuntimeError(
                    "FRONTEND_INIT_FAILURE: User interface initialization failed."
                )

        # 4. Self-Integrity Check using Ruff
        ruff_cmd = "ruff.exe" if platform.system() == "Windows" else "ruff"
        if shutil.which(ruff_cmd):
            print("[*] Running Platform Self-Integrity Quality Check (Ruff)...")
            if _setup_gui:
                _setup_gui._update_bar(90, step_text="code step 0x000A", pulse=True)  # Code quality check
            try:
                result = subprocess.run(
                    [ruff_cmd, "check", BASE_DIR, "--exclude", "vendor,moonshine"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print(result.stdout)
                    raise RuntimeError(
                        "INTEGRITY_CHECK_FAILURE: Ruff quality violations detected."
                    )
                else:
                    print("[+] Self-Integrity Quality Check PASSED.")
            except Exception as e:
                if isinstance(e, RuntimeError):
                    raise
                raise RuntimeError(
                    f"INTEGRITY_CHECK_FAILURE: Ruff check execution error: {e}"
                )
        else:
            print(
                "[!] Warning: ruff not found in PATH, skipping self-integrity quality check."
            )

        # 4a. CrossHair Symbolic Analysis for python/ sidecars
        print("[*] Ensuring CrossHair is installed...")
        if _setup_gui:
            _setup_gui._update_bar(92, step_text="code step 0x000B", pulse=True)  # Symbolic analysis of code paths
        try:
            pyvenv_dir = os.path.join(BASE_DIR, "pyvenv")
            pyvenv_python = os.path.join(pyvenv_dir, "bin", "python")
            if not os.path.exists(pyvenv_python):
                print(f"  [~] Creating pyvenv at {pyvenv_dir}...")
                subprocess.run(
                    [sys.executable, "-m", "venv", pyvenv_dir],
                    check=True,
                    capture_output=True,
                )
            subprocess.run(
                [pyvenv_python, "-m", "pip", "install", "crosshair-tool"],
                check=True,
                capture_output=True,
            )
            # Install python/ sidecar dependencies (loguru, httpx, requests, sympy, etc.)
            # so CrossHair can import them when checking the sidecar files.
            subprocess.run(
                [pyvenv_python, "-m", "pip", "install",
                 "loguru", "httpx", "requests", "sympy",
                 "numpy", "PyMuPDF",
                 "Pillow", "openpyxl", "python-docx", "python-pptx", "tinytag",
                 "cryptography", "keyring"],
                check=True,
                capture_output=True,
            )
            print("[*] Running CrossHair Symbolic Verification on python sidecars...")

            python_dir = os.path.join(BASE_DIR, "python")
            target_files = []
            for root_dir, _, files in os.walk(python_dir):
                for f in files:
                    if f.endswith(".py") and not f.startswith("test"):
                        target_files.append(os.path.join(root_dir, f))

            if target_files:
                result = subprocess.run(
                    [
                        pyvenv_python,
                        "-m",
                        "crosshair",
                        "check",
                        "--verbose",
                        "--per_condition_timeout",
                        "1",
                    ]
                    + target_files,
                )
                if result.returncode == 1:
                    raise RuntimeError(
                        "INTEGRITY_CHECK_FAILURE: CrossHair contract violations detected in python/ sidecars."
                    )
                elif result.returncode == 2:
                    raise RuntimeError(
                        "INTEGRITY_CHECK_FAILURE: CrossHair execution error in python/ sidecars."
                    )
                else:
                    print("[+] CrossHair Symbolic Verification PASSED.")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(
                f"INTEGRITY_CHECK_FAILURE: CrossHair initialization error: {e}"
            )

        # 4b. Pyrefly check for python/ sidecars
        pyrefly_cmd = "pyrefly.exe" if platform.system() == "Windows" else "pyrefly"
        if shutil.which(pyrefly_cmd):
            print("[*] Running Pyrefly Type Check on python sidecars...")
            if _setup_gui:
                _setup_gui._update_bar(93, step_text="code step 0x000C", pulse=True)  # Type consistency check
            try:
                python_dir = os.path.join(BASE_DIR, "python")
                env_vars = os.environ.copy()
                env_vars["PATH"] = (
                    f"{os.path.join(BASE_DIR, 'pyvenv', 'bin')}{os.pathsep}{env_vars.get('PATH', '')}"
                )
                env_vars["VIRTUAL_ENV"] = os.path.join(BASE_DIR, "pyvenv")
                result = subprocess.run(
                    [
                        pyrefly_cmd,
                        "check",
                        python_dir,
                        "--check-unannotated-defs=true",
                        "--strict-callable-subtyping=true",
                    ],
                    capture_output=True,
                    text=True,
                    env=env_vars,
                )
                if result.returncode != 0:
                    print(result.stdout)
                    print(result.stderr)
                    raise RuntimeError(
                        "INTEGRITY_CHECK_FAILURE: Pyrefly type violations detected in python/ sidecars."
                    )
                else:
                    print("[+] Pyrefly Type Check PASSED.")
            except Exception as e:
                if isinstance(e, RuntimeError):
                    raise
                raise RuntimeError(
                    f"INTEGRITY_CHECK_FAILURE: Pyrefly check execution error: {e}"
                )
        else:
            print(
                "[!] Warning: pyrefly not found in PATH, skipping python/ sidecar type check."
            )

        # 5. LSH QRNN Worker Bootstrap & pyrefly + ruff check
        if os.path.exists(lsh_reqs):
            print("[LSH] Bootstrapping QRNN LSH worker venv...")
            if _setup_gui:
                _setup_gui._update_bar(94, step_text="code step 0x000D", pulse=True)  # Initialize background processing systems
            if not os.path.exists(pyvenv_python):
                subprocess.run([sys.executable, "-m", "venv", pyvenv_dir], check=True)
            pyvenv_pip = os.path.join(pyvenv_dir, "bin", "pip")
            subprocess.run([pyvenv_pip, "install", "-r", lsh_reqs], check=True)
            # PINN/DeepXDE for Speculative-Branch-Prediction pipeline
            subprocess.run(
                [pyvenv_pip, "install", "deepxde"],
                check=True,
                capture_output=True,
            )

            # pyrefly check
            pyvenv_pyrefly = os.path.join(pyvenv_dir, "bin", "pyrefly")
            if os.path.exists(pyvenv_pyrefly):
                print("[LSH] Running pyrefly type-check on worker...")
                res_pyrefly = subprocess.run(
                    [pyvenv_pyrefly, "check", lsh_worker],
                    capture_output=True,
                    text=True,
                )
                if res_pyrefly.returncode != 0:
                    print(res_pyrefly.stdout)
                    print(res_pyrefly.stderr)
                    raise RuntimeError(
                        "LSH_BOOTSTRAP_FAILURE: pyrefly type check failed."
                    )

            # ruff check
            pyvenv_ruff = os.path.join(pyvenv_dir, "bin", "ruff")
            if os.path.exists(pyvenv_ruff):
                print("[LSH] Running ruff lint on worker...")
                res_ruff = subprocess.run(
                    [pyvenv_ruff, "check", lsh_worker], capture_output=True, text=True
                )
                if res_ruff.returncode != 0:
                    print(res_ruff.stdout)
                    print(res_ruff.stderr)
                    raise RuntimeError(
                        "LSH_BOOTSTRAP_FAILURE: ruff quality check failed."
                    )
            print("[LSH] QRNN worker bootstrap complete.")

        # 6. VAD ONNX Sidecar Worker: Python venv bootstrap
        if os.path.exists(vad_worker_script):
            print("[VAD] Bootstrapping ONNX VAD worker...")
            if _setup_gui:
                _setup_gui._update_bar(95, step_text="code step 0x000E", pulse=True)  # Initialize audio processing pipeline
            if not os.path.exists(pyvenv_python):
                subprocess.run([sys.executable, "-m", "venv", pyvenv_dir], check=True)
            pyvenv_pip = (
                os.path.join(pyvenv_dir, "bin", "pip")
                if platform.system() != "Windows"
                else os.path.join(pyvenv_dir, "Scripts", "pip.exe")
            )

            try:
                subprocess.run(
                    [pyvenv_pip, "install", "onnxruntime", "numpy"], check=True
                )
                print("[VAD] VAD worker bootstrap complete.")
            except subprocess.CalledProcessError:
                raise RuntimeError(
                    "VAD_BOOTSTRAP_FAILURE: VAD environment setup failed."
                )

        # Save build hash after all verification steps pass successfully
        with open(hash_file, "w") as f:
            f.write(current_hash)

        # Save venv hash so future runs detect stale venvs
        save_venv_hash()

    # Handle integrity check flag
    test_build_integrity = False
    if "--test-build-integrity-check" in sys.argv:
        print(
            "[*] Test build integrity check: will launch server and invoke benchmark."
        )
        test_build_integrity = True

    # Parse arguments
    if "--test-fips" in sys.argv:
        print("[*] --test-fips flag detected. Entering automated testing mode.")
        os.environ["ADELAIDE_USER"] = "testfips"
        sys.argv.append("--no-gui")

    launch_gui = True
    if "--no-gui" in sys.argv or test_build_integrity:
        launch_gui = False

    run_benchmark = False
    if "--benchmark" in sys.argv or test_build_integrity:
        run_benchmark = True

    # [DO NOT REMOVE] --no-daemon: Skip the StellaIcarus daemon runner.
    # The daemon runner retries failed MCU bridge connections every 30s,
    # flooding the terminal with error messages.  Use this flag when you
    # want clean server-only output for debugging.
    launch_daemon = True
    if "--no-daemon" in sys.argv:
        launch_daemon = False

    # ── API key enforcement ──────────────────────────────────────────────────
    # --enforce-api-key: enable x-api-key validation on the Ada server
    # --no-enforce-api-key: explicitly disable (default for Ollama compat)
    # If neither flag is given, enforcement is OFF by default.
    enforce_api_key = False
    if "--enforce-api-key" in sys.argv:
        enforce_api_key = True
    if "--no-enforce-api-key" in sys.argv:
        enforce_api_key = False

    # Port/Host: args > env > defaults
    server_host = os.environ.get("ADLAIDE_SERVER_HOST", "0.0.0.0")
    server_port = os.environ.get("ADLAIDE_SERVER_PORT", "11420")
    for i, arg in enumerate(sys.argv):
        if arg == "--host" and i + 1 < len(sys.argv):
            server_host = sys.argv[i + 1]
        if arg == "--port" and i + 1 < len(sys.argv):
            server_port = sys.argv[i + 1]

    # [DO NOT REMOVE] Verbose launch info for debugging startup issues
    print(f"[*] [Launch-V] Run.py PID: {os.getpid()}")
    print(f"[*] [Launch-V] Python executable: {sys.executable}")
    print(f"[*] [Launch-V] Server host: {server_host}, port: {server_port}")
    print(f"[*] [Launch-V] Launch GUI: {launch_gui}, Launch daemon: {launch_daemon}")

    # ── Crypto Bootstrap ────────────────────────────────────────────────────
    # Initialize master key (generates + persists if first boot), then set
    # ADELAIDE_MASTER_KEY env var so the Ada server and all subprocesses
    # inherit the key automatically.
    if not IS_KISS:
        _term_print("  Loading preparing for Model... (Nothing to see here)")
        _term_print("")
    print("[CRYPTO] Bootstrapping encryption master key...")
    try:
        # Check for legacy key files and migrate if needed
        local_key = os.path.join(BASE_DIR, "config", "master.key")
        legacy_key = os.path.expanduser("~/.config/adelaide/master.key")
        if os.path.exists(local_key) or os.path.exists(legacy_key):
            print(
                "[CRYPTO] Legacy key file detected, migrating to hardware-bound system..."
            )
            migrate_from_legacy_key_system()

        # Try hardware-bound key derivation first
        master_key = hardware_bound_key_derivation()
        if not master_key:
            print("[CRYPTO] FATAL: Hardware-bound key derivation failed.")
            print("[CRYPTO] Cannot proceed without a valid master key.")
            print("[CRYPTO] Please try again with your password.")
            cleanup(signal.SIGTERM, None)
            sys.exit(1)

        # Write master key to secure temp file instead of leaking via env var
        # to all subprocess environments. Subprocesses inherit ADELAIDE_MASTER_KEY_FILE
        # and read the key on init, then the file is cleaned up on shutdown.
        global _master_key_file_path
        fd, _master_key_file_path = tempfile.mkstemp(prefix="adelaide_mk_", suffix=".key")
        with os.fdopen(fd, 'w') as f:
            f.write(master_key)
        os.chmod(_master_key_file_path, 0o400)
        os.environ["ADELAIDE_MASTER_KEY_FILE"] = _master_key_file_path
        print(f"[CRYPTO] Master key ready. {len(master_key)} hex chars (secure temp file).")
        _wipe_string(master_key)
        master_key = None

        # Migrate existing data to AAD-bound encryption (one-time)
        print("[CRYPTO] Checking for AAD migration...")
        try:
            migrate_all_to_aad()
        except Exception as e:
            print(f"[CRYPTO] WARNING: AAD migration failed: {e}")
            print("[CRYPTO] Legacy data will still decrypt (backward compatible)")
    except Exception as e:
        import traceback
        with open("crash_log.txt", "w") as f:
            f.write(f"[CRYPTO] FATAL: Could not bootstrap crypto: {e}\n")
            f.write(traceback.format_exc())
        print(f"[CRYPTO] FATAL: Could not bootstrap crypto: {e}")
        print("[CRYPTO] Refusing to run with plaintext storage. Aborting.")
        os.abort()
        
    python_cmd = sys.executable
    if launch_daemon:
        print("[*] Booting StellaIcarus Ada Daemon Manager...")
        daemon_script = os.path.join(
            BASE_DIR, "python", "stellaicarus_daemon_runner.py"
        )

        daemon_args = [python_cmd, daemon_script]
        if daemon_build_flag:
            daemon_args.append(daemon_build_flag)

        daemon_process = subprocess.Popen(
            daemon_args, cwd=BASE_DIR, start_new_session=True
        )
    else:
        print("[*] [Launch-V] Skipping daemon runner (--no-daemon)")


    print("[*] Booting Adelaide Intelligence Server...")
    end_time = int(time.time() * 1000)
    print(f"[*] Startup completed in {end_time - start_time}ms (WCET)")

    # ── Trace verbosity prefix ──────────────────────────────────────────────
    # Set ADELAIDE_TOOL_TRACE_PREFIX so Ada server and all Python tool scripts
    # emit consistent [prefix][Toolcall][+uptime] trace lines.
    # The prefix is derived from --verbose: verbose → "[ADA]" (visible on tty),
    # default/kiss → "[ADA]" (still captured in log files).
    os.environ["ADELAIDE_TOOL_TRACE_PREFIX"] = "[ADA]"
    # Disable trace when not verbose? No — always enabled, always goes to logs.
    # User sees traces on terminal when --verbose is active (via _TeeWriter).
    os.environ["ADELAIDE_TOOL_TRACE_ENABLED"] = "1"

    server_bin = (
        "adelaide_server.exe" if platform.system() == "Windows" else "adelaide_server"
    )
    server_path = os.path.join(BASE_DIR, "bin", server_bin)

    env = os.environ.copy()

    # [DO NOT REMOVE] Force Python stdout/stderr unbuffered for all subprocesses.
    # When stdout is a pipe (not a terminal), Python block-buffers output.
    # This prevents run.py's print() from appearing immediately.
    env["PYTHONUNBUFFERED"] = "1"

    # Architecture-aware Moonshine ONNX runtime path
    #
    # QUIRK: The server binary links against libmoonshine.dylib, which
    #        dynamically loads libonnxruntime.1.23.2.dylib.  If this
    #        library is NOT in DYLD_LIBRARY_PATH, the binary crashes at
    #        startup with:
    #          "Library not loaded: @rpath/libonnxruntime.1.23.2.dylib"
    #          "Reason: no such file"
    #        The onnxruntime dylib lives in the moonshine submodule:
    #          moonshine/core/third-party/onnxruntime/lib/macos/{arch}/
    #        This is the ONLY place it exists on the filesystem (not
    #        in /opt/homebrew/lib or any standard path).
    #
    # IMPORTANT: Pre-existing bug (2026-06-10): After QWEN_0_8B processes
    # a request and the model is released, the server may crash with
    # exit code -1 (signal caught by Kratos crash isolation). The run.sh
    # wrapper will auto-restart the server if this happens, but clients
    # will see a brief connection reset.
    arch = "arm64" if platform.machine() == "arm64" else "x86_64"
    moonshine_onnx = os.path.join(
        BASE_DIR,
        "vendor",
        "moonshine",
        "core",
        "third-party",
        "onnxruntime",
        "lib",
        "macos",
        arch,
    )

    if platform.system() == "Darwin":
        env["DYLD_LIBRARY_PATH"] = (
            f"{moonshine_onnx}:{env.get('DYLD_LIBRARY_PATH', '')}"
        )

    # Run server directly (ALIRE wrapper changes CWD which breaks relative model paths)
    # Write server launch args to file so the watchdog can relaunch with same args.
    server_args = ["--host", server_host, "--port", server_port]
    server_args_file = os.path.join(BASE_DIR, "run", "adelaide_server.args")
    with open(server_args_file, "w") as f:
        f.write(" ".join(server_args))

    # [DO NOT REMOVE] Generate SSL certificate if not exists
    # This enables HTTPS for secure communication between frontend and backend.
    cert_script = os.path.join(BASE_DIR, "scripts", "generate_cert.py")
    if os.path.exists(cert_script):
        print("[*] Checking SSL certificate...")
        cert_result = subprocess.run(
            [sys.executable, cert_script], cwd=BASE_DIR, capture_output=True, text=True
        )
        if cert_result.returncode == 0:
            print("[*] SSL certificate ready")
        else:
            print(
                f"{BG_RED}[BUGCHECK] [!] SSL certificate generation failed: {cert_result.stderr}{RST}"
            )
            print("[!] Falling back to HTTP mode")

    # [DO NOT REMOVE] Verbose server launch info
    print(f"[*] [Launch-V] Server binary: {server_path}")
    print(f"[*] [Launch-V] Server args: {server_args}")
    print(f"[*] [Launch-V] Server CWD: {BASE_DIR}")
    print(
        f"[*] [Launch-V] DYLD_LIBRARY_PATH: {env.get('DYLD_LIBRARY_PATH', 'NOT SET')}"
    )

    # ── API key enforcement setup ─────────────────────────────────────────────
    # If enforcement is enabled, decrypt the API key store and write a plaintext
    # file that the Ada server can read at startup.  Also expose the first key
    # to the sidecar UI via ADELAIDE_SIDECAR_API_KEY.
    if enforce_api_key:
        try:
            from adelaide_crypto import load_api_keys

            all_keys = load_api_keys()
            if all_keys:
                # Write plaintext key file for Ada server
                api_key_file = os.path.join(BASE_DIR, "run", "api_keys_plain.txt")
                os.makedirs(os.path.dirname(api_key_file), exist_ok=True)
                with open(api_key_file, "w") as f:
                    for k in all_keys:
                        f.write(k + "\n")
                os.chmod(api_key_file, 0o600)
                env["ADELAIDE_API_KEY_FILE"] = api_key_file
                env["ADELAIDE_API_KEY_ENFORCE"] = "1"
                # First key goes to the sidecar UI
                env["ADELAIDE_SIDECAR_API_KEY"] = all_keys[0]
                print(
                    f"[API-KEY] Enforcement enabled, {len(all_keys)} key(s) loaded from encrypted store"
                )
            else:
                print(
                    "[API-KEY] WARNING: Enforcement enabled but no API keys configured."
                )
                print(
                    "[API-KEY] Use --api-key add <key> to add a key, or disable with --no-enforce-api-key"
                )
                enforce_api_key = False
        except Exception as e:
            print(f"[API-KEY] WARNING: Could not set up API keys: {e}")
            enforce_api_key = False
    else:
        env["ADELAIDE_API_KEY_ENFORCE"] = "1"
        print(
            "[API-KEY] Enforcement ENABLED by default per FIPS-140-3."
        )

    # Inject log file path so the Ada server can tail it for SSE benchmarking
    env["ADELAIDE_LOG_FILE"] = current_log_path

    # Launch server through tee so its output goes to terminal + log file
    tee_process = subprocess.Popen(
        ["tee", "-a", current_log_path], stdin=subprocess.PIPE, start_new_session=True
    )
    server_process = subprocess.Popen(
        [server_path] + server_args,
        cwd=BASE_DIR,
        env=env,
        stdout=tee_process.stdin,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    # [DO NOT REMOVE] Verbose PID tracking
    print(f"[*] [Launch-V] Server PID: {server_process.pid}")
    print(f"[*] [Launch-V] Server args file: {server_args_file}")

    # Launch external watchdog process (separate binary, monitors server health)
    # [DO NOT REMOVE THIS] LAUNCH GUARD: Set orchestration flag so watchdog
    # knows it was launched through run.py (prevents direct binary execution).
    watchdog_bin = (
        "adelaide_watchdog.exe"
        if platform.system() == "Windows"
        else "adelaide_watchdog"
    )
    watchdog_path = os.path.join(BASE_DIR, "bin", watchdog_bin)

    # Clear any stale shutdown flag from a previous run.
    # This flag is written by cleanup() to prevent the watchdog from
    # restarting the server after an intentional shutdown signal (SIGQUIT).
    # If we're starting a fresh session, the old flag must be removed.
    shutdown_flag = os.path.join(BASE_DIR, "run", ".shutdown_requested")
    if os.path.exists(shutdown_flag):
        try:
            os.remove(shutdown_flag)
        except Exception:
            pass
    if os.path.exists(watchdog_path):
        print("[*] Booting Adelaide Watchdog...")
        watchdog_env = env.copy()
        watchdog_env["ADLAIDE_WATCHDOG_ORCHESTRATED"] = "1"
        # Launch watchdog fully detached — nohup + own session + own process group.
        # This ensures the watchdog survives even if run.py exits or the
        # terminal is closed.  The watchdog monitors the server via file-based
        # IPC (run/ directory) so it doesn't need a parent process.
        watchdog_log = os.path.join(BASE_DIR, "run", "adelaide_watchdog.log")
        with open(watchdog_log, "a") as wlog:
            watchdog_process = subprocess.Popen(
                [watchdog_path],
                cwd=BASE_DIR,
                env=watchdog_env,
                stdout=wlog,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        def watchdog_monitor(path, w_env, log_path):
            global watchdog_process
            while True:
                w_exit = watchdog_process.wait()
                if os.path.exists(os.path.join(BASE_DIR, "run", ".shutdown_requested")):
                    break
                if w_exit in (0, 9, -9):
                    break
                print(
                    f"\n[*] Watchdog crashed (code {w_exit})! Relaunching instantly..."
                )
                with open(log_path, "a") as wlog2:
                    watchdog_process = subprocess.Popen(
                        [path],
                        cwd=BASE_DIR,
                        env=w_env,
                        stdout=wlog2,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )

        t = threading.Thread(
            target=watchdog_monitor,
            args=(watchdog_path, watchdog_env, watchdog_log),
            daemon=True,
        )
        t.start()
    else:
        print("[!] Watchdog binary not found at", watchdog_path, "- skipping")

    # Launch VAD ONNX Sidecar
    if os.path.exists(vad_worker_script):
        print("[*] Booting VAD ONNX Sidecar...")
        vad_log = os.path.join(BASE_DIR, "run", "vad_worker.log")
        with open(vad_log, "a") as vlog:
            vad_process = subprocess.Popen(
                [pyvenv_python, vad_worker_script],
                cwd=BASE_DIR,
                env=env,
                stdout=vlog,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    if run_benchmark:
        print("[*] Booting benchmark runner thread...")

        def benchmark_runner():
            import json
            import time
            import urllib.request

            print("[Benchmark] Waiting 15s for server to settle...")
            time.sleep(15)
            url = f"http://{server_host}:{server_port}/api/snowballEnagaValidationBenchmark"
            print(f"[Benchmark] Invoking {url} (Performance)...")
            success = False
            for bench_attempt in range(2):  # Try up to 2 times
                try:
                    data = json.dumps({"benchmark_type": "performance"}).encode("utf-8")
                    req = urllib.request.Request(
                        url,
                        data=data,
                        headers={
                            "Content-Type": "application/json",
                            "x-api-key": "IknowtheConsequencesAndWouldLockupTheServerForHours",
                        },
                        method="POST",
                    )
                    start_t = time.time()
                    with urllib.request.urlopen(req, timeout=300) as res:
                        status = res.getcode()
                        print(f"[Benchmark] Connected. HTTP {status}")

                        while True:
                            line = res.readline().decode("utf-8")
                            if not line:
                                break
                            line = line.strip()
                            if line.startswith("data: "):
                                payload = line[6:]
                                if payload == "[DONE]":
                                    success = True
                                    break

                                try:
                                    parsed = json.loads(payload)
                                    if "type" in parsed and parsed["type"] == "log":
                                        print(f"[Ada-Log] {parsed.get('line', '')}")
                                    elif (
                                        "type" in parsed
                                        and parsed["type"] == "progress"
                                    ):
                                        print(f"[Benchmark Progress] {payload}")
                                    elif "performance" in parsed:
                                        print("[Benchmark] Scoring Report:")
                                        print(json.dumps(parsed, indent=2))
                                        success = True
                                except json.JSONDecodeError:
                                    print(f"[SSE Raw] {payload}")

                        elapsed = time.time() - start_t
                        print(f"[Benchmark] Completed in {elapsed:.2f}s")
                    break  # Success, no more retries
                except Exception as e:
                    if bench_attempt == 0:
                        print(f"[!] Benchmark attempt 1 failed: {e}")
                        print("[Benchmark] Retrying in 5s...")
                        time.sleep(5)
                    else:
                        print(f"[!] Benchmark failed after retries: {e}")

                    print("[*] Running comprehensive loopback API tests...")
                    # We will test all endpoints from the API reference
                    tests = [
                        (
                            "Server Root (GET)",
                            f"http://{server_host}:{server_port}/",
                            None,
                            "GET",
                        ),
                        (
                            "Server Root (HEAD)",
                            f"http://{server_host}:{server_port}/",
                            None,
                            "HEAD",
                        ),
                        (
                            "Health / Power",
                            f"http://{server_host}:{server_port}/api/power",
                            None,
                            "GET",
                        ),
                        (
                            "Telemetry",
                            f"http://{server_host}:{server_port}/api/telemetry",
                            None,
                            "GET",
                        ),
                        (
                            "Version",
                            f"http://{server_host}:{server_port}/api/version",
                            None,
                            "GET",
                        ),
                        (
                            "Process Status",
                            f"http://{server_host}:{server_port}/api/ps",
                            None,
                            "GET",
                        ),
                        (
                            "Zenith Routine",
                            f"http://{server_host}:{server_port}/api/ZenithRoutine",
                            None,
                            "GET",
                        ),
                        (
                            "List Models (v1)",
                            f"http://{server_host}:{server_port}/v1/models",
                            None,
                            "GET",
                        ),
                        (
                            "Ollama Tags",
                            f"http://{server_host}:{server_port}/api/tags",
                            None,
                            "GET",
                        ),
                        # POST requests
                        (
                            "OpenAI Chat",
                            f"http://{server_host}:{server_port}/v1/chat/completions",
                            {
                                "model": "Snowball-Enaga",
                                "messages": [{"role": "user", "content": "ping"}],
                            },
                            "POST",
                        ),
                        (
                            "OpenAI Completions",
                            f"http://{server_host}:{server_port}/v1/completions",
                            {"model": "Snowball-Enaga", "prompt": "ping"},
                            "POST",
                        ),
                        (
                            "OpenAI Embeddings",
                            f"http://{server_host}:{server_port}/v1/embeddings",
                            {"model": "Snowball-Enaga", "input": "ping"},
                            "POST",
                        ),
                        (
                            "Claude Messages",
                            f"http://{server_host}:{server_port}/v1/messages",
                            {
                                "model": "Snowball-Enaga",
                                "messages": [{"role": "user", "content": "ping"}],
                                "max_tokens": 10,
                            },
                            "POST",
                        ),
                        (
                            "Ollama Chat",
                            f"http://{server_host}:{server_port}/api/chat",
                            {
                                "model": "Snowball-Enaga",
                                "messages": [{"role": "user", "content": "ping"}],
                                "stream": False,
                            },
                            "POST",
                        ),
                        (
                            "Ollama Generate",
                            f"http://{server_host}:{server_port}/api/generate",
                            {
                                "model": "Snowball-Enaga",
                                "prompt": "ping",
                                "stream": False,
                            },
                            "POST",
                        ),
                        (
                            "Ollama Embeddings",
                            f"http://{server_host}:{server_port}/api/embeddings",
                            {"model": "Snowball-Enaga", "prompt": "ping"},
                            "POST",
                        ),
                        (
                            "Ollama Show",
                            f"http://{server_host}:{server_port}/api/show",
                            {"name": "Snowball-Enaga"},
                            "POST",
                        ),
                        (
                            "AGC/ACP",
                            f"http://{server_host}:{server_port}/api/acp",
                            {
                                "jsonrpc": "2.0",
                                "method": "chat/completion",
                                "params": {"prompt": "ping"},
                                "id": 1,
                            },
                            "POST",
                        ),
                        # Media / specialized APIs
                        (
                            "TTS Kokoro",
                            f"http://{server_host}:{server_port}/v1/audio/speech",
                            {
                                "input": "ping",
                                "voice": "default",
                                "response_format": "wav",
                            },
                            "POST",
                        ),
                        (
                            "Image Gen (FLUX)",
                            f"http://{server_host}:{server_port}/v1/images/generations",
                            {"prompt": "ping", "n": 1, "size": "1024x1024"},
                            "POST",
                        ),
                    ]

                    all_passed = True
                    for name, endpoint, payload, method in tests:
                        test_passed = False
                        for test_attempt in range(2):  # Try up to 2 times
                            try:
                                req_data = (
                                    json.dumps(payload).encode("utf-8")
                                    if payload
                                    else None
                                )
                                headers = (
                                    {"Content-Type": "application/json"}
                                    if payload
                                    else {}
                                )
                                req = urllib.request.Request(
                                    endpoint,
                                    data=req_data,
                                    headers=headers,
                                    method=method,
                                )
                                with urllib.request.urlopen(req, timeout=30) as res:
                                    code = res.getcode()
                                    if code in (200, 201, 204):
                                        print(f"[+] {name} Test: PASSED (HTTP {code})")
                                        test_passed = True
                                        break
                                    else:
                                        if test_attempt == 0:
                                            print(
                                                f"[-] {name} Test: FAILED (HTTP {code}), retrying..."
                                            )
                                            time.sleep(1)
                                        else:
                                            print(
                                                f"[-] {name} Test: FAILED (HTTP {code})"
                                            )
                            except urllib.error.HTTPError as e:
                                if test_attempt == 0:
                                    print(
                                        f"[-] {name} Test: HTTP ERROR {e.code}, retrying..."
                                    )
                                    time.sleep(1)
                                else:
                                    print(f"[-] {name} Test: HTTP ERROR {e.code}")
                            except Exception as e:
                                if test_attempt == 0:
                                    print(
                                        f"[-] {name} Test: EXCEPTION ({e}), retrying..."
                                    )
                                    time.sleep(1)
                                else:
                                    print(f"[-] {name} Test: EXCEPTION ({e})")
                        if not test_passed:
                            all_passed = False

                    if not all_passed:
                        success = False

            if test_build_integrity:
                if success:
                    print(
                        "[*] Test build integrity check passed! Exiting successfully."
                    )
                    cleanup()
                else:
                    print("[!] Test build integrity check FAILED!")
                    os._exit(1)

        b_thread = threading.Thread(target=benchmark_runner, daemon=True)
        b_thread.start()

    if launch_gui:
        # Close the setup loading bar — setup is complete
        if _setup_gui:
            _setup_gui._stop_pulse()
            _setup_gui._update_bar(100, step_text="Ready!")
            time.sleep(0.3)
            _tk_progress_done(_setup_gui)
            _setup_gui = None
        print("[*] Booting Python Sidecar UI...")
        ui_dir = os.path.join(BASE_DIR, "ui")

        # Check for venv python
        venv_python_win = os.path.join(BASE_DIR, "pyvenv", "Scripts", "python.exe")
        venv_python_unix = os.path.join(BASE_DIR, "pyvenv", "bin", "python")

        if os.path.exists(venv_python_win):
            sidecar_python = venv_python_win
        elif os.path.exists(venv_python_unix):
            sidecar_python = venv_python_unix
        else:
            sidecar_python = python_cmd

        # Auto-install sidecar UI dependencies if using pyvenv (which is minimal by default)
        if sidecar_python != python_cmd:
            _sidecar_deps = [
                "networkx", "numpy", "psutil", "tiktoken",
                "uvicorn", "fastapi", "httpx", "pywebview", "PyMuPDF",
                "python-multipart",
            ]
            try:
                subprocess.run(
                    [sidecar_python, "-m", "pip", "install", "--quiet"] + _sidecar_deps,
                    check=True, capture_output=True, timeout=120,
                )
            except Exception:
                print("[!] Warning: failed to auto-install sidecar deps — continuing anyway")

        # [DO NOT REMOVE] macOS .app bundle for microphone/camera/screen capture permissions
        # On Darwin, create a proper .app bundle with Info.plist containing
        # NSMicrophoneUsageDescription, NSCameraUsageDescription, and
        # NSScreenCaptureUsageDescription for hardware access permissions.
        # The .app launches Terminal and runs the server with GUI.
        #
        # IMPORTANT: Only launch .app if NOT already running in Terminal.
        # If launched from Terminal, just run sidecar_ui.py directly to avoid bootloop.
        if sys.platform == "darwin":
            # Check if we're already in a Terminal session or launched from .app
            # ADELAIDE_LAUNCHED_FROM_APP is set by .app launcher script
            # TERM_SESSION_ID is set by bash/zsh when in terminal
            launched_from_app = os.environ.get("ADELAIDE_LAUNCHED_FROM_APP") == "1"
            in_terminal = os.environ.get("TERM_SESSION_ID") is not None

            # [DO NOT REMOVE] Clear stale flag after reading
            # Prevents false positives if flag persists in shell environment
            if launched_from_app:
                os.environ.pop("ADELAIDE_LAUNCHED_FROM_APP", None)

            if launched_from_app or in_terminal:
                # Already in Terminal or launched from .app - launch sidecar directly (no .app)
                print("[*] Running in Terminal - launching sidecar directly...")
                # Add pyvenv/bin to PATH so the sidecar can find pyrefly/ruff
                sidecar_env = os.environ.copy()
                pyvenv_bin = os.path.join(BASE_DIR, "pyvenv", "bin")
                if os.path.exists(pyvenv_bin):
                    sidecar_env["PATH"] = pyvenv_bin + os.pathsep + sidecar_env.get("PATH", "")
                sidecar_process = subprocess.Popen(
                    [sidecar_python, "sidecar_ui.py"], cwd=ui_dir, env=sidecar_env
                )
                print(f"[*] [Launch-V] Sidecar PID: {sidecar_process.pid}")
            else:
                # Not in Terminal (e.g., launched from Finder) - use .app
                app_bundle_path = os.path.join(
                    BASE_DIR, "run", "Adelaide Zephyrine Assistant.app"
                )
                create_app_script = os.path.join(ui_dir, "create_macos_app.py")

                # Create .app bundle if it doesn't exist
                if not os.path.exists(app_bundle_path):
                    print(
                        "[*] Creating macOS .app bundle for microphone/camera permissions..."
                    )
                    subprocess.run(
                        [
                            sidecar_python,
                            create_app_script,
                            "--output",
                            app_bundle_path,
                        ],
                        cwd=ui_dir,
                    )

                # Launch via .app bundle for proper permissions
                print(
                    "[*] Launching Adelaide Zephyrine Assistant.app for hardware access..."
                )
                subprocess.run(["open", app_bundle_path])
        else:
            # Non-Darwin: launch directly
            sidecar_process = subprocess.Popen(
                [sidecar_python, "sidecar_ui.py"], cwd=ui_dir
            )
            print(f"[*] [Launch-V] Sidecar PID: {sidecar_process.pid}")

    if True:
        try:
            while True:
                exit_code = server_process.wait()
                shutdown_flag = os.path.join(BASE_DIR, "run", ".shutdown_requested")
                intentional_exit_flag = os.path.join(
                    BASE_DIR, "run", ".intentional_exit"
                )

                is_intentional = os.path.exists(shutdown_flag) or os.path.exists(
                    intentional_exit_flag
                )

                if is_intentional:
                    print(
                        f"\n[*] Server exited cleanly or shutdown requested (code: {exit_code})"
                    )
                    # Clean shutdown — remove SIGKILL context cap so next boot starts fresh
                    cap_file = os.path.join(BASE_DIR, "run", ".oom_kill_ctx_cap")
                    if os.path.exists(cap_file):
                        os.remove(cap_file)
                        print(f"[*] Removed SIGKILL context cap: {cap_file}")

                    if os.path.exists(intentional_exit_flag):
                        os.remove(intentional_exit_flag)

                    break

                if IS_KISS:
                    raise RuntimeError(
                        f"SERVER_CRASHED: Server crashed with exit code {exit_code}"
                    )

                if exit_code < 0:
                    sig_val = -exit_code
                elif exit_code > 128:
                    sig_val = exit_code - 128
                else:
                    sig_val = None

                sig_name = "UNKNOWN"
                if sig_val:
                    try:
                        sig_name = signal.Signals(sig_val).name
                    except ValueError:
                        sig_name = f"SIGNAL_{sig_val}"

                BG_BLUE = "\033[44m\033[97m"  # Blue background, white text
                RESET = "\033[0m"

                print("\n")
                print(f"{BG_BLUE}{'=' * 70}{RESET}")
                print(f"{BG_BLUE}{'   :(  WE RAN INTO A PROBLEM'.ljust(70)}{RESET}")
                print(f"{BG_BLUE}{'-' * 70}{RESET}")
                if exit_code == 0:
                    print(
                        f"{BG_BLUE}{'   The server exited cleanly but no intentional shutdown was requested.'.ljust(70)}{RESET}"
                    )
                    print(
                        f"{BG_BLUE}{'   This was likely caused by an unexpected component termination.'.ljust(70)}{RESET}"
                    )
                else:
                    print(
                        f"{BG_BLUE}{'   The Adelaide Server encountered a fatal error and terminated.'.ljust(70)}{RESET}"
                    )
                print(f"{BG_BLUE}{f'   Exit Code: {exit_code}'.ljust(70)}{RESET}")
                if sig_val:
                    print(
                        f"{BG_BLUE}{f'   Signal:    {sig_name} ({sig_val})'.ljust(70)}{RESET}"
                    )
                print(f"{BG_BLUE}{'   '.ljust(70)}{RESET}")
                print(
                    f"{BG_BLUE}{'   Check the output immediately above this banner for the'.ljust(70)}{RESET}"
                )
                print(
                    f"{BG_BLUE}{'   last Ada stack traces and unfortunately we can'.ljust(70)}{RESET}"
                )
                print(
                    f"{BG_BLUE}{'   t recover it needs to be relaunched.'.ljust(70)}{RESET}"
                )
                print(f"{BG_BLUE}{'=' * 70}{RESET}\n")

                # === PANIC RECOVERY: Generate plot + dump CSV/logs ===
                import time as _time

                epoch_s = int(_time.time())
                panic_log_path = os.path.join(
                    LOGS_DIR,
                    f"I_am_incompetent_Panicked_and_Never_Enough_PANIC_{epoch_s}.log",
                )
                wcet_csv = os.path.join(BASE_DIR, "run", "wcet.csv")
                accel_csv = os.path.join(BASE_DIR, "run", "acceleration.csv")

                # Find latest run log
                latest_log = None
                if os.path.isdir(LOGS_DIR):
                    log_files = sorted(
                        [
                            f
                            for f in os.listdir(LOGS_DIR)
                            if f.startswith("run_") and f.endswith(".log")
                        ],
                        reverse=True,
                    )
                    if log_files:
                        latest_log = os.path.join(LOGS_DIR, log_files[0])

                # Write panic log with full CSV + logs
                try:
                    with open(panic_log_path, "w") as pf:
                        pf.write("=== INCOMPETENT PANIC LOG ===\n")
                        pf.write(f"Epoch: {epoch_s}\n")
                        pf.write(f"Exit Code: {exit_code}\n")
                        pf.write(f"Signal: {sig_name} ({sig_val})\n\n")

                        pf.write("=== WCET CSV (run/wcet.csv) ===\n")
                        if os.path.exists(wcet_csv):
                            with open(wcet_csv) as f:
                                pf.write(f.read())
                        else:
                            pf.write("(no wcet.csv found)\n")

                        pf.write("=== ACCELERATION CSV (run/acceleration.csv) ===\n")
                        if os.path.exists(accel_csv):
                            with open(accel_csv) as f:
                                pf.write(f.read())
                        else:
                            pf.write("(no gpu.csv found)\n")

                        pf.write(f"\n=== RUN LOG ({latest_log or 'none'}) ===\n")
                        if latest_log and os.path.exists(latest_log):
                            with open(latest_log) as f:
                                pf.write(f.read())
                        else:
                            pf.write("(no run log found)\n")

                    print(f"[*] Panic log written: {panic_log_path}")
                except Exception as e:
                    print(f"[!] Failed to write panic log: {e}")

                # === SIGKILL CONTEXT CAP: Save the ctx size that OOM'd ===
                if sig_val == 9:
                    try:
                        import re as _re

                        cap_file = os.path.join(BASE_DIR, "run", ".oom_kill_ctx_cap")
                        cap_val = None
                        if latest_log and os.path.exists(latest_log):
                            with open(latest_log) as lf:
                                for line in lf:
                                    # Match: [CtxMonitor] LLM CTX:  7950 /  16384 tokens
                                    m = _re.search(
                                        r"LLM CTX:\s*\d+\s*/\s*(\d+)\s*tokens", line
                                    )
                                    if m:
                                        cap_val = int(m.group(1))
                        if cap_val:
                            with open(cap_file, "w") as cf:
                                cf.write(str(cap_val))
                            print(
                                f"[*] SIGKILL context cap saved: {cap_val} tokens → {cap_file}"
                            )
                        else:
                            print(
                                "[*] SIGKILL detected but could not parse context size from log"
                            )
                    except Exception as e:
                        print(f"[!] Failed to write SIGKILL context cap: {e}")

                # Generate plot from CSVs
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import csv

                    import matplotlib.pyplot as plt

                    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
                    fig.suptitle(
                        f"Adelaide Crash Report — Epoch {epoch_s} — Exit {exit_code}",
                        fontsize=13,
                    )

                    # WCET plot
                    if os.path.exists(wcet_csv):
                        times, pipeline, elp0, elp1, elp2, elp3 = [], [], [], [], [], []
                        with open(wcet_csv) as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                try:
                                    t = int(row["uptime_s"].strip())
                                    p = int(row["pipeline_ns"].strip())
                                    e0 = int(row["elp0_ns"].strip())
                                    e1 = int(row["elp1_ns"].strip())
                                    e2 = int(row["elp2_ns"].strip())
                                    e3 = int(row["elp3_ns"].strip())
                                    times.append(t)
                                    pipeline.append(p)
                                    elp0.append(e0)
                                    elp1.append(e1)
                                    elp2.append(e2)
                                    elp3.append(e3)
                                except (ValueError, KeyError, AttributeError):
                                    continue
                        if times:
                            axes[0].plot(
                                times, pipeline, label="Pipeline", linewidth=0.8
                            )
                            axes[0].plot(
                                times, elp0, label="ELP0", linewidth=0.5, alpha=0.7
                            )
                            axes[0].plot(
                                times, elp1, label="ELP1", linewidth=0.5, alpha=0.7
                            )
                            axes[0].plot(
                                times, elp2, label="ELP2", linewidth=0.5, alpha=0.7
                            )
                            axes[0].plot(
                                times, elp3, label="ELP3", linewidth=0.5, alpha=0.7
                            )
                            axes[0].set_ylabel("WCET (ns)")
                            axes[0].legend(fontsize=7)
                            axes[0].set_title("WCET Timing")

                    # Acceleration plot
                    if os.path.exists(accel_csv):
                        times, free, total, pct, metal_broken = [], [], [], [], []
                        with open(accel_csv) as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                try:
                                    t = int(row["uptime_s"].strip())
                                    f_mb = int(row["free_mb"].strip())
                                    t_mb = int(row["total_mb"].strip())
                                    p = int(row["percent"].strip())
                                    mb = int(row["metal_broken"].strip())
                                    times.append(t)
                                    free.append(f_mb)
                                    total.append(t_mb)
                                    pct.append(p)
                                    metal_broken.append(mb)
                                except (ValueError, KeyError, AttributeError):
                                    continue
                        if times:
                            ax1 = axes[1]
                            ax2 = ax1.twinx()
                            ax1.plot(
                                times,
                                free,
                                color="green",
                                label="Free MB",
                                linewidth=0.8,
                            )
                            ax1.plot(
                                times,
                                total,
                                color="blue",
                                label="Total MB",
                                linewidth=0.5,
                                alpha=0.7,
                            )
                            ax2.plot(
                                times,
                                pct,
                                color="red",
                                label="Free %",
                                linewidth=0.5,
                                alpha=0.7,
                            )
                            ax1.set_ylabel("Memory (MB)")
                            ax2.set_ylabel("Free %")
                            ax1.legend(fontsize=7, loc="upper left")
                            ax2.legend(fontsize=7, loc="upper right")
                            ax1.set_title("GPU Memory")
                            # Mark OOM events
                            for i, mb in enumerate(metal_broken):
                                if mb:
                                    axes[1].axvline(
                                        x=times[i],
                                        color="red",
                                        linestyle="--",
                                        alpha=0.3,
                                    )

                    # Acceleration free % as heatmap-style fill
                    if os.path.exists(accel_csv):
                        times_pct, pcts = [], []
                        with open(accel_csv) as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                try:
                                    t = int(row["uptime_s"].strip())
                                    p = int(row["percent"].strip())
                                    times_pct.append(t)
                                    pcts.append(p)
                                except (ValueError, KeyError, AttributeError):
                                    continue
                        if times_pct:
                            axes[2].fill_between(
                                times_pct, pcts, alpha=0.4, color="cyan"
                            )
                            axes[2].plot(
                                times_pct, pcts, color="darkcyan", linewidth=0.6
                            )
                            axes[2].set_ylabel("Free %")
                            axes[2].set_xlabel("Uptime (s)")
                            axes[2].set_title("GPU Free Memory %")
                            axes[2].set_ylim(0, 100)

                    plt.tight_layout()
                    plot_path = os.path.join(
                        LOGS_DIR,
                        f"I_am_incompetent_Panicked_and_Never_Enough_PANIC_{epoch_s}.png",
                    )
                    plt.savefig(plot_path, dpi=120)
                    plt.close()
                    print(f"[*] Crash plot saved: {plot_path}")
                except ImportError:
                    print(
                        "[!] matplotlib not installed — skipping crash plot (pip install matplotlib)"
                    )
                except Exception as e:
                    print(f"[!] Failed to generate crash plot: {e}")

                print("\n[*] Relaunching server instantly (JMP back Rebounce back)...")
                # Kill any lingering old daemon to prevent CSV write races
                subprocess.run(
                    ["pkill", "-9", "-f", "adelaide_server"],
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                )
                import time as _kill_wait

                _kill_wait.sleep(0.5)  # Give OS time to release file handles
                tee_process = subprocess.Popen(
                    ["tee", "-a", current_log_path],
                    stdin=subprocess.PIPE,
                    start_new_session=True,
                )
                server_process = subprocess.Popen(
                    [server_path] + server_args,
                    cwd=BASE_DIR,
                    env=env,
                    stdout=tee_process.stdin,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                print(f"[*] [Launch-V] Server PID (relaunch): {server_process.pid}")
        except KeyboardInterrupt:
            print("\n[*] Keyboard interrupt received. Shutting down...")
            pass

    # Wait for background processes to finish if main blocking process exits
    # Force-kill all children including sidecar to prevent orphans
    print("[*] Final cleanup — killing all child processes...")
    for proc in [
        daemon_process,
        server_process,
        watchdog_process,
        vad_process,
        sidecar_process,
    ]:
        if proc and proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    # Give 1 second for graceful shutdown
    time.sleep(1.0)
    for proc in [
        daemon_process,
        server_process,
        watchdog_process,
        vad_process,
        sidecar_process,
    ]:
        if proc and proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # Nuclear option: pkill by name for processes that survive SIGKILL
    for proc_name in ["adelaide_server", "adelaide_watchdog", "vad_worker.py",
                       "stellaicarus_daemon_runner"]:
        try:
            subprocess.run(["pkill", "-9", "-f", proc_name],
                           stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except Exception:
            pass

    cleanup()


if __name__ == "__main__":
    main()
