import subprocess
import os
import sys
import gc

# Global Performance Tuning: Disable Garbage Collection
gc.disable()

class AdelaideBridge:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.process = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Handle running from root directory or from Adelaide_Lite/python directory
        if os.path.basename(base_dir) == "python" and os.path.basename(os.path.dirname(base_dir)) == "Adelaide_Lite":
            self.binary_path = os.path.join(os.path.dirname(base_dir), "bin", "adelaide_lite")
        else:
            self.binary_path = os.path.join(base_dir, "Adelaide_Lite", "bin", "adelaide_lite")
            
        self.start_process()

    def start_process(self):
        if os.path.exists(self.binary_path):
            try:
                self.process = subprocess.Popen(
                    [self.binary_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                # Read the initial "[+] Adelaide_Lite ready." line
                if self.process is not None and self.process.stdout is not None:
                    ready_line = self.process.stdout.readline().strip()
                    if "[+] Adelaide_Lite ready." not in ready_line:
                        self.process = None
                else:
                    self.process = None
            except Exception as e:
                print(f"⚠️ Failed to start Adelaide_Lite core: {e}", file=sys.stderr)
                self.process = None
        else:
            self.process = None

    def cosine_similarity(self, v1, v2):
        if self.process is None or self.process.poll() is not None:
            self.start_process()
            if self.process is None:
                return None  # Fallback to Python/numpy

        try:
            dim = len(v1)
            # Format inputs to avoid potential scientific notation issues in Ada parser
            v1_str = " ".join(f"{float(x):.10f}" for x in v1)
            v2_str = " ".join(f"{float(x):.10f}" for x in v2)
            
            # Send command and data
            if self.process is not None and self.process.stdin is not None and self.process.stdout is not None:
                self.process.stdin.write("similarity\n")
                self.process.stdin.write(f"{dim} {v1_str} {v2_str}\n")
                self.process.stdin.flush()
                
                # Read response
                resp = self.process.stdout.readline().strip()
                if resp.startswith("SIMILARITY:"):
                    val_str = resp.split(":")[1].strip()
                    return float(val_str)
        except Exception as e:
            print(f"⚠️ Adelaide_Lite IPC error: {e}", file=sys.stderr)
            # Try to restart for next call
            self.start_process()
            
        return None
