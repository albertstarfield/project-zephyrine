# ./StellaIcarus/basic_matrix_math_hook.py
import re
import sys
import time
import ast
from typing import Optional, Match

# --- ACCELERATION ENGINE SETUP ---
# Detects the most powerful available hardware backend.
_TORCH_AVAILABLE = False
_DEVICE = "cpu"
_ENGINE_NAME = "Python/CPU (Basic)"

try:
    import torch
    _TORCH_AVAILABLE = True
    
    if torch.cuda.is_available():
        _DEVICE = "cuda" # Covers NVIDIA CUDA and often AMD ROCm if pytorch-rocm is installed
        _ENGINE_NAME = f"GPU (CUDA/ROCm) - {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _DEVICE = "mps" # Apple Silicon (M1/M2/M3) Metal Performance Shaders
        _ENGINE_NAME = "GPU (Apple MPS)"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        _DEVICE = "xpu" # Intel OneAPI
        _ENGINE_NAME = "GPU (Intel XPU)"
    else:
        _DEVICE = "cpu" # Fallback to highly optimized AVX/MKL CPU kernels
        _ENGINE_NAME = "CPU (Torch AVX/MKL)"

    print(f"✅ StellaMatrixHook: Active Engine -> {_ENGINE_NAME}", file=sys.stderr)

except ImportError:
    print("⚠️ StellaMatrixHook: PyTorch not found. Matrix Ops disabled.", file=sys.stderr)


# --- ROBUST REGEX PATTERN ---
# Matches two list-like structures [[...]] with an optional operator in between.
PATTERN = re.compile(
    r"(?is)"
    r".*?" # Pre-noise
    
    # Matrix A: Capture logical nested brackets [[1,2],[3,4]]
    # We use a greedy match for the content inside, but bounded by outer brackets.
    r"(?P<mat1>\[\s*\[.*?\]\s*\])"
    
    r"\s*"
    
    # Operator (Optional, defaults to MatMul if missing but space exists)
    r"(?P<op>"
        r"[\+\-\*@xX]" # @ is the python matrix mult operator
        r"|"
        r"(?:plus|add|minus|subtract|times|multiplied\s+by|dot|matmul)"
    r")?"
    
    r"\s*"
    
    # Matrix B
    r"(?P<mat2>\[\s*\[.*?\]\s*\])"
    
    r".*" # Post-noise
)

def _parse_matrix_string(mat_str: str):
    """Safely parses a string representation of a list of lists."""
    try:
        # ast.literal_eval is safer than eval() and more flexible than json.loads()
        # It handles Python-style spacing and numbers well.
        return ast.literal_eval(mat_str.strip())
    except Exception:
        return None

def _format_tensor(tensor) -> str:
    """Formats the result tensor into a readable string."""
    # Convert back to python list for clean formatting
    lst = tensor.tolist()
    
    # If it's huge, truncate it for the chat window
    if len(lst) > 10 or len(lst[0]) > 10:
        return f"Result Matrix shape: {tensor.shape} (Too large to display fully)"
    
    # Format nicely
    rows = [str(row) for row in lst]
    return "[\n  " + ",\n  ".join(rows) + "\n]"

# --- MAIN HANDLER ---
def handler(match: Match[str], user_input: str, session_id: str) -> Optional[str]:
    if not _TORCH_AVAILABLE:
        return "I need my PyTorch upgrades to perform matrix calculations."

    SUCCESS_PREFIX = "Here is the resulting matrix:\n"
    ERROR_PREFIX = "Matrix Error:"

    try:
        # 1. Parse Inputs
        mat1_list = _parse_matrix_string(match.group("mat1"))
        mat2_list = _parse_matrix_string(match.group("mat2"))
        
        if mat1_list is None or mat2_list is None:
            return f"{ERROR_PREFIX} I couldn't parse the matrix structure. Use format [[1,2],[3,4]]."

        # 2. Determine Operator
        raw_op = (match.group("op") or "matmul").strip().lower()
        op_type = "matmul" # Default
        
        if raw_op in ['+', 'plus', 'add']: op_type = "add"
        elif raw_op in ['-', 'minus', 'subtract']: op_type = "sub"
        elif raw_op in ['*', 'x', 'times', 'multiplied by']: op_type = "elementwise_mul"
        elif raw_op in ['@', 'dot', 'matmul']: op_type = "matmul"

        # 3. Tensor Creation
        # We start on CPU. Moving to GPU has overhead.
        t1 = torch.tensor(mat1_list, dtype=torch.float32)
        t2 = torch.tensor(mat2_list, dtype=torch.float32)

        # 4. Acceleration Logic (Thresholding)
        # Only move to GPU if matrices are large enough (e.g., > 10,000 elements)
        # otherwise PCI-e transfer time > calculation time.
        use_device = "cpu"
        if _DEVICE != "cpu" and (t1.numel() > 10000 or t2.numel() > 10000):
            use_device = _DEVICE
            t1 = t1.to(use_device)
            t2 = t2.to(use_device)

        # 5. Execute Operation
        start_t = time.perf_counter()
        
        res = None
        if op_type == "add":
            res = t1 + t2
        elif op_type == "sub":
            res = t1 - t2
        elif op_type == "elementwise_mul":
            res = t1 * t2
        elif op_type == "matmul":
            res = torch.matmul(t1, t2)
            
        # Synchronize if GPU to get accurate timing (for internal logging)
        if use_device != "cpu":
            torch.cuda.synchronize() if use_device == "cuda" else None
            
        end_t = time.perf_counter()
        
        # Move back to CPU for formatting
        if use_device != "cpu":
            res = res.cpu()

        # 6. Format Output
        output_str = _format_tensor(res)
        
        # Optional: Add compute info footer
        footer = f"\n(Computed on {use_device.upper()} in {(end_t - start_t)*1000:.4f}ms)"
        
        return f"{SUCCESS_PREFIX}{output_str}{footer}"

    except RuntimeError as e:
        # PyTorch throws meaningful errors (e.g. dimension mismatch)
        return f"{ERROR_PREFIX} {str(e)}"
    except Exception as e:
        return f"{ERROR_PREFIX} Unexpected error: {e}"

# --- SELF TEST ---
if __name__ == "__main__":
    print("--- Matrix Hook Self-Test ---")
    
    # 2x2 Matrix Multiplication
    inp1 = "Calculate [[1, 2], [3, 4]] times [[2, 0], [1, 2]]" 
    # Logic: 1*2+2*1=4, 1*0+2*2=4, 3*2+4*1=10, 3*0+4*2=8 -> [[4,4],[10,8]]
    # Note: Regex matches "times" -> elementwise mul in code logic above, 
    # let's test strict MatMul syntax usually implies 'dot' or implicit in linear algebra context
    # but strictly user might mean elementwise.
    
    # Test MatMul specific
    inp2 = "MatMul [[1, 2], [3, 4]] @ [[2, 0], [1, 2]]"
    
    m = PATTERN.match(inp2)
    if m:
        print(handler(m, inp2, "test"))
    else:
        print("No Match")