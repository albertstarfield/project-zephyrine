import sys
import struct

# --- GGUF Type Constants ---
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

def get_fixed_type_size(type_id):
    """Returns byte size for fixed-width types. Returns 0 for variable types (String/Array)."""
    if type_id in [GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL]: return 1
    if type_id in [GGUF_TYPE_UINT16, GGUF_TYPE_INT16]: return 2
    if type_id in [GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32]: return 4
    if type_id in [GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64]: return 8
    return 0

def read_string(f):
    """Reads a standard GGUF string: 8-byte length + utf8 bytes."""
    length_bytes = f.read(8)
    if not length_bytes: return ""
    length = struct.unpack('<Q', length_bytes)[0]
    return f.read(length).decode('utf-8', errors='replace')

def skip_array(f):
    """Reads array header and safely advances file pointer past the content."""
    # 1. Read Item Type (4 bytes) and Array Length (8 bytes)
    item_type = struct.unpack('<I', f.read(4))[0]
    array_len = struct.unpack('<Q', f.read(8))[0]
    
    # 2. Check if it's a fixed-width type (fast skip)
    fixed_size = get_fixed_type_size(item_type)
    
    if fixed_size > 0:
        # Math: Just jump ahead
        total_bytes = fixed_size * array_len
        f.seek(total_bytes, 1) # 1 = seek relative to current position
        return f"<Array[{array_len}] of scalar type {item_type} - Skipped>"
        
    elif item_type == GGUF_TYPE_STRING:
        # Slow skip: Must read length of every string to pass it
        # This is where the previous script failed.
        for _ in range(array_len):
            s_len = struct.unpack('<Q', f.read(8))[0]
            f.seek(s_len, 1)
        return f"<Array[{array_len}] of Strings - Skipped>"
        
    elif item_type == GGUF_TYPE_ARRAY:
        # Nested array (rare). Recurse.
        for _ in range(array_len):
            skip_array(f)
        return "<Nested Array - Skipped>"
        
    return "<Unknown Array - Skipped>"

def read_gguf_header_robust(file_path):
    with open(file_path, 'rb') as f:
        # Header
        magic = f.read(4)
        if magic != b'GGUF':
            print(f"❌ Not GGUF: {magic}")
            return
        
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]

        print(f"📦 GGUF v{version} | KV: {kv_count} | Tensors: {tensor_count}")
        print("-" * 40)

        # Target keys for memory calc
        target_keys = [
            "general.architecture", 
            "general.quantization_version", 
            "general.file_type",
            "general.parameter_count",
            "general.context_length",
            "gemma3.context_length",
            "qwen2.context_length",
            "llama.context_length"
        ]

        # Iterate KV pairs
        for i in range(kv_count):
            # Read Key
            key = read_string(f)
            
            # Read Type
            val_type = struct.unpack('<I', f.read(4))[0]
            
            # Read Value
            value = None
            if val_type == GGUF_TYPE_STRING:
                value = read_string(f)
            elif val_type == GGUF_TYPE_ARRAY:
                # The Fix: Actually skip the bytes
                value = skip_array(f)
            else:
                # Read Scalar
                size = get_fixed_type_size(val_type)
                data = f.read(size)
                if val_type == GGUF_TYPE_UINT32: value = struct.unpack('<I', data)[0]
                elif val_type == GGUF_TYPE_INT32: value = struct.unpack('<i', data)[0]
                elif val_type == GGUF_TYPE_FLOAT32: value = struct.unpack('<f', data)[0]
                elif val_type == GGUF_TYPE_UINT64: value = struct.unpack('<Q', data)[0]
                elif val_type == GGUF_TYPE_INT64: value = struct.unpack('<q', data)[0]
                elif val_type == GGUF_TYPE_FLOAT64: value = struct.unpack('<d', data)[0]
                elif val_type == GGUF_TYPE_BOOL: value = struct.unpack('?', data)[0]
                else: value = f"Raw: {data.hex()}"

            # Print if it's what we want
            if key in target_keys or "context_length" in key:
                print(f"  🔹 {key}: {value}")
                
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 probe_gguf_v2.py <model.gguf>")
    else:
        try:
            read_gguf_header_robust(sys.argv[1])
        except Exception as e:
            print(f"❌ Fatal Error: {e}")
            import traceback
            traceback.print_exc()
