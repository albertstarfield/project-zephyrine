# test.py
try:
    import llama_cpp
    print("llama_cpp imported successfully!")
except ImportError as e:
    print(f"Failed to import llama_cpp: {e}")