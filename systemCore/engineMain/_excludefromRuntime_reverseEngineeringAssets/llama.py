from llama_cpp import Llama
import os
import sys

LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", "./pretrainedggufmodel/preTrainedModelBaseVLM.gguf") # Use the right path

try:
    llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=512, verbose=False) # Reduced context
    output = llm("What is the capital of France?", max_tokens=32)
    print(output)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)