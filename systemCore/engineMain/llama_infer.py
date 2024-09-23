# llama_infer.py
import torch  # Assuming your model uses PyTorch, adjust accordingly
from llama_index import LlamaModel  # Replace with actual import if different

# Load your model from the specified path
model_path = "/home/adri/Downloads/llama"
model = LlamaModel.load(model_path)

def infer(text: str) -> str:
    """
    Run inference using the loaded model.
    
    Args:
        text (str): The input text for the model.
        
    Returns:
        str: The generated output from the model.
    """
    result = model.generate(text)  # Replace with the actual model call
    return result
