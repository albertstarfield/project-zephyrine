import torch
import safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TextStreamer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import logging
import platform
import sys
import os
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_device():
    """Detects MPS or CPU, prioritizing MPS."""
    return "mps" if torch.backends.mps.is_available() else "cpu"

def dequantize_gptq_weights(state_dict):
    """Dequantizes GPTQ weights, handling act-order and non-act-order."""
    dequantized_state_dict = {}
    for name, param in state_dict.items():
        logging.debug(f"Processing layer: {name}")
        if "qweight" in name:
            qweight = param
            qzeros = state_dict[name.replace("qweight", "qzeros")]
            scales = state_dict[name.replace("qweight", "scales")]
            g_idx = state_dict.get(name.replace("qweight", "g_idx"))  # Use .get()

            logging.debug(f"  qweight shape: {qweight.shape}")
            logging.debug(f"  qzeros shape: {qzeros.shape}")
            logging.debug(f"  scales shape: {scales.shape}")
            if g_idx is not None:
                logging.debug(f"  g_idx shape: {g_idx.shape}")

            if g_idx is not None:
                dequantized_weight = dequantize_gptq(qweight, qzeros, scales, g_idx)
            else:
                dequantized_weight = dequantize_no_actorder_gptq(qweight, qzeros, scales)

            dequantized_state_dict[name.replace("qweight", "weight")] = dequantized_weight
        elif "weight" in name:
            dequantized_state_dict[name] = param.to(torch.bfloat16)
        else:
            dequantized_state_dict[name] = param
    return dequantized_state_dict

def dequantize_no_actorder_gptq(qweight, qzeros, scales, bits=4):
    """Dequantizes GPTQ weights without act-order (Attempt 4 - Simplified Bit Unpacking & Reshape)."""
    logging.debug("  Entering dequantize_no_actorder_gptq (Attempt 4)")
    hidden_size = scales.shape[0]
    input_size = scales.shape[1] # ADDED: Capture input size from scales
    wf = torch.tensor(list(range(0, bits)), device=qweight.device, dtype=torch.int32) # Simplified wf
    logging.debug(f"    wf shape: {wf.shape}")

    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0).unsqueeze(0))
    zeros = torch.bitwise_and(zeros, (2 ** bits - 1))
    zeros = zeros + 0.5
    # Correct reshape for zeros to match scales' shape - ATTEMPT 2: SAME SHAPE AS SCALES
    zeros = zeros.reshape(scales.shape[0], scales.shape[1]) # Modified reshape - SAME SHAPE AS SCALES
    logging.debug(f"    zeros shape: {zeros.shape}")

    # Correct unpacking of qweight (Attempt 4 - Simplified Reshape)
    logging.debug(f"    qweight shape (before unpacking reshape): {qweight.shape}")
    qweight = qweight.reshape(-1, qweight.shape[-1] // (32 // bits))  # Reshape for unpacking
    logging.debug(f"    qweight shape (after unpacking reshape): {qweight.shape}")

    weight = torch.zeros(qweight.shape[0] * 8, qweight.shape[1], dtype=torch.int32, device=qweight.device) # Initialize weight
    for i in range(8): # Manual unpacking loop
        shifted_qweight = torch.bitwise_right_shift(qweight, i * 4)
        bit_mask = torch.bitwise_and(shifted_qweight, 0xF) # 0xF = 1111 in binary
        weight[i::8, :] = bit_mask # Assign unpacked bits

    logging.debug(f"    weight shape (after manual bitwise ops): {weight.shape}")

    weight = weight[:scales.shape[1], :] # Trim weight to scales.shape[1] (input_size)
    weight = weight.T # Transpose to [out_features, in_features] or [scales.shape[0], scales.shape[1]]
    logging.debug(f"   weight shape (after final reshape/transpose): {weight.shape}")
    logging.debug(f"    scales shape: {scales.shape}")

    # Dequantize
    logging.debug("    Performing dequantization multiplication...")
    weight = scales * (weight - zeros)
    logging.debug(f"    weight shape (after dequantization): {weight.shape}")
    return weight.to(torch.bfloat16)

def dequantize_gptq(qweight, qzeros, scales, g_idx, bits=4):
    """Dequantizes GPTQ weights with act-order (Attempt 4 - Simplified Bit Unpacking & Reshape)."""
    logging.debug("  Entering dequantize_gptq (Attempt 4)")
    hidden_size = scales.shape[0]
    input_size = scales.shape[1]

    wf = torch.tensor(list(range(0, bits)), device=qweight.device, dtype=torch.int32) # Simplified wf
    logging.debug(f"    wf shape: {wf.shape}")

    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0).unsqueeze(0))
    zeros = torch.bitwise_and(zeros, (2 ** bits - 1))
    zeros = zeros + 0.5
    # Correct reshape for zeros to match scales' shape - ATTEMPT 2: SAME SHAPE AS SCALES
    zeros = zeros.reshape(scales.shape[0], scales.shape[1]) # Modified reshape - SAME SHAPE AS SCALES
    logging.debug(f"    zeros shape: {zeros.shape}")

    # Correct unpacking of qweight (Attempt 4 - Simplified Reshape)
    logging.debug(f"    qweight shape (before unpacking reshape): {qweight.shape}")
    qweight = qweight.reshape(-1, qweight.shape[-1] // (32 // bits)) # Reshape for unpacking
    logging.debug(f"    qweight shape (after unpacking reshape): {qweight.shape}")

    weight = torch.zeros(qweight.shape[0] * 8, qweight.shape[1], dtype=torch.int32, device=qweight.device) # Initialize weight
    for i in range(8): # Manual unpacking loop
        shifted_qweight = torch.bitwise_right_shift(qweight, i * 4)
        bit_mask = torch.bitwise_and(shifted_qweight, 0xF) # 0xF = 1111 in binary
        weight[i::8, :] = bit_mask # Assign unpacked bits

    logging.debug(f"    weight shape (after manual bitwise ops): {weight.shape}")

    weight = weight[:scales.shape[1], :] # Trim weight to scales.shape[1] (input_size)
    weight = weight.T # Transpose to [out_features, in_features] or [scales.shape[0], scales.shape[1]]
    logging.debug(f"    weight shape (after final reshape/transpose): {weight.shape}")
    logging.debug(f"    scales shape: {scales.shape}")


    # Dequantize
    logging.debug("    Performing dequantization multiplication...")
    weight = scales * (weight - zeros)
    logging.debug(f"    weight shape (after dequantization): {weight.shape}")

    # Apply inverse permutation
    invperm = torch.argsort(g_idx)
    weight = weight[:, invperm]
    return weight.to(torch.bfloat16)


def load_model_and_tokenizer(device, model_path):
    """Loads model and tokenizer, dequantizing GPTQ weights."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        safetensors_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors file found in {model_path}")
        gptq_weights_path = os.path.join(model_path, safetensors_files[0])

        with safetensors.safe_open(gptq_weights_path, framework="pt", device="cpu") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}

        dequantized_state_dict = dequantize_gptq_weights(state_dict)
        model_name_full = "google/gemma-2b"
        model = AutoModelForCausalLM.from_pretrained(
            model_name_full, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        model.load_state_dict(dequantized_state_dict, strict=False)
        return model, tokenizer

    except Exception as e:
        logging.exception(f"Error loading model/tokenizer: {e}")
        sys.exit(1)

def chat(model, tokenizer, device):
    """Runs a chatbot interaction loop."""
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ("exit", "quit", "bye"):
                break
            inputs = tokenizer(user_input, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, streamer=streamer, max_new_tokens=512,
                    do_sample=True, top_p=0.95, top_k=60, temperature=0.7
                )
            if not streamer:  # Fallback
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Bot: {output_text}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            logging.error(f"Generation error: {e}")
            print("Bot: Sorry, I had a problem.")

def train(model, tokenizer, device):
    """Fine-tunes the model using LoRA."""
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./lora-output", per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=2e-4, bf16=True, logging_steps=10, num_train_epochs=3,
        optim="adamw_torch", gradient_checkpointing=True
    )

    data = [  # Example dataset - REPLACE WITH YOUR DATA
        {"text": "What is the capital of France?", "answer": "Paris"},
        {"text": "What is the highest mountain?", "answer": "Mount Everest"},
    ]
    def gen():
        for d in data:
            yield tokenizer(
                "Question: " + d["text"] + " Answer: " + d["answer"],
                return_tensors="pt", padding="max_length", max_length=128, truncation=True
            )

    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=list(gen()),
        dataset_text_field="text", tokenizer=tokenizer,
        peft_config=model.peft_config, max_seq_length=128
    )
    trainer.train()

def main():
    """Main function: Downloads, loads, tests, and fine-tunes the model."""
    print("-" * 30)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    device = detect_device()
    print(f"Using device: {device}")
    print("-" * 30)

    model_name = "TechxGenus/gemma-2b-GPTQ"
    print(f"Downloading GPTQ model from Hugging Face Hub: {model_name}...")
    model_path = snapshot_download(repo_id=model_name)
    print(f"GPTQ model downloaded to: {model_path}")

    model, tokenizer = load_model_and_tokenizer(device, model_path)
    print("Running inference test...")
    chat(model, tokenizer, device)
    print("\nStarting fine-tuning...")
    train(model, tokenizer, device)
    print("Fine-tuning complete.")

if __name__ == "__main__":
    main()