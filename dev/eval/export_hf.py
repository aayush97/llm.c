"""
Script to convert modified GPT-2 models from .bin format to Hugging Face Llama format.

This script is designed for the custom model architecture in the modified train_gpt2.py,
which includes RMSNorm, RoPE, and a SwiGLU MLP. It maps the weights to a
LlamaForCausalLM model from the Hugging Face transformers library.

It can optionally upload the model to your account on Hugging Face if you have the CLI installed and are logged in:
  pip install -U "huggingface_hub[cli]"
  huggingface-cli login

Example usage:
  python export_hf_llama.py --input gpt2_d12_bf16.bin --output my-llama-style-model
"""

import numpy as np
import torch
import argparse
from transformers import LlamaConfig, LlamaForCausalLM, GPT2Tokenizer

# -----------------------------------------------------------------------------
# Tensor conversion functions for bfloat16 and float32

def tensor_bf16(data_int16, transpose=False):
    """Converts a numpy array of int16 (representing bfloat16) to a float32 tensor."""
    if transpose:
        data_int16 = data_int16.transpose(1, 0)
    return torch.from_numpy(data_int16).view(torch.bfloat16).to(torch.float32)

def tensor_fp32(data_float32, transpose=False):
    """Converts a numpy array of float32 to a float32 tensor."""
    if transpose:
        data_float32 = data_float32.transpose(1, 0)
    return torch.from_numpy(data_float32).view(torch.float32)

# -----------------------------------------------------------------------------
# Main conversion function

def convert(filepath, output, push_to_hub=False, out_dtype="bfloat16"):
    """
    Converts the .bin model file to a Hugging Face compatible format.
    """
    print(f"Converting model {filepath} to {output} in {out_dtype} format...")
    print(f"Push to Hugging Face Hub: {push_to_hub}")

    try:
        f = open(filepath, 'rb')
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return

    # Read the header to get model configuration
    model_header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if model_header[0] != 20240326:
        print("ERROR: Magic number mismatch in the .bin file. Is this a valid model file?")
        f.close()
        return
        
    version = model_header[1]
    if version not in [3, 5]:
        print(f"ERROR: Unsupported model version: {version}. Only versions 3 (fp32) and 5 (bf16) are supported.")
        f.close()
        return

    # Extract model dimensions from the header
    maxT = model_header[2].item()  # max sequence length (block_size)
    V = model_header[3].item()     # vocab size
    L = model_header[4].item()     # num layers
    H = model_header[5].item()     # num heads
    C = model_header[6].item()     # channels (n_embd)
    Vp = model_header[7].item()    # padded vocab size

    print(f"Model properties: version={version}, max_seq_len={maxT}, vocab_size={V}, padded_vocab_size={Vp}, layers={L}, heads={H}, channels={C}")

    # The MLP intermediate size in the modified train_gpt2.py is non-standard
    intermediate_size = (C // 3) * 8

    # Determine data type for reading weights
    dtype = np.float32 if version == 3 else np.int16
    tensor_converter = tensor_fp32 if version == 3 else tensor_bf16
    
    # --- 1. Load weights from the .bin file ---
    # The loading process must match the exact order they were written in train_gpt2.py
    print("Loading weights from .bin file...")
    w = {}
    
    def read_tensor(shape):
        num_elements = np.prod(shape)
        data = np.frombuffer(f.read(num_elements * np.dtype(dtype).itemsize), dtype=dtype)
        return data.reshape(shape)

    w['wte'] = read_tensor((Vp, C))
    w['ln1w'] = read_tensor((L, C))
    w['qkvw'] = read_tensor((L, 3 * C, C))
    w['qkvb'] = read_tensor((L, 3 * C)) # Will be discarded as Llama has no bias in attention
    w['attprojw'] = read_tensor((L, C, C))
    w['attprojb'] = read_tensor((L, C)) # Will be discarded
    w['ln2w'] = read_tensor((L, C))
    
    # MLP weights are interleaved in the file (gate, up, down for layer 0, then layer 1, etc.)
    w['gate_projw'], w['up_projw'], w['down_projw'] = [], [], []
    for i in range(L):
        w['gate_projw'].append(read_tensor((intermediate_size, C)))
        w['up_projw'].append(read_tensor((intermediate_size, C)))
        w['down_projw'].append(read_tensor((C, intermediate_size)))
    
    w['lnfw'] = read_tensor((C,))
    w['lnfb'] = read_tensor((C,)) # Will be discarded (final norm is RMSNorm)

    # Ensure the file is fully read and then close
    if f.read() != b'':
        print("WARNING: Extra data found at the end of the file.")
    f.close()
    print("Finished loading weights.")
    
    # --- 2. Create a Hugging Face Llama model and map the weights ---
    print("Creating Hugging Face Llama model and mapping weights...")
    model_dict = {}
    
    # Unpad the vocabulary weights
    wte_unpadded = tensor_converter(w['wte'])[:V, :]
    model_dict['model.embed_tokens.weight'] = wte_unpadded
    model_dict['lm_head.weight'] = wte_unpadded  # Tie weights

    for i in range(L):
        # Attention weights
        # FIX: Removed `transpose=True`. The tensor shape is (3*C, C), which correctly splits into three (C, C) tensors along dim=0.
        q, k, v = torch.split(tensor_converter(w['qkvw'][i]), C, dim=0)
        model_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = q
        model_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = k
        model_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = v
        # FIX: Removed `transpose=True`. o_proj weight is already in the correct (C, C) format.
        model_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = tensor_converter(w['attprojw'][i])
        
        # MLP weights
        # FIX: Removed `transpose=True`. Weights are already in the correct (out, in) format.
        model_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = tensor_converter(w['gate_projw'][i])
        model_dict[f'model.layers.{i}.mlp.up_proj.weight'] = tensor_converter(w['up_projw'][i])
        model_dict[f'model.layers.{i}.mlp.down_proj.weight'] = tensor_converter(w['down_projw'][i])

        # RMSNorm weights
        model_dict[f'model.layers.{i}.input_layernorm.weight'] = tensor_converter(w['ln1w'][i])
        model_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = tensor_converter(w['ln2w'][i])
        
    # Final normalization layer
    model_dict['model.norm.weight'] = tensor_converter(w['lnfw'])
    
    print("NOTE: Discarding attention biases and final layernorm bias as they are not used in the Llama architecture.")

    # --- 3. Configure, create, and save the final model ---
    config = LlamaConfig(
        vocab_size=V,
        max_position_embeddings=maxT,
        hidden_size=C,
        intermediate_size=intermediate_size,
        num_hidden_layers=L,
        num_attention_heads=H,
        hidden_act="silu", # Corresponds to SiLU/Swish activation in SwiGLU
        rms_norm_eps=1e-6, # From train_gpt2.py
        rope_theta=10000.0, # From train_gpt2.py
        tie_word_embeddings=True,
    )
    
    print("Initializing LlamaForCausalLM model...")
    model = LlamaForCausalLM(config)
    if out_dtype == "bfloat16":
        model = model.to(torch.bfloat16)

    # Load the state dictionary and handle any mismatches
    load_result = model.load_state_dict(model_dict, strict=False)
    if load_result.missing_keys:
        print("WARNING: Missing keys during model load:", load_result.missing_keys)
    if load_result.unexpected_keys:
        print("WARNING: Unexpected keys during model load:", load_result.unexpected_keys)

    print(f"Saving model to directory: {output}")
    model.save_pretrained(output, max_shard_size="5GB", safe_serialization=True)

    # Save the tokenizer (using the standard gpt2 tokenizer as per the training script)
    print("Saving tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output)

    print("Conversion complete!")

    if push_to_hub:
        print(f"Uploading {output} to Hugging Face Hub...")
        try:
            model.push_to_hub(output)
            tokenizer.push_to_hub(output)
            print("Upload successful.")
        except Exception as e:
            print(f"An error occurred during upload: {e}")

def spin(output):
    """A quick test function to generate text from the exported model."""
    print("\n" + "-"*80)
    print("Taking the exported model for a spin...")
    print('-'*80)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(output)
        model = AutoModelForCausalLM.from_pretrained(
            output, 
            torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32, 
            device_map=device
        )
        model.eval()

        prompt = "In a world where magic is real,"
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_tokens = model.generate(tokens, max_new_tokens=64, repetition_penalty=1.2, top_k=40, do_sample=True)
        
        sample = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print("\n--- GENERATED SAMPLE ---")
        print(sample)
        print("------------------------")
    except ImportError:
        print("Could not import transformers. Skipping model test.")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert custom llm.c .bin models to HF Llama format.")
    parser.add_argument("--input", "-i", help="The name of the input .bin model file", type=str, required=True)
    parser.add_argument("--output", "-o", help="The Hugging Face output model directory", type=str, required=True)
    parser.add_argument("--dtype", "-d", help="Output dtype: float32 or bfloat16 (default)", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--push", "-p", help="Push the model to your Hugging Face account", action="store_true")
    parser.add_argument("--no-spin", help="Do not run a test generation at the end", action="store_true")
    args = parser.parse_args()
    
    convert(args.input, args.output, args.push, args.dtype)
    if not args.no_spin:
        spin(args.output)

