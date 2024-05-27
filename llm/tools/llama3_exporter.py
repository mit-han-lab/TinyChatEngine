"""Implementation of exporting LLaMA-3 PyTorch model to TinyChatEngine format.

Usage:
   python llama3_exporter.py <path of hugging face model checkpoint> <output dir>

Example commandline:
   python tools/llama3_exporter.py --model models/Meta-Llama-3-8B-Instruct --output models/LLaMA_3_8B_Instruct
"""
import argparse
import math
import os
import struct

import torch
from transformers import LlamaForCausalLM
import numpy as np

n_head = 32
n_kv_head = 8
n_kv_groups = n_head // n_kv_head
embed_dim = 4096
head_dim = embed_dim // n_head

@torch.no_grad()
def _export_model(model, prefix):

    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "lm_head.bin"), "wb") as f:
        f.write(model.lm_head._parameters["weight"].cpu().float().numpy().tobytes())
    _export_llama3_model(model.model, os.path.join(f"{outpath}", "decoder"))


def _export_embed_tokens(embed_tokens, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(embed_tokens.weight.cpu().float().numpy().tobytes())


def _export_llama3_model(model, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)

    _export_embed_tokens(model.embed_tokens, os.path.join(outpath, "embed_tokens"))
    _export_LlamaRMSNorm(model.norm, os.path.join(outpath, "norm"))
    for idx, layer in enumerate(model.layers):
        _export_llama3_layer(layer, os.path.join(outpath, f"layer{idx}"))


def _export_LlamaRMSNorm(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op.weight.cpu().float().numpy().tobytes())


def _export_llama3_layer(layer, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    _export_attention_params(layer.self_attn, os.path.join(outpath, "self_attn"))
    _export_LlamaRMSNorm(layer.input_layernorm, os.path.join(outpath, "input_layernorm"))
    _export_LlamaRMSNorm(
        layer.post_attention_layernorm,
        os.path.join(outpath, "post_attention_layernorm"),
    )
    _export_linearfp(layer.mlp.gate_proj, os.path.join(outpath, "gate_proj"))
    _export_linearfp(layer.mlp.down_proj, os.path.join(outpath, "down_proj"))
    _export_linearfp(layer.mlp.up_proj, os.path.join(outpath, "up_proj"))


def _export_linearfp(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op._parameters["weight"].cpu().float().numpy().tobytes())

def _export_Linearfp_GQAtoMHA(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)

    # Load weight
    weight_data = op._parameters["weight"].cpu().float().squeeze().numpy()
    # Reshape weight
    # Original size is (n_kv_head, head_dim)
    # Reshape to (n_kv_head, head_dim * n_kv_groups)
    weight_data = weight_data.reshape((embed_dim, embed_dim // n_kv_groups))
    # weight_data = weight_data.reshape((embed_dim // n_kv_groups, embed_dim))
    # # Duplicate weight along the first axis (head_dim, hidden_dim) -> (n_heads * head_dim, hidden_dim)
    # if len(weight_data.shape) == 2:
    #     repeat_weight_data = np.tile(weight_data, (n_kv_groups, 1))
    # elif len(weight_data.shape) == 1:
    #     repeat_weight_data = np.tile(weight_data, (n_kv_groups))
    repeat_weight_data = np.tile(weight_data, (1, n_kv_groups))
    # repeat_weight_data = np.repeat(weight_data, n_kv_groups, axis=1)
    # repeat_weight_data = np.tile(weight_data, (n_kv_groups, 1))

    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(repeat_weight_data.tobytes())

def _export_rotaryEmbedding(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "cos_cached.bin"), "wb") as f:
        f.write(op.cos_cached.cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "sin_cached.bin"), "wb") as f:
        f.write(op.sin_cached.cpu().float().numpy().tobytes())


def _export_BMM_F32T(alpha, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "alpha.bin"), "wb") as f:
        f.write(struct.pack("f", alpha))


def _export_attention_params(attn, prefix: str):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    _export_linearfp(attn.q_proj, os.path.join(outpath, "q_proj"))
    # # GQA to MHA
    # _export_Linearfp_GQAtoMHA(attn.k_proj, os.path.join(outpath, "k_proj"))
    # _export_Linearfp_GQAtoMHA(attn.v_proj, os.path.join(outpath, "v_proj"))
    # Original
    _export_linearfp(attn.k_proj, os.path.join(outpath, "k_proj"))
    _export_linearfp(attn.v_proj, os.path.join(outpath, "v_proj"))
    _export_linearfp(attn.o_proj, os.path.join(outpath, "o_proj"))
    qk_bmm_alpha = 1 / math.sqrt(attn.head_dim)
    _export_BMM_F32T(qk_bmm_alpha, os.path.join(outpath, "qk_bmm"))
    _export_rotaryEmbedding(attn.rotary_emb, os.path.join(outpath, "rotary_emb"))


def main():
    """Export a LLaMA-3 model to TinyChatEngine format."""
    parser = argparse.ArgumentParser(description="export LLaMA-3 pytorch model to TinyChatEngine format.")
    parser.add_argument("--hf_path", type=str, help="Path to huggingface model hub", default=None)
    parser.add_argument("--model", type=str, help="Path of the LLaMA-3 torch model")
    parser.add_argument("--output", type=str, help="Output directory of the exported model")

    args = parser.parse_args()

    if args.hf_path is None:
        if not os.path.exists(args.model):
            print(f"The model path '{args.model}' does not exist.")
            return

        if not os.path.exists(args.output):
            print(f"The output path '{args.output}' does not exist. Creating a new directory...")
            os.makedirs(args.output, exist_ok=True)

        print("Loading model...")
        if args.model.endswith(".pt"):
            if args.model.split("/")[-1].lower().startswith("llama3"):
                if args.model.split("-")[2].lower() == "8b":
                    print("Loading LLaMA-3 8B model...")
                    model = LlamaForCausalLM.from_pretrained("/home/wweichen/workspace/models/llm/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)
            else:
                print("Model not supported.")
                return
            
            model.load_state_dict(torch.load(args.model))
        else:
            model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=True, offload_state_dict=True)
    else:
        model = LlamaForCausalLM.from_pretrained(args.hf_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)

    print("Start exporting LLaMA-3 model...")
    _export_model(model, args.output)
    print("Finished exporting LLaMA-3 model.")


if __name__ == "__main__":
    main()
