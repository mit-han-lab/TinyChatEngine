"""Implementation of exporting StarCoder PyTorch model to TinyChatEngine format.

Usage:
   python starcoder_exporter.py <path of hugging face model checkpoint> <output dir>

Example commandline:
   python starcoder_exporter.py ~/starcoder models/StarCoder
"""
import argparse
import math
import os
import struct
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoConfig

n_head = 48
embed_dim = 6144
head_dim = embed_dim // n_head

@torch.no_grad()
def _export_model(model, prefix):

    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "lm_head.bin"), "wb") as f:
        f.write(model.lm_head._parameters["weight"].cpu().float().numpy().tobytes())
    _export_starcoder_model(model.transformer, os.path.join(f"{outpath}", "decoder"))


def _export_wte_wpe(embed_tokens, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(embed_tokens.weight.cpu().float().numpy().tobytes())


def _export_starcoder_model(model, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)

    _export_wte_wpe(model.wte, os.path.join(outpath, "wte"))
    _export_wte_wpe(model.wpe, os.path.join(outpath, "wpe"))
    _export_LayerNorm(model.ln_f, os.path.join(f"{outpath}", "ln_f"))
    for idx, layer in enumerate(model.h):
        _export_starcoder_layer(layer, os.path.join(outpath, f"layer{idx}"))

def _export_LayerNorm(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op.weight.cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "bias.bin"), "wb") as f:
        f.write(op.bias.cpu().float().numpy().tobytes())

def _export_starcoder_layer(layer, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    _export_attention_params(layer.attn, os.path.join(outpath, "attn"))
    _export_LayerNorm(layer.ln_1, os.path.join(f"{outpath}", "ln_1"))
    _export_LinearFP(layer.mlp.c_fc, os.path.join(outpath, "c_fc"))
    _export_LinearFP(layer.mlp.c_proj, os.path.join(outpath, "c_proj"))
    _export_LayerNorm(layer.ln_2, os.path.join(f"{outpath}", "ln_2"))

def _export_LinearFP(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op._parameters["weight"].cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "bias.bin"), "wb") as f:
        f.write(op._parameters["bias"].cpu().float().numpy().tobytes())

def _export_LinearFP_MQAtoMHA(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)

    # Weight
    weight_data = op._parameters["weight"].cpu().float().squeeze().numpy()
    # ((n_heads + 2) * head_dim, hidden_dim) -> (3 * n_heads * head_dim, hidden_dim)
    q, k, v = np.split(weight_data, (n_head * head_dim, (n_head + 1) * head_dim), axis=0)
    # duplicate k, v along the first axis (head_dim, hidden_dim) -> (n_heads * head_dim, hidden_dim)
    if len(k.shape) == 2:
        k = np.tile(k, (n_head, 1))
        v = np.tile(v, (n_head, 1))
    elif len(k.shape) == 1:
        k = np.tile(k, (n_head))
        v = np.tile(v, (n_head))
    # concat q, k, v along the first axis (n_heads * head_dim, hidden_dim) -> (3 * n_heads * head_dim, hidden_dim)
    weight_data = np.concatenate((q, k, v), axis=0)

    # Bias
    bias_data = op._parameters["bias"].cpu().float().squeeze().numpy()
    # ((n_heads + 2) * head_dim, hidden_dim) -> (3 * n_heads * head_dim, hidden_dim)
    q, k, v = np.split(bias_data, (n_head * head_dim, (n_head + 1) * head_dim), axis=0)
    # duplicate k, v along the first axis (head_dim, hidden_dim) -> (n_heads * head_dim, hidden_dim)
    if len(k.shape) == 2:
        k = np.tile(k, (n_head, 1))
        v = np.tile(v, (n_head, 1))
    elif len(k.shape) == 1:
        k = np.tile(k, (n_head))
        v = np.tile(v, (n_head))
    # concat q, k, v along the first axis (n_heads * head_dim, hidden_dim) -> (3 * n_heads * head_dim, hidden_dim)
    bias_data = np.concatenate((q, k, v), axis=0)

    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(weight_data.tobytes())
    with open(os.path.join(f"{outpath}", "bias.bin"), "wb") as f:
        f.write(bias_data.tobytes())

def _export_BMM_F32T(alpha, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "alpha.bin"), "wb") as f:
        f.write(struct.pack("f", alpha))


def _export_attention_params(attn, prefix: str):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    # with open(os.path.join(outpath, "scaling.bin"), "wb") as f:
    #     f.write(bytearray(struct.pack("f", attn.scaling)))
    _export_LinearFP_MQAtoMHA(attn.c_attn, os.path.join(outpath, "c_attn"))
    _export_LinearFP(attn.c_proj, os.path.join(outpath, "c_proj"))
    qk_bmm_alpha = 1 / math.sqrt(head_dim)
    _export_BMM_F32T(qk_bmm_alpha, os.path.join(outpath, "qk_bmm"))


def main():
    """Export a StarCoder model to TinyChatEngine format."""
    parser = argparse.ArgumentParser(description="export StarCoder pytorch model to TinyChatEngine format.")
    parser.add_argument("--hf_path", type=str, help="Path to huggingface model hub", default=None)
    parser.add_argument("--model", type=str, help="Path of the StarCoder torch model")
    parser.add_argument("--output", type=str, help="Output directory of the exported model")

    args = parser.parse_args()

    if args.hf_path is None:
        if not os.path.exists(args.model):
            print(f"The model path '{args.model}' does not exist.")
            return

        if not os.path.exists(args.output):
            print(f"The model path '{args.output}' does not exist.")
            return

        print("Loading model...")
        if args.model.endswith(".pt"):
            if args.model.split("/")[-1].lower().startswith("starcoder"):
                print("Loading StarCoder model...");
                config = AutoConfig.from_pretrained("models/starcoder", trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained("models/starcoder", config=config, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)
            else:
                print("Model not supported.")
                return
            
            model.load_state_dict(torch.load(args.model))
        else:
            print("Loading StarCoder model...");
            config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(args.model, config=config, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.hf_path, torch_dtype=torch.float32)

    print("Start exporting the model...")
    _export_model(model, args.output)


if __name__ == "__main__":
    main()
