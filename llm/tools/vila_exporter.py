"""Implementation of exporting VILA PyTorch model to TinyChatEngine format.

Usage:
   python vila_exporter.py <path of hugging face model checkpoint> <output dir>

Example commandline:
   python tools/vila_exporter.py --model models/vila-7b --output models/VILA_7B
"""

import argparse
import math
import os
import struct
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

# from llava.conversation import SeparatorStyle, conv_templates
# from llava.eval.utils import preprocess_image
# from llava.model import *
# from llava.model.utils import KeywordsStoppingCriteria
# from llava.model.visual_attn_scale import new_attention_forward
# from llava.utils import disable_torch_init
import sys
sys.path.append('../../../LLaVA')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava import LlavaLlamaForCausalLM

@torch.no_grad()
def _export_model(model, prefix):

    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "lm_head.bin"), "wb") as f:
        f.write(model.lm_head._parameters["weight"].cpu().float().numpy().tobytes())
    _export_llama_model(model.model, os.path.join(f"{outpath}", "decoder"))

    # Export to Clip's folder "models/CLIP_ViT_Large"
    # _export_mm_projector(model.model.mm_projector, f"models/CLIP_ViT_Large/mm_projector")
    for idx, mm_projector in enumerate(model.model.mm_projector):
        if idx == 0 or idx == 2:
            # _export_mm_projector(mm_projector, os.path.join(outpath, f"mm_projector_{idx}"))
            # Export to Clip's folder "models/CLIP_ViT_Large"
            _export_mm_projector(mm_projector, f"models/CLIP_ViT_Large/mm_projector_{idx}")


def _export_mm_projector(mm_projector, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(mm_projector.weight.cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "bias.bin"), "wb") as f:
        f.write(mm_projector.bias.cpu().float().numpy().tobytes())


def _export_embed_tokens(embed_tokens, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(embed_tokens.weight.cpu().float().numpy().tobytes())


def _export_llama_model(model, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)

    _export_embed_tokens(model.embed_tokens, os.path.join(outpath, "embed_tokens"))
    _export_LlamaRMSNorm(model.norm, os.path.join(outpath, "norm"))
    for idx, layer in enumerate(model.layers):
        _export_llama_layer(layer, os.path.join(outpath, f"layer{idx}"))


def _export_LlamaRMSNorm(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op.weight.cpu().float().numpy().tobytes())


def _export_llama_layer(layer, prefix):
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
    _export_linearfp(attn.k_proj, os.path.join(outpath, "k_proj"))
    _export_linearfp(attn.v_proj, os.path.join(outpath, "v_proj"))
    _export_linearfp(attn.q_proj, os.path.join(outpath, "q_proj"))
    _export_linearfp(attn.o_proj, os.path.join(outpath, "o_proj"))
    qk_bmm_alpha = 1 / math.sqrt(attn.head_dim)
    _export_BMM_F32T(qk_bmm_alpha, os.path.join(outpath, "qk_bmm"))
    _export_rotaryEmbedding(attn.rotary_emb, os.path.join(outpath, "rotary_emb"))


def main():
    """Export a VILA model to TinyChatEngine format."""
    parser = argparse.ArgumentParser(description="export VILA pytorch model to TinyChatEngine format.")
    parser.add_argument("--hf_path", type=str, help="Path to huggingface model hub", default=None)
    parser.add_argument("--model", type=str, help="Path of the VILA torch model")
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
            if args.model.split("/")[-1].lower().startswith("vila"):
                if args.model.split("-")[2].lower() == "7b":
                    print("Loading VILA 7B model...")
                    # config = AutoConfig.from_pretrained("Efficient-Large-Model/vila-7b", trust_remote_code=True)
                    # processor = AutoProcessor.from_pretrained("Efficient-Large-Model/vila-7b")
                    # model = AutoModelForCausalLM.from_pretrained("Efficient-Large-Model/vila-7b", config=config, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)
                    # config = AutoConfig.from_pretrained("/home/wweichen/workspace/models/LLM/vila_llava-7b", trust_remote_code=True)
                    # processor = AutoProcessor.from_pretrained("/home/wweichen/workspace/models/LLM/vila-7b")
                    model = LlavaLlamaForCausalLM.from_pretrained("/home/wweichen/workspace/models/LLM/vila_llava-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)
                elif args.model.split("-")[1].lower() == "13b":
                    print("Loading VILA 13B model...")
                    config = AutoConfig.from_pretrained("/home/wweichen/workspace/models/LLM/vila-13b", trust_remote_code=True)
                    # processor = AutoProcessor.from_pretrained("/home/wweichen/workspace/models/LLM/vila-13b")
                    model = LlavaLlamaForCausalLM.from_pretrained("/home/wweichen/workspace/models/LLM/vila-13b", config=config, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)
                elif args.model.split("-")[2].lower() == "2.7b":
                    print("Loading VILA 2.7B model...")
                    model = LlavaLlamaForCausalLM.from_pretrained("/home/wweichen/workspace/models/LLM/vila_llava-2.7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)
                else:
                    print("Model size not supported.")
                    return
            else:
                print("Model type not supported.")
                return
            
            model.load_state_dict(torch.load(args.model), strict=False)
        else:
            config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            # processor = AutoProcessor.from_pretrained(args.model)
            # model = AutoModelForCausalLM.from_pretrained(args.model, config=config, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)
            model = LlavaLlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)
    else:
        # processor = AutoProcessor.from_pretrained(args.hf_path)
        model = AutoModelForCausalLM.from_pretrained(args.hf_path, torch_dtype=torch.float16)

    print("Start exporting VILA model...")
    _export_model(model, args.output)
    print("Finished exporting VILA model.")


if __name__ == "__main__":
    main()
