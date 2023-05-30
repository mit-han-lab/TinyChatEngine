import math
import os
import struct

import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer


def export_LlamaRMSNorm(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op.weight.cpu().float().numpy().tobytes())


def export_llama_layer(layer, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    export_attention_params(layer.self_attn, os.path.join(outpath, "self_attn"))
    export_LlamaRMSNorm(layer.input_layernorm, os.path.join(outpath, "input_layernorm"))
    export_LlamaRMSNorm(
        layer.post_attention_layernorm,
        os.path.join(outpath, "post_attention_layernorm"),
    )
    export_linearfp(layer.mlp.gate_proj, os.path.join(outpath, "gate_proj"))
    export_linearfp(layer.mlp.down_proj, os.path.join(outpath, "down_proj"))
    export_linearfp(layer.mlp.up_proj, os.path.join(outpath, "up_proj"))


def export_linearfp(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op._parameters["weight"].cpu().float().numpy().tobytes())


def export_rotaryEmbedding(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "cos_cached.bin"), "wb") as f:
        f.write(op.cos_cached.cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "sin_cached.bin"), "wb") as f:
        f.write(op.sin_cached.cpu().float().numpy().tobytes())


def export_BMM_F32T(alpha, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "alpha.bin"), "wb") as f:
        f.write(struct.pack("f", alpha))


def export_attention_params(attn, prefix: str):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    export_linearfp(attn.k_proj, os.path.join(outpath, "k_proj"))
    export_linearfp(attn.v_proj, os.path.join(outpath, "v_proj"))
    export_linearfp(attn.q_proj, os.path.join(outpath, "q_proj"))
    export_linearfp(attn.o_proj, os.path.join(outpath, "o_proj"))
    qk_bmm_alpha = 1 / math.sqrt(attn.head_dim)
    export_BMM_F32T(qk_bmm_alpha, os.path.join(outpath, "qk_bmm"))
    export_rotaryEmbedding(attn.rotary_emb, os.path.join(outpath, "rotary_emb"))


@torch.no_grad()
def text_generation(tokenizer, model):
    inputs = tokenizer(
        "Building website in 10 steps:",
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.25,
    )
    print("Generating...")
    # export_attention_params(model.model.layers[0].self_attn, "transformer/assets/llama/tests/atten/first_attn")
    export_llama_layer(model.model.layers[0], "transformer/assets/llama/tests/layer0")
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=64,
    )
    for s in generation_output.sequences:
        print(tokenizer.decode(s))


tokenizer = LlamaTokenizer.from_pretrained("reference_py/llama7B")
model = LlamaForCausalLM.from_pretrained("reference_py/llama7B")

text_generation(tokenizer=tokenizer, model=model)
