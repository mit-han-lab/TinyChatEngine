"""Implementation of exporting Clip PyTorch model to TinyChatEngine format.

Usage:
   python clip_exporter.py <path of hugging face model checkpoint> <output dir>

Example commandline:
   python tools/clip_exporter.py --model models/clip-vit-large-patch14-336 --output models/CLIP_ViT_Large
"""
import argparse
import math
import os
import struct

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

@torch.no_grad()
def _export_vision_model(model, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    _export_embeddings(model.embeddings, os.path.join(outpath, "embeddings"))
    _export_LayerNorm(model.pre_layrnorm, os.path.join(outpath, "pre_layernorm"))
    _export_encoder(model.encoder, os.path.join(outpath, "encoder"))


def _export_embeddings(embeddings, prefix):
    # class_embedding
    outpath = prefix + "/class_embedding"
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(embeddings.class_embedding.cpu().float().numpy().tobytes())
    # patch_embedding
    outpath = prefix + "/patch_embedding"
    os.makedirs(outpath, exist_ok=True)
    # print(f"Transpose patch_embedding from {embeddings.patch_embedding.weight.cpu().float().numpy().shape} to {embeddings.patch_embedding.weight.cpu().float().numpy().transpose(0, 2, 3, 1).shape}")
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        # f.write(embeddings.patch_embedding.weight.cpu().float().numpy().tobytes())
        f.write(embeddings.patch_embedding.weight.cpu().float().numpy().transpose(0, 2, 3, 1).tobytes())
    # position_embedding
    outpath = prefix + "/position_embedding"
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(embeddings.position_embedding.weight.cpu().float().numpy().tobytes())


def _export_encoder(model, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    for idx, layer in enumerate(model.layers):
        _export_encoder_layer(layer, os.path.join(outpath, f"layer{idx}"))


def _export_encoder_layer(layer, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    _export_attention_params(layer.self_attn, os.path.join(outpath, "self_attn"))
    _export_LayerNorm(layer.layer_norm1, os.path.join(outpath, "layer_norm1"))
    _export_linearfp(layer.mlp.fc1, os.path.join(outpath, "mlp_fc1"))
    _export_linearfp(layer.mlp.fc2, os.path.join(outpath, "mlp_fc2"))
    _export_LayerNorm(layer.layer_norm2, os.path.join(outpath, "layer_norm2"))


def _export_LayerNorm(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op.weight.cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "bias.bin"), "wb") as f:
        f.write(op.bias.cpu().float().numpy().tobytes())


def _export_linearfp(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op._parameters["weight"].cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "bias.bin"), "wb") as f:
        f.write(op._parameters["bias"].cpu().float().numpy().tobytes())


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
    _export_linearfp(attn.out_proj, os.path.join(outpath, "out_proj"))
    qk_bmm_alpha = 1 / math.sqrt(attn.head_dim)
    _export_BMM_F32T(qk_bmm_alpha, os.path.join(outpath, "qk_bmm"))


def _export_processor(processor, prefix):
    outpath = prefix + "/image_processor"
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "image_mean.bin"), "wb") as f:
        # f.write(processor.image_processor.image_mean.numpy().astype(np.float32).tobytes())
        # Convert list to numpy array
        f.write(np.array(processor.image_processor.image_mean).astype(np.float32).tobytes())
        
    with open(os.path.join(f"{outpath}", "image_std.bin"), "wb") as f:
        # f.write(processor.image_processor.image_std.numpy().astype(np.float32).tobytes())
        # Convert list to numpy array
        f.write(np.array(processor.image_processor.image_std).astype(np.float32).tobytes())

def main():
    """Export a Clip model to TinyChatEngine format."""
    parser = argparse.ArgumentParser(description="export Clip pytorch model to TinyChatEngine format.")
    parser.add_argument("--hf_path", type=str, help="Path to huggingface model hub", default=None)
    parser.add_argument("--model", type=str, help="Path of the Clip torch model")
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
            if args.model.split("/")[-1].lower().startswith("clip"):
                if args.model.split("-")[1].lower() == "vit" and args.model.split("-")[2].lower() == "large":
                    print("Loading Clip ViT Large .pt model...");
                    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)
                    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
                else:
                    print("Model size not supported.")
                    return
            else:
                print("Model type not supported.")
                return
            
            model.load_state_dict(torch.load(args.model))
        else:
            print("Loading Clip ViT Large model...")
            model = CLIPModel.from_pretrained(args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True, offload_state_dict=True)
            processor = CLIPProcessor.from_pretrained(args.model)
    else:
        print("Loading Clip ViT Large model from Hugging Face...")
        model = CLIPModel.from_pretrained(args.hf_path, torch_dtype=torch.float32)
        processor = CLIPProcessor.from_pretrained(args.hf_path)

    print("Start exporting Clip Vision model...")
    print("Pop out the last layer of the vision model.")
    model.vision_model.encoder.layers.pop(-1)
    # for name, param in model.named_parameters():
    #     print (name)
    _export_vision_model(model.vision_model, args.output)
    _export_processor(processor, args.output)
    print("Finished exporting Clip Vision model.")


if __name__ == "__main__":
    main()
