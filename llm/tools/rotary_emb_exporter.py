"""Implementation of exporting rotary_emb to TinyChatEngine format.

Usage:
   python rotary_emb_exporter.py <output dir>

Example commandline:
   python tools/rotary_emb_exporter.py models/Mistral_7B
"""
import argparse
import os
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

class RotaryEmbedding():
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.inv_freq = inv_freq

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

        self.device = device

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, seq_len=None, output_path=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=self.device, dtype=torch.get_default_dtype())

        # return (
        #     self.cos_cached[:seq_len].to(dtype=x.dtype),
        #     self.sin_cached[:seq_len].to(dtype=x.dtype),
        # )
        # Save cos and sin to file as binary with float32
        with open(os.path.join(output_path, "cos_cached.bin"), "wb") as f:
            f.write(self.cos_cached[:seq_len].cpu().float().numpy().tobytes())
        with open(os.path.join(output_path, "sin_cached.bin"), "wb") as f:
            f.write(self.sin_cached[:seq_len].cpu().float().numpy().tobytes())

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    if args.output_dir:
        output_path = args.output_dir
    else:
        output_path = "models/Mistral_7B/decoder/layer0/self_attn/rotary_emb"
        print(f"output_path not defined, using {output_path} by default.")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Define the model and device is CPU
    model = RotaryEmbedding(dim=128, max_position_embeddings=32768, base=1000000, device="cpu")
    # Run the model
    model.forward(seq_len=32768, output_path=output_path)

    # Copy the cos and sin to other layers
    def copy_file(src, dst):
        import shutil
        shutil.copy(src, dst)
    # Copy from layer0 to layer1 - 31
    for i in range(1, 32):
        copy_file(os.path.join(output_path, "cos_cached.bin"), os.path.join(output_path.replace("layer0", f"layer{i}"), "cos_cached.bin"))
        copy_file(os.path.join(output_path, "sin_cached.bin"), os.path.join(output_path.replace("layer0", f"layer{i}"), "sin_cached.bin"))
    

if __name__ == "__main__":
    main()
