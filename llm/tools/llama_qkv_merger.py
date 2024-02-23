"""Merge LLaMA model's qkv layers.

Usage:
    python llama_qkv_merger.py --model_path <model_path> --output_path <output_path>

Example commands:
    python llama_qkv_merger.py --model_path INT4/models/LLaMA_7B_2_chat --output_path INT4/models/LLaMA_7B_2_chat
"""
import argparse
import os
import numpy as np

# Merge QKV
def _merge_qkv(input_path: str, output_path: str):
    # self_attn/q_proj
    weight_path = f"{input_path}/q_proj"
    qs_q, d_q, m_q, zp_q = _read_weight_from_file(weight_path)
    # self_attn/k_proj
    weight_path = f"{input_path}/k_proj"
    qs_k, d_k, m_k, zp_k = _read_weight_from_file(weight_path)
    # self_attn/v_proj
    weight_path = f"{input_path}/v_proj"
    qs_v, d_v, m_v, zp_v = _read_weight_from_file(weight_path)

    # Merge QKV
    ## Merge weight
    qs_q = np.frombuffer(qs_q, dtype=np.int32)
    qs_k = np.frombuffer(qs_k, dtype=np.int32)
    qs_v = np.frombuffer(qs_v, dtype=np.int32)
    qs_qkv = np.concatenate((qs_q, qs_k, qs_v), axis=0)
    ## Merge scaling factor
    d_q = np.frombuffer(d_q, dtype=np.float32)
    d_k = np.frombuffer(d_k, dtype=np.float32)
    d_v = np.frombuffer(d_v, dtype=np.float32)
    d_qkv = np.concatenate((d_q, d_k, d_v), axis=0)
    ## Merge offset
    m_q = np.frombuffer(m_q, dtype=np.int32)
    m_k = np.frombuffer(m_k, dtype=np.int32)
    m_v = np.frombuffer(m_v, dtype=np.int32)
    m_qkv = np.concatenate((m_q, m_k, m_v), axis=0)
    ## Merge zero point
    zp_q = np.frombuffer(zp_q, dtype=np.int32)
    zp_k = np.frombuffer(zp_k, dtype=np.int32)
    zp_v = np.frombuffer(zp_v, dtype=np.int32)
    zp_qkv = np.concatenate((zp_q, zp_k, zp_v), axis=0)
    
    # Write merged QKV
    os.makedirs(output_path, exist_ok=True)
    with open(output_path + "/weight_int4.bin", "wb") as f:
        f.write(qs_qkv)
    with open(output_path + "/scaling_factor_int4.bin", "wb") as f:
        f.write(d_qkv)
    with open(output_path + "/offset_int4.bin", "wb") as f:
        f.write(m_qkv)
    with open(output_path + "/zero_point_int4.bin", "wb") as f:
        f.write(zp_qkv)

    f.close()

# Read quantized data from binary file
def _read_weight_from_file(prefix: str):
    with open(prefix + "/weight_int4.bin", "rb") as f:
        qs_data = f.read()
    with open(prefix + "/scaling_factor_int4.bin", "rb") as f:
        d_data = f.read()
    with open(prefix + "/offset_int4.bin", "rb") as f:
        m_data = f.read()
    with open(prefix + "/zero_point_int4.bin", "rb") as f:
        zp_data = f.read()

    f.close()

    return qs_data, d_data, m_data, zp_data


# Merge model
def _merge_model(
    prefix: str,
    output_path: str,
):
    # Check model name
    model_name_size = prefix.split("/")[-1]
    if model_name_size.startswith("LLaMA_7B") or model_name_size.startswith("CodeLLaMA_7B") or model_name_size.startswith("LLaVA_7B") or model_name_size.startswith("VILA_7B"):
        layer_num = 32
    elif model_name_size.startswith("LLaMA_13B") or model_name_size.startswith("CodeLLaMA_13B"):
        layer_num = 40
    else:
        raise ValueError(
            "Invalid model name. Expected 'LLaMA_7B', 'CodeLLaMA_7B', 'LLaMA_13B', 'CodeLLaMA_13B', 'LLaVA_7B', or 'VILA_7B'."
        )

    print(f"Merge {model_name_size}'s QKV layers...")

    model_name = model_name_size

    # LLaMA
    if model_name.startswith("LLaMA") or model_name.startswith("CodeLLaMA") or model_name.startswith("LLaVA") or model_name.startswith("VILA"):
        if model_name.startswith("LLaMA_7B") or model_name.startswith("CodeLLaMA_7B") or model_name.startswith("LLaVA_7B") or model_name.startswith("VILA_7B"):
            embed_dim = 4096
            hidden_dim = 11008
        elif model_name.startswith("LLaMA_13B") or model_name.startswith("CodeLLaMA_13B"):
            embed_dim = 5120
            hidden_dim = 13824
        else:
            raise NotImplementedError(f"{model_name} not supported.")

        # Merge layers
        for idx in range(layer_num):
            # Merge QKV
            _merge_qkv(
                input_path=f"{prefix}/decoder/layer{idx}/self_attn",
                output_path=f"{output_path}/decoder/layer{idx}/self_attn/qkv_proj",
            )

    print(f"All the QKV of {model_name_size} has been merged.")

# Main function
def main():
    """Take arguments and merge the model's QKV."""

    def _get_parser():
        parser = argparse.ArgumentParser(description="Merge model's QKV layers.")
        parser.add_argument("--model_path", type=str, default="INT4/models/LLaMA_13B_2_chat", help="Model path")
        parser.add_argument("--output_path", type=str, default=None, help="Output path")
        return parser

    parser = _get_parser()
    args = parser.parse_args()
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = args.model_path
        print(f"output_path not defined, using {args.model_path} by default.")
    _merge_model(
        prefix=args.model_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
