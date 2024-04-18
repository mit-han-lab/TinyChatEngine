"""Quantize LLaMA model to specific Quantization methods.

Usage:
   python model_quantizer.py --model_path <path to TinyEngine modelt> --method <Quantization method>

Example commands:
   python tools/model_quantizer.py --model_path models/LLaMA_7B_2_chat --method QM_x86

   python tools/model_quantizer.py --model_path FP32/models/OPT_125m --method QM_ARM --output_path INT4

"""
import argparse
import os
import shutil

import numpy as np
from quantize_constants import STORE_FP16
from quantize_methods import (
    _convert_to_fp16,
    quantize_row_q4_2,
    quantize_row_q4_3,
    quantize_row_q4_4,
    quantize_row_q4_6,
)

quantization_funcs = {
    "QM_x86": quantize_row_q4_3,
    "QM_METAL": quantize_row_q4_2,
    "QM_ARM": quantize_row_q4_4,
    "QM_CUDA": quantize_row_q4_6,
}


# Write quantized data into binary file
def _write_weight_to_file(prefix: str, qs, d, m, zp, is_cuda=False, is_lm_head=False):
    # Convert to bytes
    if is_cuda:
        qs_data = np.asarray(qs, dtype=np.int32).tobytes()
        d_data = np.asarray(d, dtype=np.float16).tobytes()  # Need to be converted to fp16 in CUDA
        m_data = np.asarray(
            m, dtype=np.float16
        ).tobytes()  # TODO: Currently, we don't use offsets for CUDA so this is redundant
        zp_data = np.asarray(zp, dtype=np.int32).tobytes()
    else:
        qs_data = np.asarray(qs, dtype=np.uint8).tobytes()
        d_data = np.asarray(d, dtype=np.float32).tobytes()
        m_data = np.asarray(m, dtype=np.float32).tobytes()
        zp_data = np.asarray(zp, dtype=np.float32).tobytes()

    # Write data
    if is_lm_head:
        out_path = prefix + "/lm_head"
    else:
        out_path = prefix
    os.makedirs(out_path, exist_ok=True)

    with open(out_path + "/weight_int4.bin", "wb") as f:
        f.write(qs_data)
    with open(out_path + "/scaling_factor_int4.bin", "wb") as f:
        f.write(d_data)
    with open(out_path + "/offset_int4.bin", "wb") as f:
        f.write(m_data)
    with open(out_path + "/zero_point_int4.bin", "wb") as f:
        f.write(zp_data)

    f.close()


# Quantize fp32 data to fp16, and write it into binary file
def _write_fp16_to_file(output_path: str, input_path: str, array_size, dim_x, dim_y, dim_z):
    assert array_size == dim_x * dim_y * dim_z

    with open(input_path, mode="rb") as fp:
        fp32_data = fp.read()
    fp.close()

    # Convert to bytes
    fp16_data = _convert_to_fp16(np.frombuffer(fp32_data, dtype=np.float32)).tobytes()

    os.makedirs(output_path, exist_ok=True)
    output_path = output_path.split(".bin")[0] + "_half.bin"
    with open(output_path, "wb") as f:
        f.write(fp16_data)

    f.close()


def _rm_and_cp_dir_if_exist(src, des):
    if os.path.exists(des):
        shutil.rmtree(des)
    shutil.copytree(src, des)


# Read quantized data from binary file
def _read_weight_from_file(prefix: str):
    print(f"Reading quantized data from {prefix}...")
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


# Quantize model
def _quantize_model(
    prefix: str,
    method: str,
    output_path: str,
    data_type: str = "fp32",
    is_cuda: bool = False,
):
    # Check model name
    model_name_size = prefix.split("/")[-1]
    if model_name_size == "OPT_125m":
        layer_num = 12
    elif model_name_size == "OPT_1.3B":
        layer_num = 24
    elif model_name_size == "OPT_6.7B":
        layer_num = 32
    elif model_name_size.startswith("LLaMA_7B") or model_name_size.startswith("CodeLLaMA_7B") or model_name_size.startswith("LLaVA_7B") or model_name_size.startswith("VILA_7B") or model_name_size.startswith("VILA_2.7B"):
        layer_num = 32
    elif model_name_size.startswith("Mistral_7B"):
        layer_num = 32
    elif model_name_size.startswith("LLaMA_13B") or model_name_size.startswith("CodeLLaMA_13B") or model_name_size.startswith("LLaVA_13B") or model_name_size.startswith("VILA_13B"):
        layer_num = 40
    elif model_name_size.startswith("StarCoder"):
        layer_num = 40
    else:
        raise ValueError(
            "Invalid model name. Expected 'OPT_125m', 'OPT_1.3B', 'OPT_6.7B', 'LLaMA_7B', 'LLaMA_13B', 'CodeLLaMA_7B', \
            'CodeLLaMA_13B', 'StarCoder', 'LLaVA_7B', 'LLaVA_13B', 'VILA_2.7B', 'VILA_7B', 'VILA_13B', or 'Mistral_7B'."
        )

    # Check quantization method
    if method not in quantization_funcs:
        raise ValueError(f"Invalid quantization method. Expected one of {quantization_funcs.keys()}")
    quantize_method = quantization_funcs[method]

    # Check data type
    if data_type == "fp32":
        bytes_per_element = 4
    elif data_type == "fp16":
        bytes_per_element = 2
    elif data_type == "int8":
        bytes_per_element = 1
    else:
        raise ValueError("Invalid data type. Expected 'fp32', 'fp16', or 'int8'.")

    print(f"Quantizing {model_name_size} with {method} method... (original data type: {data_type})")

    model_name = model_name_size
    # OPT
    if model_name.startswith("OPT"):
        # Quantize lm_head
        file_path = f"{prefix}"
        weight_path = f"{file_path}/lm_head.bin"
        file_size_bytes = os.path.getsize(weight_path)
        if file_size_bytes % bytes_per_element != 0:
            raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        array_size = file_size_bytes // bytes_per_element
        qs, d, m, zp = quantize_method(weight_path, array_size, data_type, 1, array_size)  # TODO: fix this for QM_CUDA
        _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda, True)
        print("Quantization of lm_head finished.")

        # Quantize embed_positions
        dir_path = f"{prefix}/decoder/embed_positions"
        _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

        # embed_tokens
        dir_path = f"{prefix}/decoder/embed_tokens"
        _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

        # final_layer_norm
        dir_path = f"{prefix}/decoder/final_layer_norm"
        _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

        # Quantize layers
        for idx in range(layer_num):
            # Quantize fc1
            file_path = f"{prefix}/decoder/layer{idx}/fc1"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(
                weight_path, array_size, data_type, 1, array_size
            )  # TODO: fix this for QM_CUDA
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)
            os.system(f"cp {os.path.join(file_path, 'bias.bin')} {os.path.join(output_path, file_path, 'bias.bin')}")

            # Quantize fc2
            file_path = f"{prefix}/decoder/layer{idx}/fc2"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(
                weight_path, array_size, data_type, 1, array_size
            )  # TODO: fix this for QM_CUDA
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)
            os.system(f"cp {os.path.join(file_path, 'bias.bin')} {os.path.join(output_path, file_path, 'bias.bin')}")

            # final_layer_norm
            dir_path = f"{prefix}/decoder/layer{idx}/final_layer_norm"
            _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

            # self_attn_layer_norm
            dir_path = f"{prefix}/decoder/layer{idx}/self_attn_layer_norm"
            _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

            # Quantize self_attn/k_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/k_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(
                weight_path, array_size, data_type, 1, array_size
            )  # TODO: fix this for QM_CUDA
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)
            os.system(f"cp {os.path.join(file_path, 'bias.bin')} {os.path.join(output_path, file_path, 'bias.bin')}")

            # Quantize self_attn/out_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/out_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(
                weight_path, array_size, data_type, 1, array_size
            )  # TODO: fix this for QM_CUDA
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)
            os.system(f"cp {os.path.join(file_path, 'bias.bin')} {os.path.join(output_path, file_path, 'bias.bin')}")

            # Quantize self_attn/q_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/q_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(
                weight_path, array_size, data_type, 1, array_size
            )  # TODO: fix this for QM_CUDA
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)
            os.system(f"cp {os.path.join(file_path, 'bias.bin')} {os.path.join(output_path, file_path, 'bias.bin')}")

            # Quantize self_attn/v_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/v_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(
                weight_path, array_size, data_type, 1, array_size
            )  # TODO: fix this for QM_CUDA
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)
            os.system(f"cp {os.path.join(file_path, 'bias.bin')} {os.path.join(output_path, file_path, 'bias.bin')}")

            # Scaling
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/"
            os.system(
                f"cp {os.path.join(file_path, 'scaling.bin')} {os.path.join(output_path, file_path, 'scaling.bin')}"
            )

            print(f"Quantization of layer {idx} finished.")

    # LLaMA / LLaVA / VILA
    elif model_name.startswith("LLaMA") or model_name.startswith("CodeLLaMA") or model_name.startswith("LLaVA") \
         or model_name.startswith("VILA") or model_name.startswith("Mistral"):
        if model_name.startswith("LLaMA_7B") or model_name.startswith("CodeLLaMA_7B") or model_name.startswith("LLaVA_7B") \
           or model_name.startswith("VILA_7B"):
            embed_dim = 4096
            hidden_dim = 11008
        elif model_name.startswith("LLaMA_13B") or model_name.startswith("CodeLLaMA_13B") or model_name.startswith("LLaVA_13B") \
             or model_name.startswith("VILA_13B"):
            embed_dim = 5120
            hidden_dim = 13824
        elif model_name.startswith("VILA_2.7B"):
            embed_dim = 2560
            hidden_dim = 6912
        elif model_name.startswith("Mistral_7B"):
            embed_dim = 4096
            hidden_dim = 14336
        else:
            raise NotImplementedError(f"{model_name} not supported.")
        
        if model_name.startswith("LLaMA_7B") or model_name.startswith("LLaMA_13B") or model_name.startswith("LLaVA_7B") or model_name.startswith("LLaVA_13B") or model_name.startswith("VILA_2.7B") or model_name.startswith("Mistral_7B"):
            vocab_size = 32000
        elif model_name.startswith("VILA_2.7B") or mmodel_name.startswith("VILA_7B") or model_name.startswith("VILA_13B"):
            vocab_size = 32000
        elif model_name.startswith("CodeLLaMA_7B") or model_name.startswith("CodeLLaMA_13B"):
            vocab_size = 32016
        
        if model_name.startswith("LLaVA_7B") or model_name.startswith("LLaVA_13B") or model_name.startswith("VILA_2.7B") or model_name.startswith("VILA_7B") or model_name.startswith("VILA_13B"):
            max_seq_len = 4096
        elif model_name.startswith("Mistral_7B"):
            max_seq_len = 32768
        else:
            max_seq_len = 2048

        # Quantize lm_head
        file_path = f"{prefix}"
        weight_path = f"{file_path}/lm_head.bin"
        file_size_bytes = os.path.getsize(weight_path)
        if file_size_bytes % bytes_per_element != 0:
            raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        array_size = file_size_bytes // bytes_per_element
        qs, d, m, zp = quantize_method(weight_path, array_size, data_type, embed_dim, vocab_size)
        _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda, True)
        print("Quantization of lm_head finished.")

        # Quantize embed_tokens
        dir_path = f"{prefix}/decoder/embed_tokens"
        _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))
        print("Quantization of embed_tokens finished.")

        # Quantize layers
        for idx in range(layer_num):
            # Quantize down_proj
            file_path = f"{prefix}/decoder/layer{idx}/down_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(weight_path, array_size, data_type, hidden_dim, embed_dim)
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)

            # Quantize gate_proj
            file_path = f"{prefix}/decoder/layer{idx}/gate_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(weight_path, array_size, data_type, embed_dim, hidden_dim)
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)

            # Quantize input_layernorm
            dir_path = f"{prefix}/decoder/layer{idx}/input_layernorm"
            _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

            # Quantize post_attention_layernorm
            dir_path = f"{prefix}/decoder/layer{idx}/post_attention_layernorm"
            _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

            # Quantize self_attn/k_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/k_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(weight_path, array_size, data_type, embed_dim, embed_dim)
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)

            # Quantize self_attn/o_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/o_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(weight_path, array_size, data_type, embed_dim, embed_dim)
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)

            # Quantize self_attn/q_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/q_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(weight_path, array_size, data_type, embed_dim, embed_dim)
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)

            # Quantize self_attn/v_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/v_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(weight_path, array_size, data_type, embed_dim, embed_dim)
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)

            # Quantize self_attn/rotary_emb
            # Quantize self_attn/qk_bmm
            if is_cuda:
                file_path = f"{prefix}/decoder/layer{idx}/self_attn/rotary_emb"
                # cos_cached.bin
                weight_path = f"{file_path}/cos_cached.bin"
                file_size_bytes = os.path.getsize(weight_path)
                if file_size_bytes % bytes_per_element != 0:
                    raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
                array_size = file_size_bytes // bytes_per_element
                _write_fp16_to_file(os.path.join(output_path, weight_path), weight_path, array_size, 1, max_seq_len, 128)
                # sin_cached.bin
                weight_path = f"{file_path}/sin_cached.bin"
                file_size_bytes = os.path.getsize(weight_path)
                if file_size_bytes % bytes_per_element != 0:
                    raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
                array_size = file_size_bytes // bytes_per_element
                _write_fp16_to_file(os.path.join(output_path, weight_path), weight_path, array_size, 1, max_seq_len, 128)
                file_path = f"{prefix}/decoder/layer{idx}/self_attn/qk_bmm"
                weight_path = f"{file_path}/alpha.bin"
                file_size_bytes = os.path.getsize(weight_path)
                if file_size_bytes % bytes_per_element != 0:
                    raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
                array_size = file_size_bytes // bytes_per_element
                _write_fp16_to_file(os.path.join(output_path, weight_path), weight_path, array_size, 1, 1, 1)
            else:
                # cp self_attn/rotary_emb
                dir_path = f"{prefix}/decoder/layer{idx}/self_attn/rotary_emb"
                _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))
                # cp self_attn/qk_bmm
                dir_path = f"{prefix}/decoder/layer{idx}/self_attn/qk_bmm"
                _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

            # Quantize up_proj
            file_path = f"{prefix}/decoder/layer{idx}/up_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(weight_path, array_size, data_type, embed_dim, hidden_dim)
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)

            print(f"Quantization of layer {idx} finished.")

        # Quantize norm
        file_path = f"{prefix}/decoder/norm"
        weight_path = f"{file_path}/weight.bin"
        _rm_and_cp_dir_if_exist(file_path, os.path.join(output_path, file_path))
        print("Quantization of norm finished.")

    # StarCoder
    elif model_name.startswith("StarCoder"):
        # Quantize lm_head
        file_path = f"{prefix}"
        weight_path = f"{file_path}/lm_head.bin"
        file_size_bytes = os.path.getsize(weight_path)
        if file_size_bytes % bytes_per_element != 0:
            raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        array_size = file_size_bytes // bytes_per_element
        qs, d, m, zp = quantize_method(weight_path, array_size, data_type, 1, array_size)  # TODO: fix this for QM_CUDA
        _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda, True)
        print("Quantization of lm_head finished.")

        # Quantize wpe
        dir_path = f"{prefix}/decoder/wpe"
        _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

        # wte
        dir_path = f"{prefix}/decoder/wte"
        _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

        # ln_f
        dir_path = f"{prefix}/decoder/ln_f"
        _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

        # Quantize layers
        for idx in range(layer_num):
            # Quantize c_fc
            file_path = f"{prefix}/decoder/layer{idx}/c_fc"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(
                weight_path, array_size, data_type, 1, array_size
            )  # TODO: fix this for QM_CUDA
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)
            os.system(f"cp {os.path.join(file_path, 'bias.bin')} {os.path.join(output_path, file_path, 'bias.bin')}")

            # Quantize c_proj
            file_path = f"{prefix}/decoder/layer{idx}/c_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(
                weight_path, array_size, data_type, 1, array_size
            )  # TODO: fix this for QM_CUDA
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)
            os.system(f"cp {os.path.join(file_path, 'bias.bin')} {os.path.join(output_path, file_path, 'bias.bin')}")

            # ln_1
            dir_path = f"{prefix}/decoder/layer{idx}/ln_1"
            _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

            # ln_2
            dir_path = f"{prefix}/decoder/layer{idx}/ln_2"
            _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

            # Quantize attn/c_attn
            file_path = f"{prefix}/decoder/layer{idx}/attn/c_attn"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(
                weight_path, array_size, data_type, 1, array_size
            )  # TODO: fix this for QM_CUDA
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)
            os.system(f"cp {os.path.join(file_path, 'bias.bin')} {os.path.join(output_path, file_path, 'bias.bin')}")

            # Quantize attn/c_proj
            file_path = f"{prefix}/decoder/layer{idx}/attn/c_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            qs, d, m, zp = quantize_method(
                weight_path, array_size, data_type, 1, array_size
            )  # TODO: fix this for QM_CUDA
            _write_weight_to_file(os.path.join(output_path, file_path), qs, d, m, zp, is_cuda)
            os.system(f"cp {os.path.join(file_path, 'bias.bin')} {os.path.join(output_path, file_path, 'bias.bin')}")

            # cp attn/qk_bmm
            dir_path = f"{prefix}/decoder/layer{idx}/attn/qk_bmm"
            _rm_and_cp_dir_if_exist(dir_path, os.path.join(output_path, dir_path))

            print(f"Quantization of layer {idx} finished.")

    print(f"All the weights of {model_name_size} has been quantized with {method} method.")


# Test function
def _test():
    print("Test function starts.")

    prefix = "models/LLaMA_7B"
    method = "Q4_0"
    data_type = "fp32"

    # Check model name
    model_name_size = prefix.rsplit("/", maxsplit=1)[-1]
    if model_name_size not in ["OPT_125m", "OPT_1.3B", "OPT_6.7B", "LLaMA_7B"]:
        raise ValueError("Invalid model name. Expected 'OPT_125m', 'OPT_1.3B', 'OPT_6.7B', or 'LLaMA_7B'.")

    # Check quantization method
    if method not in quantization_funcs:
        raise ValueError(f"Invalid quantization method. Expected one of {quantization_funcs.keys()}")
    quantize_method = quantization_funcs[method]

    # Check data type
    if data_type == "fp32":
        bytes_per_element = 4
    elif data_type == "fp16":
        bytes_per_element = 2
    elif data_type == "int8":
        bytes_per_element = 1
    else:
        raise ValueError("Invalid data type. Expected 'fp32', 'fp16', or 'int8'.")

    # Quantize down_proj in layer 0
    file_path = f"{prefix}"
    weight_path = f"{prefix}/lm_head.bin"
    file_size_bytes = os.path.getsize(weight_path)
    if file_size_bytes % bytes_per_element != 0:
        raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
    array_size = file_size_bytes // bytes_per_element
    print(f"Quantizing '{weight_path}' with {method} method... (original data type: {data_type})")
    qs, d, m, zp = quantize_method(weight_path, array_size, data_type)

    file_path += "/lm_head"

    _write_weight_to_file(file_path, qs, d, m, zp)

    read_qs, read_d, read_m, read_zp = _read_weight_from_file(file_path)

    # Check weights
    # first_half_qs = np.bitwise_and(qs, 0x0F)
    # second_half_qs = np.bitwise_and(qs, 0xF0) >> 4
    # first_half_read_qs = np.bitwise_and(np.frombuffer(read_qs, dtype=np.int8), 0x0F)
    # second_half_read_qs = np.bitwise_and(np.frombuffer(read_qs, dtype=np.int8), 0xF0) >> 4
    # print(f"first_half_qs:       {first_half_qs[0:2, :16]}")
    # print(f"first_half_read_qs:  {first_half_read_qs[:32]}")
    # print(f"second_half_qs:      {second_half_qs[0:2, :16]}")
    # print(f"second_half_read_qs: {second_half_read_qs[:32]}")

    # print(f"shalen of qs:       {qs.shape}")
    # print(f"length of first_half_qs:       {len(first_half_qs)}")
    # print(f"length of second_half_qs:      {len(second_half_qs)}")
    # print(f"length of first_half_read_qs:  {len(first_half_read_qs)}")
    # print(f"length of second_half_read_qs: {len(second_half_read_qs)}")

    # Check weights
    qs = np.frombuffer(qs, dtype=np.uint8)
    qs = np.array(qs, dtype=np.int32)
    print(f"qs:      {qs.flatten()[:32]}")
    print(f"length of qs:      {len(qs)}")
    read_qs = np.frombuffer(read_qs, dtype=np.uint8)
    read_qs = np.array(read_qs, dtype=np.int32)
    print(f"read_qs:      {read_qs[:32]}")
    print(f"length of read_qs:      {len(read_qs)}")

    # Check scaling factors
    if STORE_FP16:
        read_d = np.frombuffer(read_d, dtype=np.float16)
    else:
        read_d = np.frombuffer(read_d, dtype=np.float32)
    print(f"d:      {d}")
    print(f"read_d: {read_d}")
    print(f"length of d:      {len(d)}")

    # Check offsets
    if STORE_FP16:
        read_m = np.frombuffer(read_m, dtype=np.float16)
    else:
        read_m = np.frombuffer(read_m, dtype=np.float32)
    print(f"m:      {m}")
    print(f"read_m: {read_m}")
    print(f"length of m:      {len(m)}")

    # Check zero points
    if STORE_FP16:
        read_zp = np.frombuffer(read_zp, dtype=np.float16)
    else:
        read_zp = np.frombuffer(read_zp, dtype=np.float32)
    print(f"zp:      {zp}")
    print(f"read_zp: {read_zp}")


# Main function
def main():
    """Take arguments and quantize the model."""

    def _get_parser():
        parser = argparse.ArgumentParser(description="Quantize model")
        parser.add_argument("--model_path", type=str, default="models/LLaMA_7B", help="Model path")
        parser.add_argument("--method", type=str, default="QM_ARM", help="Quantization method")
        parser.add_argument("--data_type", type=str, default="fp32", help="Data type")
        parser.add_argument("--output_path", type=str, default=None, help="Quantization method")
        return parser

    parser = _get_parser()
    args = parser.parse_args()
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = "INT4"
        print(f"output_path not defined, using {output_path} by default.")
    _quantize_model(
        prefix=args.model_path,
        method=args.method,
        output_path=output_path,
        data_type=args.data_type,
        is_cuda=args.method == "QM_CUDA",
    )


if __name__ == "__main__":
    main()
