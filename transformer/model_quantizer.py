import argparse
import os

import numpy as np
# import torch

STORE_FP16 = False

QK4_0 = 32
QK4_1 = 32

# Define structure replacements
class block_q4_0:
    def __init__(self):
        self.d = 0
        self.qs = np.zeros(QK4_0 // 2, dtype=np.uint8)


class block_q4_1:
    def __init__(self):
        self.d = 0
        self.m = 0
        self.qs = np.zeros(QK4_1 // 2, dtype=np.uint8)


# Converters
def convert_to_fp16(val):
    return np.float16(val)


# 4-bit Quantization method 0
def quantize_row_q4_0(input_path, k, data_type, cuda_is_available, input_channel, output_channel, group_size):
    qk = group_size
    assert k % qk == 0
    nb = k // qk

    assert k == input_channel * output_channel

    with open(input_path, mode="rb") as fp:
        origin_weight = fp.read()
    fp.close()

    if data_type == "fp32":
        x = np.frombuffer(origin_weight, dtype=np.float32)
    elif data_type == "fp16":
        x = np.frombuffer(origin_weight, dtype=np.float16)
    elif data_type == "int8":
        x = np.frombuffer(origin_weight, dtype=np.int8)

    # Reshape x to be a 2D array with shape (nb, qk)
    x = x.reshape(nb, qk)

    # Get the indices of maximum absolute values along axis 1
    idx_max_abs = np.argmax(np.abs(x), axis=1)
    max_vals = x[np.arange(x.shape[0]), idx_max_abs]
    # max_vals = x.max(axis=1)
    min_vals = np.zeros(nb, dtype=np.float32)
    d_vals = max_vals / -8

    id_vals = 1.0 / d_vals
    id_vals[d_vals == 0] = 0.0

    if STORE_FP16:
        d = convert_to_fp16(d_vals)  # scaling factors
        m = convert_to_fp16(min_vals)  # offsets
        zp = convert_to_fp16(8.0)  # zero point
    else:
        d = np.float32(d_vals)  # scaling factors
        m = np.float32(min_vals)  # offsets
        zp = np.float32([8.0])  # zero point

    if cuda_is_available:
        xi = ((x * id_vals[:, np.newaxis]) + 8.5).clip(0, 15).astype(np.uint32)
        xi = xi.reshape(-1).reshape(output_channel, input_channel).transpose()
        qs = np.zeros((input_channel, output_channel // 8), dtype=np.int32)
    
        # Store weights in row major for CUDA (kernel: IC, OC // 8 [int32])
        for idx in range(output_channel // 8):
            qs[:, idx] = qs[:, idx] | xi[:, idx * 8] | (xi[:, idx * 8 + 1] << 4) | (xi[:, idx * 8 + 2] << 8) | (xi[:, idx * 8 + 3] << 12) | (xi[:, idx * 8 + 4] << 16) | (xi[:, idx * 8 + 5] << 20) | (xi[:, idx * 8 + 6] << 24) | (xi[:, idx * 8 + 7] << 28)

        # Store scaling_factors in row major for CUDA (scaling_factors: IC // G, OC [float16])
        d = d.reshape(-1).reshape(output_channel, input_channel // qk).transpose()

        # Store zero_points in row major for CUDA (zeros: IC // G, OC // 8 [int32])
        # print(f"zp_before: {zp}")
        zp = zp.astype(np.uint32)
        # print(f"zp_after: {zp}")
        zp_pack = np.zeros(1, dtype=np.int32)
        # print(f"zp_pack_before: {zp_pack}")
        zp_pack = zp | (zp << 4) | (zp << 8) | (zp << 12) | (zp << 16) | (zp << 20) | (zp << 24) | (zp << 28)
        # print(f"zp_pack_after: {zp_pack}")
        zp = np.tile(zp_pack, (input_channel // qk, output_channel // 8))

        # TODO: Currently, we don't use offsets for CUDA

    else:
        xi = ((x * id_vals[:, np.newaxis]) + 8.5).clip(0, 15).astype(np.uint8)
        qs = np.zeros((nb, qk // 2), dtype=np.uint8)

        # Store weights in column major for CPU (kernel: OC, IC // 2 [uint8])
        for idx in range(qk // 2):
            qs[:, idx] = xi[:, idx * 2] | (xi[:, idx * 2 + 1] << 4)
        

    return qs, d, m, zp


# 4-bit Quantization method 1
def quantize_row_q4_1(input_path, k, data_type, cuda_is_available, input_channel, output_channel, group_size):
    qk = group_size
    assert k % qk == 0
    nb = k // qk

    assert k == input_channel * output_channel

    with open(input_path, mode="rb") as fp:
        origin_weight = fp.read()
    fp.close()

    if data_type == "fp32":
        x = np.frombuffer(origin_weight, dtype=np.float32)
    elif data_type == "fp16":
        x = np.frombuffer(origin_weight, dtype=np.float16)
    elif data_type == "int8":
        x = np.frombuffer(origin_weight, dtype=np.int8)

    # Reshape x to be a 2D array with shape (nb, qk)
    x = x.reshape(nb, qk)

    max_vals = x.max(axis=1)
    min_vals = x.min(axis=1)
    d_vals = (max_vals - min_vals) / 15

    id_vals = 1.0 / d_vals
    id_vals[d_vals == 0] = 0.0

    if STORE_FP16:
        d = convert_to_fp16(d_vals)  # scaling factors
        m = convert_to_fp16(min_vals)  # offsets
        zp = convert_to_fp16(0.5)  # zero point
    else:
        d = np.float32(d_vals)  # scaling factors
        m = np.float32(min_vals)  # offsets
        zp = np.float32(0.5)  # zero point
    qs = np.zeros((nb, qk // 2), dtype=np.uint8)

    xi = (((x - min_vals[:, np.newaxis]) * id_vals[:, np.newaxis]) + zp).clip(0, 15).astype(np.uint8)
    # xi0 = xi[:, :qk//2]
    # xi1 = xi[:, qk//2:]
    # qs = xi0 | (xi1 << 4)

    for idx in range(qk // 2):
        qs[:, idx] = xi[:, idx * 2] | (xi[:, idx * 2 + 1] << 4)

    return qs, d, m, zp


# Write quantized data into binary file
def write_weight_to_file(prefix: str, qs, d, m, zp, is_lm_head=False, cuda_is_available=False):
    # Convert to bytes
    if cuda_is_available:
        qs_data = np.asarray(qs, dtype=np.int32).tobytes()
        d_data = np.asarray(d, dtype=np.float16).tobytes()
        m_data = np.asarray(m, dtype=np.float32).tobytes() # TODO: Currently, we don't use offsets for CUDA so this is redundant
        zp_data = np.asarray(zp, dtype=np.int32).tobytes()
    else:
        qs_data = np.asarray(qs, dtype=np.uint8).tobytes()
        d_data = np.asarray(d, dtype=np.float32).tobytes()
        m_data = np.asarray(m, dtype=np.float32).tobytes()
        zp_data = np.asarray(zp, dtype=np.float32).tobytes()

    # Write data
    if is_lm_head:
        out_path = prefix + "/lm_head"
        os.makedirs(out_path, exist_ok=True)
    else:
        out_path = prefix

    with open(out_path + "/weight_int4.bin", "wb") as f:
        f.write(qs_data)
    with open(out_path + "/scaling_factor_int4.bin", "wb") as f:
        f.write(d_data)
    with open(out_path + "/offset_int4.bin", "wb") as f:
        f.write(m_data)
    with open(out_path + "/zero_point_int4.bin", "wb") as f:
        f.write(zp_data)

    f.close()


# Read quantized data from binary file
def read_weight_from_file(prefix: str):
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
def quantize_model(prefix, method="Q4_0", data_type="fp32", cuda_is_available=False, group_size=128):
    # Check model name
    model_name_size = prefix.split("/")[-1]
    if model_name_size == "OPT_125m":
        layer_num = 12
    elif model_name_size == "OPT_1.3B":
        layer_num = 24
    elif model_name_size == "OPT_6.7B":
        layer_num = 32
    elif model_name_size == "LLaMA_7B":
        layer_num = 32
    else:
        raise ValueError("Invalid model name. Expected 'OPT_125m', 'OPT_1.3B', 'OPT_6.7B', or 'LLaMA_7B'.")

    # Check quantization method
    if method not in ["Q4_0", "Q4_1"]:
        raise ValueError("Invalid quantization method. Expected 'Q4_0' or 'Q4_1'.")

    # Check data type
    if data_type == "fp32":
        bytes_per_element = 4
    elif data_type == "fp16":
        bytes_per_element = 2
    elif data_type == "int8":
        bytes_per_element = 1
    else:
        raise ValueError("Invalid data type. Expected 'fp32', 'fp16', or 'int8'.")

    if cuda_is_available:
        print(f"Quantizing {model_name_size} with {method} method and group size {group_size} for GPUs (cuda_is_available={cuda_is_available}). Original data type: {data_type}")
    else:
        print(f"Quantizing {model_name_size} with {method} method and group size {group_size} for CPUs (cuda_is_available={cuda_is_available}). Original data type: {data_type}")

    model_name = model_name_size.split("_")[0]
    # OPT
    if model_name == "OPT":
        # Quantize lm_head
        file_path = f"{prefix}"
        weight_path = f"{file_path}/lm_head.bin"
        file_size_bytes = os.path.getsize(weight_path)
        if file_size_bytes % bytes_per_element != 0:
            raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        array_size = file_size_bytes // bytes_per_element
        if method == "Q4_0":
            qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
        elif method == "Q4_1":
            qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
        write_weight_to_file(file_path, qs, d, m, zp, True, cuda_is_available)
        print(f"Quantization of lm_head finished.")

        # # Quantize embed_positions
        # file_path = f"{prefix}/decoder/embed_positions"
        # weight_path = f"{file_path}/weight.bin"
        # file_size_bytes = os.path.getsize(weight_path)
        # if file_size_bytes % bytes_per_element != 0:
        #     raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        # array_size = file_size_bytes // bytes_per_element
        # if method == "Q4_0":
        #     qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
        # elif method == "Q4_1":
        #     qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
        # write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)
        # print(f"Quantization of embed_positions finished.")

        # # Quantize embed_tokens
        # file_path = f"{prefix}/decoder/embed_tokens"
        # weight_path = f"{file_path}/weight.bin"
        # file_size_bytes = os.path.getsize(weight_path)
        # if file_size_bytes % bytes_per_element != 0:
        #     raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        # array_size = file_size_bytes // bytes_per_element
        # if method == "Q4_0":
        #     qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
        # elif method == "Q4_1":
        #     qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
        # write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)
        # print(f"Quantization of embed_tokens finished.")

        # # Quantize final_layer_norm
        # file_path = f"{prefix}/decoder/final_layer_norm"
        # weight_path = f"{file_path}/weight.bin"
        # file_size_bytes = os.path.getsize(weight_path)
        # if file_size_bytes % bytes_per_element != 0:
        #     raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        # array_size = file_size_bytes // bytes_per_element
        # if method == "Q4_0":
        #     qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
        # elif method == "Q4_1":
        #     qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
        # write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)
        # print(f"Quantization of final_layer_norm finished.")

        # Quantize layers
        for idx in range(layer_num):
            # Quantize fc1
            file_path = f"{prefix}/decoder/layer{idx}/fc1"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # Quantize fc2
            file_path = f"{prefix}/decoder/layer{idx}/fc2"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # # Quantize final_layer_norm
            # file_path = f"{prefix}/decoder/layer{idx}/final_layer_norm"
            # weight_path = f"{file_path}/weight.bin"
            # file_size_bytes = os.path.getsize(weight_path)
            # if file_size_bytes % bytes_per_element != 0:
            #     raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            # array_size = file_size_bytes // bytes_per_element
            # if method == "Q4_0":
            #     qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
            # elif method == "Q4_1":
            #     qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
            # write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # Quantize self_attn/k_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/k_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # Quantize self_attn/out_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/out_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # Quantize self_attn/q_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/q_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # Quantize self_attn/v_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/v_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # # Quantize self_attn_layer_norm
            # file_path = f"{prefix}/decoder/layer{idx}/self_attn_layer_norm"
            # weight_path = f"{file_path}/weight.bin"
            # file_size_bytes = os.path.getsize(weight_path)
            # if file_size_bytes % bytes_per_element != 0:
            #     raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            # array_size = file_size_bytes // bytes_per_element
            # if method == "Q4_0":
            #     qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
            # elif method == "Q4_1":
            #     qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
            # write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # print(f"Quantization of layer {idx} finished.")

    # LLaMA
    elif model_name == "LLaMA":
        # Quantize lm_head
        file_path = f"{prefix}"
        weight_path = f"{file_path}/lm_head.bin"
        file_size_bytes = os.path.getsize(weight_path)
        if file_size_bytes % bytes_per_element != 0:
            raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        array_size = file_size_bytes // bytes_per_element
        if method == "Q4_0":
            qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available, 4096, 32000, group_size)
        elif method == "Q4_1":
            qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available, 4096, 32000, group_size)
        write_weight_to_file(file_path, qs, d, m, zp, True, cuda_is_available)
        print(f"Quantization of lm_head finished.")

        # # Quantize embed_tokens
        # file_path = f"{prefix}/decoder/embed_tokens"
        # weight_path = f"{file_path}/weight.bin"
        # file_size_bytes = os.path.getsize(weight_path)
        # if file_size_bytes % bytes_per_element != 0:
        #     raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        # array_size = file_size_bytes // bytes_per_element
        # if method == "Q4_0":
        #     qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available, group_size)
        # elif method == "Q4_1":
        #     qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available, group_size)
        # write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)
        # print(f"Quantization of embed_tokens finished.")

        # Quantize layers
        for idx in range(layer_num):
            # Quantize down_proj
            file_path = f"{prefix}/decoder/layer{idx}/down_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available, 11008, 4096, group_size)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available, 11008, 4096, group_size)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # Quantize gate_proj
            file_path = f"{prefix}/decoder/layer{idx}/gate_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available, 4096, 11008, group_size)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available, 4096, 11008, group_size)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # # Quantize input_layernorm
            # file_path = f"{prefix}/decoder/layer{idx}/input_layernorm"
            # weight_path = f"{file_path}/weight.bin"
            # file_size_bytes = os.path.getsize(weight_path)
            # if file_size_bytes % bytes_per_element != 0:
            #     raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            # array_size = file_size_bytes // bytes_per_element
            # if method == "Q4_0":
            #     qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
            # elif method == "Q4_1":
            #     qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
            # write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # # Quantize post_attention_layernorm
            # file_path = f"{prefix}/decoder/layer{idx}/post_attention_layernorm"
            # weight_path = f"{file_path}/weight.bin"
            # file_size_bytes = os.path.getsize(weight_path)
            # if file_size_bytes % bytes_per_element != 0:
            #     raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            # array_size = file_size_bytes // bytes_per_element
            # if method == "Q4_0":
            #     qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
            # elif method == "Q4_1":
            #     qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
            # write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # Quantize self_attn/k_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/k_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available, 4096, 4096, group_size)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available, 4096, 4096, group_size)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # Quantize self_attn/o_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/o_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available, 4096, 4096, group_size)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available, 4096, 4096, group_size)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # Quantize self_attn/q_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/q_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available, 4096, 4096, group_size)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available, 4096, 4096, group_size)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # Quantize self_attn/v_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/v_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available, 4096, 4096, group_size)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available, 4096, 4096, group_size)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            # Quantize up_proj
            file_path = f"{prefix}/decoder/layer{idx}/up_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == "Q4_0":
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available, 4096, 11008, group_size)
            elif method == "Q4_1":
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available, 4096, 11008, group_size)
            write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

            print(f"Quantization of layer {idx} finished.")

        # # Quantize norm
        # file_path = f"{prefix}/decoder/norm"
        # weight_path = f"{file_path}/weight.bin"
        # file_size_bytes = os.path.getsize(weight_path)
        # if file_size_bytes % bytes_per_element != 0:
        #     raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        # array_size = file_size_bytes // bytes_per_element
        # if method == "Q4_0":
        #     qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
        # elif method == "Q4_1":
        #     qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)
        # write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)
        # print(f"Quantization of norm finished.")

    if cuda_is_available:
        print(f"All the {data_type} weights of {model_name_size} has been quantized with {method} method and group size {group_size} for GPUs (cuda_is_available={cuda_is_available}).")    
    else:
        print(f"All the {data_type} weights of {model_name_size} has been quantized with {method} method and group size {group_size} for CPUs (cuda_is_available={cuda_is_available}).")    


# Test function
def test():
    print("Test function starts.")

    # Assume your array is arr
    arr = np.random.rand(5, 4)

    # Transpose your array
    arr_transposed = arr.transpose()

    print(f"arr: {arr}")
    print(f"arr_transposed: {arr_transposed}")

    # prefix = "models/LLaMA_7B"
    # method = "Q4_0"
    # data_type = "fp32"
    # # cuda_is_available = torch.cuda.is_available()
    # cuda_is_available = False

    # # Check model name
    # model_name_size = prefix.split("/")[-1]
    # if model_name_size not in ["OPT_125m", "OPT_1.3B", "OPT_6.7B", "LLaMA_7B"]:
    #     raise ValueError("Invalid model name. Expected 'OPT_125m', 'OPT_1.3B', 'OPT_6.7B', or 'LLaMA_7B'.")

    # # Check quantization method
    # if method not in ["Q4_0", "Q4_1"]:
    #     raise ValueError("Invalid quantization method. Expected 'Q4_0' or 'Q4_1'.")

    # # Check data type
    # if data_type == "fp32":
    #     bytes_per_element = 4
    # elif data_type == "fp16":
    #     bytes_per_element = 2
    # elif data_type == "int8":
    #     bytes_per_element = 1
    # else:
    #     raise ValueError("Invalid data type. Expected 'fp32', 'fp16', or 'int8'.")

    # # Quantize down_proj in layer 0
    # file_path = f"{prefix}"
    # weight_path = f"{prefix}/lm_head.bin"
    # file_size_bytes = os.path.getsize(weight_path)
    # if file_size_bytes % bytes_per_element != 0:
    #     raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
    # array_size = file_size_bytes // bytes_per_element
    # print(f"Quantizing '{weight_path}' with {method} method... (original data type: {data_type})")
    # if method == "Q4_0":
    #     qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type, cuda_is_available)
    # elif method == "Q4_1":
    #     qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type, cuda_is_available)

    # file_path += "/lm_head"

    # write_weight_to_file(file_path, qs, d, m, zp, False, cuda_is_available)

    # read_qs, read_d, read_m, read_zp = read_weight_from_file(file_path)

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

    # # Check weights
    # qs = np.frombuffer(qs, dtype=np.uint8)
    # qs = np.array(qs, dtype=np.int32)
    # print(f"qs:      {qs.flatten()[:32]}")
    # print(f"length of qs:      {len(qs)}")
    # read_qs = np.frombuffer(read_qs, dtype=np.uint8)
    # read_qs = np.array(read_qs, dtype=np.int32)
    # print(f"read_qs:      {read_qs[:32]}")
    # print(f"length of read_qs:      {len(read_qs)}")

    # # Check scaling factors
    # if STORE_FP16:
    #     read_d = np.frombuffer(read_d, dtype=np.float16)
    # else:
    #     read_d = np.frombuffer(read_d, dtype=np.float32)
    # print(f"d:      {d}")
    # print(f"read_d: {read_d}")
    # print(f"length of d:      {len(d)}")

    # # Check offsets
    # if STORE_FP16:
    #     read_m = np.frombuffer(read_m, dtype=np.float16)
    # else:
    #     read_m = np.frombuffer(read_m, dtype=np.float32)
    # print(f"m:      {m}")
    # print(f"read_m: {read_m}")
    # print(f"length of m:      {len(m)}")

    # # Check zero points
    # if STORE_FP16:
    #     read_zp = np.frombuffer(read_zp, dtype=np.float16)
    # else:
    #     read_zp = np.frombuffer(read_zp, dtype=np.float32)
    # print(f"zp:      {zp}")
    # print(f"read_zp: {read_zp}")


# Main function
def main():
    def get_parser():
        parser = argparse.ArgumentParser(description="Quantize model")
        parser.add_argument("--model_path", type=str, default="models/LLaMA_7B", help="Model path")
        parser.add_argument("--method", type=str, default="Q4_0", help="Quantization method")
        parser.add_argument("--data_type", type=str, default="fp32", help="Data type")
        parser.add_argument("--group_size", type=int, default=128, help="Quantization group size")
        parser.add_argument("--cuda_is_available", type=bool, default=False, help="Quantize weights into general format or GPU format")
        return parser

    parser = get_parser()
    args = parser.parse_args()
    # cuda_is_available = torch.cuda.is_available()
    print(f"Quantization START!")
    quantize_model(prefix=args.model_path, method=args.method, data_type=args.data_type, cuda_is_available=args.cuda_is_available, group_size=args.group_size)
    print(f"Quantization DONE!")


if __name__ == "__main__":
    main()
    # test()
