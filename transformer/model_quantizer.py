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

    # Only for testing
    # print_x = x.reshape(-1).reshape(output_channel, input_channel).transpose()
    # print(f"size of print_x: {print_x.shape}")
    # print(f"print_x: {print_x}")

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
        xi = ((x * id_vals[:, np.newaxis]) + 8.5).clip(0, 15).astype(np.int32)
        xi = xi.reshape(-1).reshape(output_channel, input_channel).transpose()
        qs = np.zeros((input_channel, output_channel // 8), dtype=np.int32)
    
        # Store weights in row major for CUDA (kernel: IC, OC // 8 [int32])
        # order of weights is 0 2 4 6 1 3 5 7
        for idx in range(output_channel // 8):
            # qs[:, idx] = qs[:, idx] | xi[:, idx * 8] | (xi[:, idx * 8 + 1] << 4) | (xi[:, idx * 8 + 2] << 8) | (xi[:, idx * 8 + 3] << 12) \
            #  | (xi[:, idx * 8 + 4] << 16) | (xi[:, idx * 8 + 5] << 20) | (xi[:, idx * 8 + 6] << 24) | (xi[:, idx * 8 + 7] << 28)
            qs[:, idx] = qs[:, idx] | xi[:, idx * 8] | (xi[:, idx * 8 + 2] << 4) | (xi[:, idx * 8 + 4] << 8) | (xi[:, idx * 8 + 6] << 12) | (xi[:, idx * 8 + 1] << 16) | (xi[:, idx * 8 + 3] << 20) | (xi[:, idx * 8 + 5] << 24) | ((xi[:, idx * 8 + 7] & 0xf) << 28)

        # Store scaling_factors in row major for CUDA (scaling_factors: IC // G, OC [float16])
        d = d.reshape(-1).reshape(output_channel, input_channel // qk).transpose()

        # unreorder_d = d.reshape(-1).reshape(output_channel, input_channel // qk).transpose()
        # # for j in range(0, output_channel, 8):
        # #     d[:, j] = unreorder_d[:, j]
        # #     d[:, j + 1] = unreorder_d[:, j + 2]
        # #     d[:, j + 2] = unreorder_d[:, j + 4]
        # #     d[:, j + 3] = unreorder_d[:, j + 6]
        # #     d[:, j + 4] = unreorder_d[:, j + 1]
        # #     d[:, j + 5] = unreorder_d[:, j + 3]
        # #     d[:, j + 6] = unreorder_d[:, j + 5]
        # #     d[:, j + 7] = unreorder_d[:, j + 7]
        # indices = np.arange(output_channel)
        # indices = (indices // 8 * 8) + np.array([0, 2, 4, 6, 1, 3, 5, 7])[indices % 8]
        # d = unreorder_d[:, indices]

        # print(f"ddddd: {d[0, :32]}")

        # Store zero_points in row major for CUDA (zeros: IC // G, OC // 8 [int32])
        # print(f"zp_before: {zp}")
        zp = zp.astype(np.int32)
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
        d_data = np.asarray(d, dtype=np.float16).tobytes() # Need to ne converted to fp16 in CUDA
        m_data = np.asarray(m, dtype=np.float16).tobytes() # TODO: Currently, we don't use offsets for CUDA so this is redundant
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
    print("Test function START!")

    prefix = "models/LLaMA_7B"
    method = "Q4_0"
    data_type = "fp32"
    # cuda_is_available = torch.cuda.is_available()

    # Check model name
    model_name_size = prefix.split("/")[-1]
    if model_name_size not in ["OPT_125m", "OPT_1.3B", "OPT_6.7B", "LLaMA_7B"]:
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

    ### Testing CPU version
    print("Testing CPU version!")

    # Quantize down_proj in layer 0
    cuda_is_available = False
    group_size = 32
    file_path = f"{prefix}/decoder/layer0/down_proj"
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
    read_qs, read_d, read_m, read_zp = read_weight_from_file(file_path)

    # # Check weights
    # first_half_qs = np.bitwise_and(qs, 0x0F)
    # second_half_qs = np.bitwise_and(qs, 0xF0) >> 4
    # first_half_read_qs = np.bitwise_and(np.frombuffer(read_qs, dtype=np.uint8), 0x0F)
    # second_half_read_qs = np.bitwise_and(np.frombuffer(read_qs, dtype=np.uint8), 0xF0) >> 4
    # # print(f"first_half_qs:       {first_half_qs[0:2, :16].reshape(-1)}")
    # # print(f"first_half_read_qs:  {first_half_read_qs[:32]}")
    # if np.array_equal(first_half_qs.reshape(-1), first_half_read_qs):
    #     print("first_half_qs and first_half_read_qs are equal.")
    # else:
    #     raise ValueError("first_half_qs and first_half_read_qs are NOT equal.")
    # # print(f"second_half_qs:      {second_half_qs[0:2, :16].reshape(-1)}")
    # # print(f"second_half_read_qs: {second_half_read_qs[:32]}")
    # if np.array_equal(second_half_qs.reshape(-1), second_half_read_qs):
    #     print("second_half_qs and second_half_read_qs are equal.")
    # else:
    #     raise ValueError("second_half_qs and second_half_read_qs are NOT equal.")

    # print(f"shape of qs:       {qs.shape}")
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
    # # print(f"d:      {d}")
    # # print(f"read_d: {read_d}")
    # # print(f"length of d:      {len(d)}")
    # if np.array_equal(d, read_d):
    #     print("d and read_d are equal.")
    # else:
    #     raise ValueError("d and read_d are NOT equal.")

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
    # # print(f"zp:      {zp}")
    # # print(f"read_zp: {read_zp}")
    # if np.array_equal(zp, read_zp):
    #     print("zp and read_zp are equal.\n")
    # else:
    #     raise ValueError("zp and read_zp are NOT equal.")


    ### Testing GPI version
    print("Testing GPU version!")

    # Quantize down_proj in layer 0
    cuda_is_available = True
    group_size = 32
    file_path = f"{prefix}/decoder/layer0/self_attn/q_proj"
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
    read_qs, read_d, read_m, read_zp = read_weight_from_file(file_path)

    # # Check weights
    # first_half_qs =   np.bitwise_and(qs, 0x0000000F)
    # second_half_qs =  np.bitwise_and(qs, 0x000000F0) >> 4
    # third_half_qs =   np.bitwise_and(qs, 0x00000F00) >> 8
    # fourth_half_qs =  np.bitwise_and(qs, 0x0000F000) >> 12
    # fifth_half_qs =   np.bitwise_and(qs, 0x000F0000) >> 16
    # sixth_half_qs =   np.bitwise_and(qs, 0x00F00000) >> 20
    # seventh_half_qs = np.bitwise_and(qs, 0x0F000000) >> 24
    # eighth_half_qs =  np.bitwise_and(qs, 0xF0000000) >> 28
    first_half_read_qs =   np.bitwise_and(np.frombuffer(read_qs, dtype=np.int32), 0x0000000F)
    second_half_read_qs =  np.bitwise_and(np.frombuffer(read_qs, dtype=np.int32), 0x000000F0) >> 4
    third_half_read_qs =   np.bitwise_and(np.frombuffer(read_qs, dtype=np.int32), 0x00000F00) >> 8
    fourth_half_read_qs =  np.bitwise_and(np.frombuffer(read_qs, dtype=np.int32), 0x0000F000) >> 12
    fifth_half_read_qs =   np.bitwise_and(np.frombuffer(read_qs, dtype=np.int32), 0x000F0000) >> 16
    sixth_half_read_qs =   np.bitwise_and(np.frombuffer(read_qs, dtype=np.int32), 0x00F00000) >> 20
    seventh_half_read_qs = np.bitwise_and(np.frombuffer(read_qs, dtype=np.int32), 0x0F000000) >> 24
    eighth_half_read_qs =  np.bitwise_and(np.frombuffer(read_qs, dtype=np.int32), 0xF0000000) >> 28

    print(f"qs:       {qs[0, :128]}")

    # # print(f"first_half_qs:       {first_half_qs[0:2, :16].reshape(-1)}")
    # # print(f"first_half_read_qs:  {first_half_read_qs[:32]}")
    # if np.array_equal(first_half_qs.reshape(-1), first_half_read_qs):
    #     print("first_half_qs and first_half_read_qs are equal.")
    # else:
    #     raise ValueError("first_half_qs and first_half_read_qs are NOT equal.")
    # # print(f"second_half_qs:      {second_half_qs[0:2, :16].reshape(-1)}")
    # # print(f"second_half_read_qs: {second_half_read_qs[:32]}")
    # if np.array_equal(second_half_qs.reshape(-1), second_half_read_qs):
    #     print("second_half_qs and second_half_read_qs are equal.")
    # else:
    #     raise ValueError("second_half_qs and second_half_read_qs are NOT equal.")
    
    # if np.array_equal(third_half_qs.reshape(-1), third_half_read_qs):
    #     print("third_half_qs and third_half_read_qs are equal.")
    # else:
    #     raise ValueError("third_half_qs and third_half_read_qs are NOT equal.")

    # if np.array_equal(fourth_half_qs.reshape(-1), fourth_half_read_qs):
    #     print("fourth_half_qs and fourth_half_read_qs are equal.")
    # else:
    #     raise ValueError("fourth_half_qs and fourth_half_read_qs are NOT equal.")
    
    # if np.array_equal(fifth_half_qs.reshape(-1), fifth_half_read_qs):
    #     print("fifth_half_qs and fifth_half_read_qs are equal.")
    # else:
    #     raise ValueError("fifth_half_qs and fifth_half_read_qs are NOT equal.")

    # if np.array_equal(sixth_half_qs.reshape(-1), sixth_half_read_qs):
    #     print("sixth_half_qs and sixth_half_read_qs are equal.")
    # else:
    #     raise ValueError("sixth_half_qs and sixth_half_read_qs are NOT equal.")
    
    # if np.array_equal(seventh_half_qs.reshape(-1), seventh_half_read_qs):
    #     print("seventh_half_qs and seventh_half_read_qs are equal.")
    # else:
    #     raise ValueError("seventh_half_qs and seventh_half_read_qs are NOT equal.")
    
    # if np.array_equal(eighth_half_qs.reshape(-1), eighth_half_read_qs):
    #     print("eighth_half_qs and eighth_half_read_qs are equal.")
    # else:
    #     raise ValueError("eighth_half_qs and eighth_half_read_qs are NOT equal.")
    
    # # Check scaling factors
    read_d = np.frombuffer(read_d, dtype=np.float32)
    print(f"d: {d[0, :128]}")
    # # print(f"read_d: {read_d[:32]}")
    # print(f"length of d:      {len(d.reshape(-1))}")
    # print(f"length of read_d: {len(read_d)}")
    # if np.array_equal(d.reshape(-1), read_d):
    #     print("d and read_d are equal.")
    # else:
    #     raise ValueError("d and read_d are NOT equal.")
    
    # # Check zero points
    # first_half_zp =   np.bitwise_and(zp, 0x0000000F)
    # second_half_zp =  np.bitwise_and(zp, 0x000000F0) >> 4
    # third_half_zp =   np.bitwise_and(zp, 0x00000F00) >> 8
    # fourth_half_zp =  np.bitwise_and(zp, 0x0000F000) >> 12
    # fifth_half_zp =   np.bitwise_and(zp, 0x000F0000) >> 16
    # sixth_half_zp =   np.bitwise_and(zp, 0x00F00000) >> 20
    # seventh_half_zp = np.bitwise_and(zp, 0x0F000000) >> 24
    # eighth_half_zp =  np.bitwise_and(zp, 0xF0000000) >> 28
    first_half_read_zp =   np.bitwise_and(np.frombuffer(read_zp, dtype=np.int32), 0x0000000F)
    second_half_read_zp =  np.bitwise_and(np.frombuffer(read_zp, dtype=np.int32), 0x000000F0) >> 4
    third_half_read_zp =   np.bitwise_and(np.frombuffer(read_zp, dtype=np.int32), 0x00000F00) >> 8
    fourth_half_read_zp =  np.bitwise_and(np.frombuffer(read_zp, dtype=np.int32), 0x0000F000) >> 12
    fifth_half_read_zp =   np.bitwise_and(np.frombuffer(read_zp, dtype=np.int32), 0x000F0000) >> 16
    sixth_half_read_zp =   np.bitwise_and(np.frombuffer(read_zp, dtype=np.int32), 0x00F00000) >> 20
    seventh_half_read_zp = np.bitwise_and(np.frombuffer(read_zp, dtype=np.int32), 0x0F000000) >> 24
    eighth_half_read_zp =  np.bitwise_and(np.frombuffer(read_zp, dtype=np.int32), 0xF0000000) >> 28

    print(f"zp:       {zp[0, :128]}")

    # print(f"first_half_zp: {first_half_zp[:, :2]}")
    # if np.array_equal(first_half_zp.reshape(-1), first_half_read_zp):
    #     print("first_half_zp and first_half_read_zp are equal.")
    # else:
    #     raise ValueError("first_half_zp and first_half_read_zp are NOT equal.")
    
    # if np.array_equal(second_half_zp.reshape(-1), second_half_read_zp):
    #     print("second_half_zp and second_half_read_zp are equal.")
    # else:
    #     raise ValueError("second_half_zp and second_half_read_zp are NOT equal.")
    
    # if np.array_equal(third_half_zp.reshape(-1), third_half_read_zp):
    #     print("third_half_zp and third_half_read_zp are equal.")
    # else:
    #     raise ValueError("third_half_zp and third_half_read_zp are NOT equal.")

    # if np.array_equal(fourth_half_zp.reshape(-1), fourth_half_read_zp):
    #     print("fourth_half_zp and fourth_half_read_zp are equal.")
    # else:
    #     raise ValueError("fourth_half_zp and fourth_half_read_zp are NOT equal.")
    
    # if np.array_equal(fifth_half_zp.reshape(-1), fifth_half_read_zp):
    #     print("fifth_half_zp and fifth_half_read_zp are equal.")
    # else:
    #     raise ValueError("fifth_half_zp and fifth_half_read_zp are NOT equal.")

    # if np.array_equal(sixth_half_zp.reshape(-1), sixth_half_read_zp):
    #     print("sixth_half_zp and sixth_half_read_zp are equal.")
    # else:
    #     raise ValueError("sixth_half_zp and sixth_half_read_zp are NOT equal.")
    
    # if np.array_equal(seventh_half_zp.reshape(-1), seventh_half_read_zp):
    #     print("seventh_half_zp and seventh_half_read_zp are equal.")
    # else:
    #     raise ValueError("seventh_half_zp and seventh_half_read_zp are NOT equal.")
    
    # if np.array_equal(eighth_half_zp.reshape(-1), eighth_half_read_zp):
    #     print("eighth_half_zp and eighth_half_read_zp are equal.\n")
    # else:
    #     raise ValueError("eighth_half_zp and eighth_half_read_zp are NOT equal.")


    # Check final result
    first_half_read_qs = first_half_read_qs.reshape(-1).reshape(11008, 512)
    second_half_read_qs = second_half_read_qs.reshape(-1).reshape(11008, 512)
    third_half_read_qs = third_half_read_qs.reshape(-1).reshape(11008, 512)
    fourth_half_read_qs = fourth_half_read_qs.reshape(-1).reshape(11008, 512)
    fifth_half_read_qs = fifth_half_read_qs.reshape(-1).reshape(11008, 512)
    sixth_half_read_qs = sixth_half_read_qs.reshape(-1).reshape(11008, 512)
    seventh_half_read_qs = seventh_half_read_qs.reshape(-1).reshape(11008, 512)
    eighth_half_read_qs = eighth_half_read_qs.reshape(-1).reshape(11008, 512)

    first_half_read_zp = first_half_read_zp.reshape(-1).reshape(11008 // group_size, 512)
    second_half_read_zp = second_half_read_zp.reshape(-1).reshape(11008 // group_size, 512)
    third_half_read_zp = third_half_read_zp.reshape(-1).reshape(11008 // group_size, 512)
    fourth_half_read_zp = fourth_half_read_zp.reshape(-1).reshape(11008 // group_size, 512)
    fifth_half_read_zp = fifth_half_read_zp.reshape(-1).reshape(11008 // group_size, 512)
    sixth_half_read_zp = sixth_half_read_zp.reshape(-1).reshape(11008 // group_size, 512)
    seventh_half_read_zp = seventh_half_read_zp.reshape(-1).reshape(11008 // group_size, 512)
    eighth_half_read_zp = eighth_half_read_zp.reshape(-1).reshape(11008 // group_size, 512)

    read_d = read_d.reshape(-1).reshape(11008 // group_size, 4096)
    # result = np.zeros((11008, 4096), dtype=np.float32)
    
    # for i in range(11008):
    #     for j in range(512):
    #         result[i, j * 8] = (first_half_read_qs[i, j] - first_half_read_zp[i // group_size, j]) * read_d[i // group_size, j * 8]
    #         result[i, j * 8 + 1] = (second_half_read_qs[i, j] - second_half_read_zp[i // group_size, j]) * read_d[i // group_size, j * 8 + 1]
    #         result[i, j * 8 + 2] = (third_half_read_qs[i, j] - third_half_read_zp[i // group_size, j]) * read_d[i // group_size, j * 8 + 2]
    #         result[i, j * 8 + 3] = (fourth_half_read_qs[i, j] - fourth_half_read_zp[i // group_size, j]) * read_d[i // group_size, j * 8 + 3]
    #         result[i, j * 8 + 4] = (fifth_half_read_qs[i, j] - fifth_half_read_zp[i // group_size, j]) * read_d[i // group_size, j * 8 + 4]
    #         result[i, j * 8 + 5] = (sixth_half_read_qs[i, j] - sixth_half_read_zp[i // group_size, j]) * read_d[i // group_size, j * 8 + 5]
    #         result[i, j * 8 + 6] = (seventh_half_read_qs[i, j] - seventh_half_read_zp[i // group_size, j]) * read_d[i // group_size, j * 8 + 6]
    #         result[i, j * 8 + 7] = (eighth_half_read_qs[i, j] - eighth_half_read_zp[i // group_size, j]) * read_d[i // group_size, j * 8 + 7]

    # print(f"result:      {result[0, :32]}\n")

    i_indices = np.arange(11008)[:, None]
    j_indices = np.arange(512)[None, :]
    result_new = np.zeros((11008, 4096), dtype=np.float32)

    result_new[i_indices, j_indices * 8] = (first_half_read_qs[i_indices, j_indices] - first_half_read_zp[i_indices // group_size, j_indices]) * read_d[i_indices // group_size, j_indices * 8]
    result_new[i_indices, j_indices * 8 + 1] = (second_half_read_qs[i_indices, j_indices] - second_half_read_zp[i_indices // group_size, j_indices]) * read_d[i_indices // group_size, j_indices * 8 + 1]
    result_new[i_indices, j_indices * 8 + 2] = (third_half_read_qs[i_indices, j_indices] - third_half_read_zp[i_indices // group_size, j_indices]) * read_d[i_indices // group_size, j_indices * 8 + 2]
    result_new[i_indices, j_indices * 8 + 3] = (fourth_half_read_qs[i_indices, j_indices] - fourth_half_read_zp[i_indices // group_size, j_indices]) * read_d[i_indices // group_size, j_indices * 8 + 3]
    result_new[i_indices, j_indices * 8 + 4] = (fifth_half_read_qs[i_indices, j_indices] - fifth_half_read_zp[i_indices // group_size, j_indices]) * read_d[i_indices // group_size, j_indices * 8 + 4]
    result_new[i_indices, j_indices * 8 + 5] = (sixth_half_read_qs[i_indices, j_indices] - sixth_half_read_zp[i_indices // group_size, j_indices]) * read_d[i_indices // group_size, j_indices * 8 + 5]
    result_new[i_indices, j_indices * 8 + 6] = (seventh_half_read_qs[i_indices, j_indices] - seventh_half_read_zp[i_indices // group_size, j_indices]) * read_d[i_indices // group_size, j_indices * 8 + 6]
    result_new[i_indices, j_indices * 8 + 7] = (eighth_half_read_qs[i_indices, j_indices] - eighth_half_read_zp[i_indices // group_size, j_indices]) * read_d[i_indices // group_size, j_indices * 8 + 7]

    print(f"result_new:      {result_new}")

    # if np.array_equal(result, result_new):
    #     print("result and result_new are equal!")
    # else:
    #     raise ValueError("result and result_new are NOT equal!")
    
    print("Test function DONE!")


# Main function
def main():
    def get_parser():
        parser = argparse.ArgumentParser(description="Quantize model")
        parser.add_argument("--model_path", type=str, default="models/LLaMA_7B", help="Model path")
        parser.add_argument("--method", type=str, default="Q4_0", help="Quantization method")
        parser.add_argument("--data_type", type=str, default="fp32", help="Data type")
        parser.add_argument("--group_size", type=int, default=32, help="Quantization group size")
        # TODO: We should remove the following line and make it detect CUDA automatically in the future.
        parser.add_argument("--cuda_is_available", type=bool, default=False, help="Quantize weights into general format or GPU format")
        return parser

    parser = get_parser()
    args = parser.parse_args()
    # TODO: We should use the following line in the future (currently we set cuda_is_available manually)
    # cuda_is_available = torch.cuda.is_available()
    # TODO: We should remove the following four lines in the future (currently we only support group_size=32 for CPUs and group_size=128 for GPUs)
    # if args.cuda_is_available:
    #     args.group_size = 128
    # else:
    #     args.group_size = 32

    print(f"Quantization START!")
    quantize_model(prefix=args.model_path, method=args.method, data_type=args.data_type, cuda_is_available=args.cuda_is_available, group_size=args.group_size)
    print(f"Quantization DONE!")


if __name__ == "__main__":
    main()
    #test()
