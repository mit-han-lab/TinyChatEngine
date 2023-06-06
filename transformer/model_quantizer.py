import os
import numpy as np
import argparse

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
def quantize_row_q4_0(input_path, k, data_type):
    qk = QK4_0
    assert k % qk == 0
    nb = k // qk

    with open(input_path, mode='rb') as fp:
        origin_weight = fp.read()
    fp.close()

    if data_type == 'fp32':
        x = np.frombuffer(origin_weight, dtype=np.float32)
    elif data_type == 'fp16':
        x = np.frombuffer(origin_weight, dtype=np.float16)
    elif data_type == 'int8':
        x = np.frombuffer(origin_weight, dtype=np.int8)

    # Reshape x to be a 2D array with shape (nb, qk)
    x = x.reshape(nb, qk)

    max_vals = x.max(axis=1)
    min_vals = np.zeros(nb, dtype=np.float32)
    d_vals = max_vals / -8

    id_vals = 1.0 / d_vals
    id_vals[d_vals == 0] = 0.0

    if STORE_FP16:
        d = convert_to_fp16(d_vals)  # scaling factors
        m = convert_to_fp16(min_vals)  # offsets
        zp = convert_to_fp16(8.5)  # zero point
    else:
        d = d_vals  # scaling factors
        m = min_vals  # offsets
        zp = np.float32(8.5)  # zero point
    qs = np.zeros((nb, qk // 2), dtype=np.uint8)

    xi = ((x * id_vals[:, np.newaxis]) + zp).clip(0, 15).astype(np.uint8)
    xi0 = xi[:, :qk//2]
    xi1 = xi[:, qk//2:]
    qs = xi0 | (xi1 << 4)

    return qs, d, m, zp

# 4-bit Quantization method 1
def quantize_row_q4_1(input_path, k, data_type):
    qk = QK4_1
    assert k % qk == 0
    nb = k // qk

    with open(input_path, mode='rb') as fp:
        origin_weight = fp.read()
    fp.close()

    if data_type == 'fp32':
        x = np.frombuffer(origin_weight, dtype=np.float32)
    elif data_type == 'fp16':
        x = np.frombuffer(origin_weight, dtype=np.float16)
    elif data_type == 'int8':
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
        d = d_vals  # scaling factors
        m = min_vals  # offsets
        zp = np.float32(0.5)  # zero point
    qs = np.zeros((nb, qk // 2), dtype=np.uint8)

    xi = (((x - min_vals[:, np.newaxis]) * id_vals[:, np.newaxis]) + zp).clip(0, 15).astype(np.uint8)
    xi0 = xi[:, :qk//2]
    xi1 = xi[:, qk//2:]
    qs = xi0 | (xi1 << 4)

    return qs, d, m, zp

# Write quantized data into binary file
def write_weight_to_file(prefix:str, qs, d, m, zp):
    # Convert to bytes
    qs_data = qs.tobytes()
    d_data = d.tobytes()
    m_data = m.tobytes()
    zp_data = zp.tobytes()
    
    # Write data
    with open(prefix + "/weight_int4.bin", "wb") as f:
        f.write(qs_data)
    with open(prefix + "/scaling_factor_int4.bin", "wb") as f:
        f.write(d_data)
    with open(prefix + "/offset_int4.bin", "wb") as f:
        f.write(m_data)
    with open(prefix + "/zero_point_int4.bin", "wb") as f:
        f.write(zp_data)

    f.close()

# Read quantized data from binary file
def read_weight_from_file(prefix:str):
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
def quantize_model(prefix, method='Q4_0', data_type='fp32'):
    # Check model name
    model_name_size = prefix.split('/')[-1]
    if model_name_size == 'OPT_125m':
        layer_num = 12
    elif model_name_size == 'OPT_1.3B':
        layer_num = 24
    elif model_name_size == 'OPT_6.7B':
        layer_num = 32
    elif model_name_size == 'LLaMA_7B':
        layer_num = 32
    else:
        raise ValueError("Invalid model name. Expected 'OPT_125m', 'OPT_1.3B', 'OPT_6.7B', or 'LLaMA_7B'.")

    # Check quantization method
    if method not in ['Q4_0', 'Q4_1']:
        raise ValueError("Invalid quantization method. Expected 'Q4_0' or 'Q4_1'.")

    # Check data type
    if data_type == 'fp32':
        bytes_per_element = 4
    elif data_type == 'fp16':
        bytes_per_element = 2
    elif data_type == 'int8':
        bytes_per_element = 1
    else:
        raise ValueError("Invalid data type. Expected 'fp32', 'fp16', or 'int8'.")

    print(f"Quantizing {model_name_size} with {method} method... (original data type: {data_type})")

    # Quantize embed_positions
    model_name = model_name_size.split('_')[0]
    # OPT
    if model_name == 'OPT':
        file_path = f"{prefix}/decoder/embed_positions"
        weight_path = f"{file_path}/weight.bin"
        file_size_bytes = os.path.getsize(weight_path)
        if file_size_bytes % bytes_per_element != 0:
            raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        array_size = file_size_bytes // bytes_per_element
        if method == 'Q4_0':
            qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
        elif method == 'Q4_1':
            qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
        write_weight_to_file(file_path, qs, d, m, zp)
        print(f"Quantization of embed_positions finished.")

        # Quantize embed_tokens
        file_path = f"{prefix}/decoder/embed_tokens"
        weight_path = f"{file_path}/weight.bin"
        file_size_bytes = os.path.getsize(weight_path)
        if file_size_bytes % bytes_per_element != 0:
            raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        array_size = file_size_bytes // bytes_per_element
        if method == 'Q4_0':
            qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
        elif method == 'Q4_1':
            qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
        write_weight_to_file(file_path, qs, d, m, zp)
        print(f"Quantization of embed_tokens finished.")

        # Quantize final_layer_norm
        file_path = f"{prefix}/decoder/final_layer_norm"
        weight_path = f"{file_path}/weight.bin"
        file_size_bytes = os.path.getsize(weight_path)
        if file_size_bytes % bytes_per_element != 0:
            raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        array_size = file_size_bytes // bytes_per_element
        if method == 'Q4_0':
            qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
        elif method == 'Q4_1':
            qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
        write_weight_to_file(file_path, qs, d, m, zp)
        print(f"Quantization of final_layer_norm finished.")
        
        # Quantize layers
        for idx in range(layer_num):
            # Quantize fc1
            file_path = f"{prefix}/decoder/layer{idx}/fc1"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize fc2
            file_path = f"{prefix}/decoder/layer{idx}/fc2"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize final_layer_norm
            file_path = f"{prefix}/decoder/layer{idx}/final_layer_norm"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize self_attn/k_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/k_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize self_attn/out_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/out_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize self_attn/q_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/q_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize self_attn/v_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/v_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize self_attn_layer_norm
            file_path = f"{prefix}/decoder/layer{idx}/self_attn_layer_norm"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            print(f"Quantization of layer {idx} finished.")
    
    # LLaMA
    elif model_name == 'LLaMA':
        # Quantize embed_tokens
        file_path = f"{prefix}/decoder/embed_tokens"
        weight_path = f"{file_path}/weight.bin"
        file_size_bytes = os.path.getsize(weight_path)
        if file_size_bytes % bytes_per_element != 0:
            raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
        array_size = file_size_bytes // bytes_per_element
        if method == 'Q4_0':
            qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
        elif method == 'Q4_1':
            qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
        write_weight_to_file(file_path, qs, d, m, zp)
        print(f"Quantization of embed_tokens finished.")

        # Quantize layers
        for idx in range(layer_num):
            # Quantize down_proj
            file_path = f"{prefix}/decoder/layer{idx}/down_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize gate_proj
            file_path = f"{prefix}/decoder/layer{idx}/gate_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize input_layernorm
            file_path = f"{prefix}/decoder/layer{idx}/input_layernorm"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize post_attention_layernorm
            file_path = f"{prefix}/decoder/layer{idx}/post_attention_layernorm"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize self_attn/k_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/k_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize self_attn/o_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/o_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize self_attn/q_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/q_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize self_attn/v_proj
            file_path = f"{prefix}/decoder/layer{idx}/self_attn/v_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            # Quantize up_proj
            file_path = f"{prefix}/decoder/layer{idx}/up_proj"
            weight_path = f"{file_path}/weight.bin"
            file_size_bytes = os.path.getsize(weight_path)
            if file_size_bytes % bytes_per_element != 0:
                raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
            array_size = file_size_bytes // bytes_per_element
            if method == 'Q4_0':
                qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
            elif method == 'Q4_1':
                qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
            write_weight_to_file(file_path, qs, d, m, zp)

            print(f"Quantization of layer {idx} finished.")

    print(f"All the weights of {model_name_size} has been quantized with {method} method.")

# Test function
def test():
    print("Test function starts.")

    prefix = "models/LLaMA_7B"
    method = "Q4_0"
    data_type = "fp32"
    
        # Check model name
    model_name_size = prefix.split('/')[-1]
    if model_name_size not in ['OPT_125m', 'OPT_1.3B', 'OPT_6.7B', 'LLaMA_7B']:
        raise ValueError("Invalid model name. Expected 'OPT_125m', 'OPT_1.3B', 'OPT_6.7B', or 'LLaMA_7B'.")

    # Check quantization method
    if method not in ['Q4_0', 'Q4_1']:
        raise ValueError("Invalid quantization method. Expected 'Q4_0' or 'Q4_1'.")

    # Check data type
    if data_type == 'fp32':
        bytes_per_element = 4
    elif data_type == 'fp16':
        bytes_per_element = 2
    elif data_type == 'int8':
        bytes_per_element = 1
    else:
        raise ValueError("Invalid data type. Expected 'fp32', 'fp16', or 'int8'.")

    print(f"Quantizing {model_name_size} with {method} method... (original data type: {data_type})")

    # Quantize down_proj in layer 0
    file_path = f"{prefix}/decoder/layer0/down_proj"
    weight_path = f"{file_path}/weight.bin"
    file_size_bytes = os.path.getsize(weight_path)
    if file_size_bytes % bytes_per_element != 0:
        raise ValueError(f"Invalid file size of {weight_path}. Expected multiple of element number.")
    array_size = file_size_bytes // bytes_per_element
    if method == 'Q4_0':
        qs, d, m, zp = quantize_row_q4_0(weight_path, array_size, data_type)
    elif method == 'Q4_1':
        qs, d, m, zp = quantize_row_q4_1(weight_path, array_size, data_type)
    write_weight_to_file(file_path, qs, d, m, zp)

    read_qs, read_d, read_m, read_zp = read_weight_from_file(file_path)

    # Check weights
    first_half_qs = np.bitwise_and(qs, 0x0F)
    second_half_qs = np.bitwise_and(qs, 0xF0) >> 4
    first_half_read_qs = np.bitwise_and(np.frombuffer(read_qs, dtype=np.int8), 0x0F)
    second_half_read_qs = np.bitwise_and(np.frombuffer(read_qs, dtype=np.int8), 0xF0) >> 4
    print(f"first_half_qs:       {first_half_qs[0:2, :16]}")
    print(f"first_half_read_qs:  {first_half_read_qs[:32]}")
    print(f"second_half_qs:      {second_half_qs[0:2, :16]}")
    print(f"second_half_read_qs: {second_half_read_qs[:32]}")

    # Check scaling factors
    if STORE_FP16:
        read_d = np.frombuffer(read_d, dtype=np.float16)
    else:
        read_d = np.frombuffer(read_d, dtype=np.float32)
    print(f"d:      {d}")
    print(f"read_d: {read_d}")

    # Check offsets
    if STORE_FP16:
        read_m = np.frombuffer(read_m, dtype=np.float16)
    else:
        read_m = np.frombuffer(read_m, dtype=np.float32)
    print(f"m:      {m}")
    print(f"read_m: {read_m}")

    # Check zero points
    if STORE_FP16:
        read_zp = np.frombuffer(read_zp, dtype=np.float16)
    else:
        read_zp = np.frombuffer(read_zp, dtype=np.float32)
    print(f"zp:      {zp}")
    print(f"read_zp: {read_zp}")

# Main function
def main():
    def get_parser():
        parser = argparse.ArgumentParser(description="Quantize model")
        parser.add_argument("--model_path", type=str, default="models/LLaMA_7B", help="Model path")
        parser.add_argument("--method", type=str, default="Q4_0", help="Quantization method")
        parser.add_argument("--data_type", type=str, default="fp32", help="Data type")
        return parser
    
    parser = get_parser()
    args = parser.parse_args()
    quantize_model(prefix=args.model_path, method=args.method, data_type=args.data_type)

if __name__ == "__main__":
    # main()
    test()