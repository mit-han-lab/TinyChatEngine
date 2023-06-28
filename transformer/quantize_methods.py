import numpy as np
from quantize_constants import STORE_FP16

QK4_0 = 32
QK4_1 = 32
QK4_2 = 32

# Converters
def convert_to_fp16(val):
    return np.float16(val)


# 4-bit Quantization method 0
def quantize_row_q4_0(input_path, k, data_type):
    qk = QK4_0
    assert k % qk == 0
    nb = k // qk

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
    qs = np.zeros((nb, qk // 2), dtype=np.uint8)

    xi = ((x * id_vals[:, np.newaxis]) + 8.5).clip(0, 15).astype(np.uint8)
    # xi0 = xi[:, :qk//2]
    # xi1 = xi[:, qk//2:]
    # qs = xi0 | (xi1 << 4)

    for idx in range(qk // 2):
        qs[:, idx] = xi[:, idx * 2] | (xi[:, idx * 2 + 1] << 4)

    return qs, d, m, zp


# 4-bit Quantization method 1
def quantize_row_q4_1(input_path, k, data_type):
    qk = QK4_1
    assert k % qk == 0
    nb = k // qk

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


# interleaving format
# sequential: (a, b), (c, d), (e, f), (g, h): 32 bit = 4xuint8
# expected layout of inB: (a, e), (b, f), (c, g), (d, h)
# low; (a, 0), (b, 0), (c, 0), (d, 0)
# high: (e, 0), (f, 0), (g, 0), (h, 0)
def quantize_row_q4_2(input_path, k, data_type):
    qk = QK4_2
    assert k % qk == 0
    nb = k // qk

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
    qs = np.zeros((nb, qk // 2), dtype=np.uint8)

    xi = ((x * id_vals[:, np.newaxis]) + 8.5).clip(0, 15).astype(np.uint8)
    # xi0 = xi[:, :qk//2]
    # xi1 = xi[:, qk//2:]
    # qs = xi0 | (xi1 << 4)

    for idx in range(qk // 8):
        qs[:, idx * 4] = xi[:, idx * 8] | (xi[:, idx * 8 + 4] << 4)
        qs[:, idx * 4 + 1] = xi[:, idx * 8 + 1] | (xi[:, idx * 8 + 5] << 4)
        qs[:, idx * 4 + 2] = xi[:, idx * 8 + 2] | (xi[:, idx * 8 + 6] << 4)
        qs[:, idx * 4 + 3] = xi[:, idx * 8 + 3] | (xi[:, idx * 8 + 7] << 4)

    return qs, d, m, zp
