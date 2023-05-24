import os


def export_model(model, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "lm_head.bin"), "wb") as f:
        f.write(model.lm_head._parameters["weight"].cpu().float().numpy().tobytes())
    export_decoder(model.model.decoder, os.path.join(f"{outpath}", "decoder"))


def export_decoder(decoder, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    export_embed_tokens(decoder.embed_tokens, os.path.join(f"{outpath}", "embed_tokens"))
    export_LayerNorm(decoder.final_layer_norm, os.path.join(f"{outpath}", "final_layer_norm"))
    export_embed_tokens(decoder.embed_positions, os.path.join(f"{outpath}", "embed_positions"))
    idx = 0
    for layer in decoder.layers:
        export_decoder_layer(layer, os.path.join(f"{outpath}", f"layer{idx}"))
        idx += 1


def export_embed_tokens(embed_tokens, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(embed_tokens.weight.cpu().float().numpy().tobytes())


def export_BMM_S8T_S8N_F32T(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "alpha.bin"), "wb") as f:
        f.write(op.a.cpu().float().numpy().tobytes())


def export_BMM_S8T_S8N_S8T(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "alpha.bin"), "wb") as f:
        f.write(op.a.cpu().float().numpy().tobytes())


def export_W8A8B8O8Linear(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op.weight.cpu().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "bias.bin"), "wb") as f:
        f.write((op.bias.cpu().float() * (op.b.item() / op.a.item())).round().int().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "bias_int8.bin"), "wb") as f:
        f.write((op.bias.cpu().numpy().tobytes()))
    with open(os.path.join(f"{outpath}", "alpha.bin"), "wb") as f:
        f.write(op.a.cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "beta.bin"), "wb") as f:
        f.write(op.b.cpu().float().numpy().tobytes())


def export_W8A8BFP32OFP32Linear(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op.weight.cpu().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "bias.bin"), "wb") as f:
        f.write(op.bias.cpu().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "alpha.bin"), "wb") as f:
        f.write(op.a.cpu().float().numpy().tobytes())


def export_LayerNorm(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op.weight.cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "bias.bin"), "wb") as f:
        f.write(op.bias.cpu().float().numpy().tobytes())


def export_LayerNormQ(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op.weight.cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "bias.bin"), "wb") as f:
        f.write(op.bias.cpu().float().numpy().tobytes())


def export_attention_params(attn, prefix: str):
    outpath = prefix
    export_BMM_S8T_S8N_F32T(attn.qk_bmm, os.path.join(outpath, "qk_bmm"))
    export_BMM_S8T_S8N_S8T(attn.pv_bmm, os.path.join(outpath, "pv_bmm"))
    export_W8A8B8O8Linear(attn.k_proj, os.path.join(outpath, "k_proj"))
    export_W8A8B8O8Linear(attn.v_proj, os.path.join(outpath, "v_proj"))
    export_W8A8B8O8Linear(attn.q_proj, os.path.join(outpath, "q_proj"))
    export_W8A8BFP32OFP32Linear(attn.out_proj, os.path.join(outpath, "out_proj"))


def export_decoder_layer(layer, prefix: str):
    outpath = prefix
    export_attention_params(layer.self_attn, os.path.join(outpath, "self_attn"))
    export_LayerNormQ(layer.self_attn_layer_norm, os.path.join(outpath, "self_attn_layer_norm"))
    export_W8A8B8O8Linear(layer.fc1, os.path.join(outpath, "fc1"))
    export_W8A8BFP32OFP32Linear(layer.fc2, os.path.join(outpath, "fc2"))
    export_LayerNormQ(layer.final_layer_norm, os.path.join(outpath, "final_layer_norm"))
