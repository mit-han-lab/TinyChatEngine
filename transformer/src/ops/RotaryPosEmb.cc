#include <cmath>
#include "operators.h"

// TODO: optimize this with multithreading
void RotaryPosEmb::forward(Matrix3D<float> &key, Matrix3D<float> &value, int start_idx, int len) {
    PROFILE_START(profile_name);
    int num_heads = key.m_dim_x;
    int head_embed = cos.m_dim_z;
    int max_sqlen = cos.m_dim_y;

    assert(key.m_dim_z == cos.m_dim_z);
    assert(value.m_dim_z == cos.m_dim_z);
    assert(max_sqlen > len + start_idx);

    // cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    // query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    // cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    // sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    // q_embed = (q * cos) + (rotate_half(q) * sin)
    // k_embed = (k * cos) + (rotate_half(k) * sin)
    // rotate_half: torch.cat((-x2, x1), dim=-1)

    // first half
    for (int b = 0; b < num_heads; b++){
        for(int i = 0; i < len; i++) {
            for (int j = 0; j < head_embed / 2; j++){
                key(b, i, j) = (key(b, i, j) * cos(0, i + start_idx, j)) - (key(b, i, j+64) * sin(0, i + start_idx, j));
                value(b, i, j) = (value(b, i, j) * cos(0, i + start_idx, j)) - (value(b, i, j+64) * sin(0, i + start_idx, j));
            }
        }
    }

    // second half
    for (int b = 0; b < num_heads; b++){
        for(int i = 0; i < len; i++) {
            for (int j = head_embed / 2; j < head_embed; j++){
                key(b, i, j) = (key(b, i, j) * cos(0, i + start_idx, j)) + (key(b, i, j-64) * sin(0, i + start_idx, j));
                value(b, i, j) = (value(b, i, j) * value(0, i + start_idx, j)) + (value(b, i, j-64) * sin(0, i + start_idx, j));
            }
        }
    }

    PROFILE_END(profile_name);
}