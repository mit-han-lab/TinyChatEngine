#!/bin/bash

# Copy from layer 0 to layer 31
for i in {0..31}; do
  cp -r INT4/models/CodeLLaMA_7B_Instruct/decoder/layer${i}/self_attn/rotary_emb/* INT4/models/Mistral_7B/decoder/layer${i}/self_attn/rotary_emb/
done