#!/bin/bash

# # E.g., Quantize and export Mistral-7B model
# python tools/mistral_exporter.py --model ../../llm-awq-mistral/quant_cache/mistral-7b-w4-g32-awq-v2.pt --output models/Mistral_7B
# python tools/rotary_emb_exporter.py
# # For x86
# python tools/model_quantizer.py --model_path models/Mistral_7B --method QM_x86
# mkdir Mistral_7B_for_x86
# mkdir Mistral_7B_for_x86/INT4
# mkdir Mistral_7B_for_x86/INT4/models
# mv INT4/models/Mistral_7B Mistral_7B_for_x86/INT4/models
# cd Mistral_7B_for_x86/
# zip -r Mistral_7B_v0.2_Instruct.zip INT4
# cd ..
# # For ARM
# python tools/model_quantizer.py --model_path models/Mistral_7B --method QM_ARM
# mkdir Mistral_7B_for_ARM
# mkdir Mistral_7B_for_ARM/INT4
# mkdir Mistral_7B_for_ARM/INT4/models
# mv INT4/models/Mistral_7B Mistral_7B_for_ARM/INT4/models
# cd Mistral_7B_for_ARM/
# zip -r Mistral_7B_v0.2_Instruct.zip INT4
# cd ..
# # fp32
# mkdir Mistral_7B_FP32
# mkdir Mistral_7B_FP32/models
# mv models/Mistral_7B Mistral_7B_FP32/models
# cd Mistral_7B_FP32/
# zip -r Mistral_7B_v0.2_Instruct.zip models
# cd ..


# E.g., Quantize and export LLaMA3-8B model
python tools/llama3_exporter.py --model ../../llm-awq/quant_cache/llama3-8b-w4-g32-awq-v2.pt --output models/LLaMA_3_8B_Instruct
python tools/rotary_emb_exporter.py
# For ARM
python tools/model_quantizer.py --model_path models/LLaMA_3_8B_Instruct --method QM_ARM
mkdir LLaMA_3_8B_Instruct_for_ARM
mkdir LLaMA_3_8B_Instruct_for_ARM/INT4
mkdir LLaMA_3_8B_Instruct_for_ARM/INT4/models
mv INT4/models/LLaMA_3_8B_Instruct LLaMA_3_8B_Instruct_for_ARM/INT4/models
cd LLaMA_3_8B_Instruct_for_ARM/
zip -r LLaMA_3_8B_Instruct.zip INT4
cd ..
# For x86
python tools/model_quantizer.py --model_path models/LLaMA_3_8B_Instruct --method QM_x86
mkdir LLaMA_3_8B_Instruct_for_x86
mkdir LLaMA_3_8B_Instruct_for_x86/INT4
mkdir LLaMA_3_8B_Instruct_for_x86/INT4/models
mv INT4/models/LLaMA_3_8B_Instruct LLaMA_3_8B_Instruct_for_x86/INT4/models
cd LLaMA_3_8B_Instruct_for_x86/
zip -r LLaMA_3_8B_Instruct.zip INT4
cd ..
# fp32
mkdir LLaMA_3_8B_Instruct_FP32
mkdir LLaMA_3_8B_Instruct_FP32/models
mv models/LLaMA_3_8B_Instruct LLaMA_3_8B_Instruct_FP32/models
cd LLaMA_3_8B_Instruct_FP32/
zip -r LLaMA_3_8B_Instruct.zip models
cd ..
