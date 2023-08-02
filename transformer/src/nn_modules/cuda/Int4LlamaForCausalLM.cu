#include <chrono>

#include "Int4LlamaForCausalLM.h"
#include "operators.h"
#include "utils.h"

Int4LlamaForCausalLM::Int4LlamaForCausalLM(std::string param_path, const struct model_config config) {
    allocate_aligned_memory_gpu(logits_output_half, config.max_sqlen * config.vocsize * sizeof(float16_t));
    allocate_aligned_memory_gpu(logits_output, config.max_sqlen * config.vocsize * sizeof(float));
    allocate_aligned_memory_gpu(lm_head_weight, (config.embed_dim * config.vocsize * sizeof(int)) / 8);

    allocate_aligned_memory_gpu(split_8_buffer, config.max_sqlen * config.vocsize * sizeof(float16_t) * 8);

    this->decoder = Int4llamaDecoder(param_path + "/decoder", config);
    this->lm_head = Linear_half_int4(Matrix3D<int>(lm_head_weight, 1, config.vocsize / 8, config.embed_dim),
                                   param_path + "/lm_head");
}

struct Int4LlamaForCausalLM_output Int4LlamaForCausalLM::forward(const struct Int4LlamaForCausalLM_input &input) {
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float milliseconds = 0;

    PROFILE_START(profile_name);
    int sqlen = input.input_ids.m_dim_z;

    struct Int4llamaDecoder_output decoder_output;

    // cudaEventRecord(start);

    if (input.has_past_keys_values) {
        struct Int4llamaDecoder_input decoder_input = {input.input_ids, input.past_keys, input.past_values};
        decoder_output = this->decoder.forward(decoder_input);

    } else {
        struct Int4llamaDecoder_input decoder_input = {input.input_ids};
        decoder_output = this->decoder.forward(decoder_input);
    }

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("The elapsed time of Int4llamaDecoder is %.2f ms\n", milliseconds);


    // cudaEventRecord(start);

    Matrix3D<float16_t> logits_half(logits_output_half, 1, sqlen, this->decoder.voc_size);
    this->lm_head.forward(decoder_output.last_hidden_state, logits_half, split_8_buffer);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("The elapsed time of lm_head.forward is %.2f ms\n", milliseconds);


    // cudaEventRecord(start);

    Matrix3D<float> logits(logits_output, 1, sqlen, this->decoder.voc_size);
    int threadsPerBlock_1D = 1024;
    int blocksPerGrid =(sqlen * this->decoder.voc_size + threadsPerBlock_1D - 1) / threadsPerBlock_1D;
    half2float<<<blocksPerGrid, threadsPerBlock_1D>>>(logits_output_half, logits_output, sqlen * this->decoder.voc_size);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("The elapsed time of half2float is %.2f ms\n", milliseconds);


    // cudaEventRecord(start);

    cudaDeviceSynchronize();

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("The elapsed time of cudaDeviceSynchronize is %.2f ms\n", milliseconds);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    struct Int4LlamaForCausalLM_output LMoutput = {logits, decoder_output.past_keys, decoder_output.past_values};
    PROFILE_END(profile_name);

    return LMoutput;
}
