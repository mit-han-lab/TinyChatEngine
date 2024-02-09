
#include "Generate.h"
#include "LLaMATokenizer.h"
#include "common.h"
#include "utils.h"
#include <thread>
#include <string>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


struct clip_model_config {
    int image_size = 336;
    int patch_size = 14;
    int num_patches = (image_size / patch_size) * (image_size / patch_size);
    int num_positions = num_patches + 1;
    int projection_dim = 768;
    int mmproj_dim = 4096;
    // float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    // float image_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    float image_mean[3] = {0.48145466f, 0.48145466f, 0.48145466f};
    float image_std[3] = {0.26862954f, 0.26862954f, 0.26862954f};
};

struct llava_image_embed {
    float *embed;
    int n_image_pos;
};

struct clip_image_u8 {
    int nx;
    int ny;
    uint8_t *data = NULL;
    size_t size;
};

struct clip_image_f32 {
    int nx;
    int ny;
    float *data = NULL;
    size_t size;
};

clip_image_u8*  make_clip_image_u8()  { return new clip_image_u8();  }
clip_image_f32* make_clip_image_f32() { return new clip_image_f32(); }
void clip_image_u8_free(clip_image_u8 * img)   { if (img->data) { delete[] img->data; } delete img; }
void clip_image_f32_free(clip_image_f32 * img) { if (img->data) { delete[] img->data; } delete img; }

static struct llava_image_embed* load_image(std::string image, void *clip_model_ptr, bool is_vila);
struct llava_image_embed* llava_image_embed_make_with_filename(clip_model_config *clip_config, void *clip_model_ptr, const char *image_path, bool is_vila);
static bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long *sizeOut);
struct llava_image_embed* llava_image_embed_make_with_bytes(clip_model_config *clip_config, void *clip_model_ptr, const unsigned char *image_bytes, int image_bytes_length, bool is_vila);
bool clip_image_load_from_bytes(const unsigned char *bytes, size_t bytes_length, struct clip_image_u8 *img);
static bool llava_image_embed_make_with_clip_img(clip_model_config *clip_config, void *clip_model_ptr, const clip_image_u8 *img, float **image_embd_out, int *n_img_pos_out, bool is_vila);
static bool encode_image_with_clip(clip_model_config *clip_config, void *clip_model_ptr, const clip_image_u8 *img, float *image_embd, int *n_img_pos, bool is_vila);
bool clip_image_preprocess(clip_model_config *clip_config, void *clip_model_ptr, const clip_image_u8 *img, clip_image_f32 *res, const bool pad2square);


// Function to speak in the background
static void sayInBackground(const std::string& text) {
    std::string command = "./application/sts_utils/speak \"" + text + "\"";
    int result = std::system(command.c_str());
    (void)result;
}

std::string LLaVAGenerate(std::string llama_param_path, void* llama_model_ptr, std::string clip_param_path, void* clip_model_ptr, int model_type, 
                          std::string text, std::string img_path, const struct opt_params generation_config, std::string voc_path, bool interactive, 
                          bool voicechat, bool is_vila) {
    std::vector<int> last_n_tokens(generation_config.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    std::vector<int> embd;
    std::vector<int> generate_ids;

    // Tokenize first-part text
    const int max_token = 2048;
    std::vector<int> input_ids(max_token);
    llama_vocab vocab = llama_init_vocab(voc_path.c_str());
    const int n = llama_tokenize(vocab, text.c_str(), input_ids.data(), input_ids.size(), true);
    input_ids.resize(n);

    bool is_codellama = false;
    if (llama_param_path.find("CodeLLaMA") != std::string::npos) {
        is_codellama = true;
    }

    int n_consumed = 0;
    while ((int)input_ids.size() > n_consumed) {
        embd.push_back(input_ids[n_consumed]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(input_ids[n_consumed]);
        ++n_consumed;

        if ((int)embd.size() >= generation_config.n_batch) {
            break;
        }
    }

    bool previous_two_hash = false;
    int break_cnt = 2;
    bool new_prompt = true;
    static bool first_prompt = true;
    static bool has_past_kv = false;
    static std::vector<Matrix3D<float>> past_keys, past_values;
    int n_remain = generation_config.n_predict;
    std::string output;
    while (n_remain != 0 && break_cnt) {
        std::vector<float> logits(generation_config.n_vocab);

        int sqlen = 1;
        if (new_prompt) {
            sqlen = input_ids.size();
        }
        if (model_type == LLaVA_INT4 || model_type == VILA_INT4) {
            Int4LlamaForCausalLM *model = static_cast<Int4LlamaForCausalLM *>(llama_model_ptr);
            struct Int4LlamaForCausalLM_output model_output;
            struct Int4LlamaForCausalLM_input model_input;
            if (has_past_kv) {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                model_input = {input_ids_mat, past_keys, past_values};
            } else {
                // Load and preprocess image
                auto image_embed = load_image(img_path, clip_model_ptr, is_vila);
                sqlen = input_ids.size() + 576;
                int first_sqlen = input_ids.size();
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, first_sqlen);
                Matrix3D<float> image_embed_mat(image_embed->embed, 1, 576, 4096);
                model_input = {input_ids_mat, image_embed_mat};
            }
            if (!new_prompt) STATS_START("Inference latency");
            model_output = model->forward(llama_param_path, model_input);
            if (!new_prompt) STATS_END("Inference latency");
            past_keys = model_output.past_keys;
            past_values = model_output.past_values;
            // memcpy model_ouput.logits[-1] to logits
            memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
                   generation_config.n_vocab * sizeof(float));
        } else if (model_type == LLaVA_FP32 || model_type == VILA_FP32) {
            Fp32LlamaForCausalLM *model = static_cast<Fp32LlamaForCausalLM *>(llama_model_ptr);
            struct Fp32LlamaForCausalLM_output model_output;
            struct Fp32LlamaForCausalLM_input model_input;
            if (has_past_kv) {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                model_input = {input_ids_mat, past_keys, past_values};
            } else {
                auto image_embed = load_image(img_path, clip_model_ptr, is_vila);
                sqlen = input_ids.size() + 576;
                int first_sqlen = input_ids.size();
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, first_sqlen);
                Matrix3D<float> image_embed_mat(image_embed->embed, 1, 576, 4096);
                model_input = {input_ids_mat, image_embed_mat};
            }
            if (!new_prompt) STATS_START("Inference latency");
            model_output = model->forward(model_input);
            if (!new_prompt) STATS_END("Inference latency");
            past_keys = model_output.past_keys;
            past_values = model_output.past_values;
            // memcpy model_ouput.logits[-1] to logits
            memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
                   generation_config.n_vocab * sizeof(float));
        }
        has_past_kv = true;

        if (first_prompt) {
            break;
        }

        // Generate
        const int n_ctx = generation_config.n_ctx;
        const float temp = generation_config.temp;
        const int32_t top_k = generation_config.top_k <= 0 ? generation_config.n_vocab : generation_config.top_k;
        const float top_p = generation_config.top_p;
        const float tfs_z = generation_config.tfs_z;
        const float typical_p = generation_config.typical_p;
        const int32_t repeat_last_n = generation_config.repeat_last_n < 0 ? n_ctx : generation_config.repeat_last_n;
        const float repeat_penalty = generation_config.repeat_penalty;
        const float alpha_presence = generation_config.presence_penalty;
        const float alpha_frequency = generation_config.frequency_penalty;
        const int mirostat = generation_config.mirostat;
        const float mirostat_tau = generation_config.mirostat_tau;
        const float mirostat_eta = generation_config.mirostat_eta;
        const int n_vocab = generation_config.n_vocab;

        std::vector<OPT_token_data> candidates;
        candidates.reserve(n_vocab);
        for (int token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(OPT_token_data{token_id, logits[token_id], 0.0f});
        }

        OPT_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

        // Apply penalties
        auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
        sample_repetition_penalty(&candidates_p, last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                  last_n_repeat, repeat_penalty);
        sample_frequency_and_presence_penalties(&candidates_p,
                                                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                last_n_repeat, alpha_frequency, alpha_presence);

        int id = 0;
        if (temp <= 0) {
            id = sample_token_greedy(&candidates_p);
        } else {
            if (mirostat == 1) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                const int mirostat_m = 100;
                sample_temperature(&candidates_p, temp);
                id =
                    sample_token_mirostat(n_vocab, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
            } else if (mirostat == 2) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                sample_temperature(&candidates_p, temp);
                id = sample_token_mirostat_v2(&candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
            } else {
                // Temperature sampling
                sample_top_k(&candidates_p, top_k, 1);
                sample_tail_free(&candidates_p, tfs_z, 1);
                sample_typical(&candidates_p, typical_p, 1);
                sample_top_p(&candidates_p, top_p, 1);
                sample_temperature(&candidates_p, temp);
                id = sample_token(&candidates_p);
            }
        }

        if (id == 2) {
            break_cnt--;
            continue;
        }  // eos
        else if (id == 1)
            continue;
        break_cnt = 2;

        bool skip = false;
        if (id == 2277 && !previous_two_hash) {
            previous_two_hash = true;
            skip = true;
        } else if (previous_two_hash && id == 29937) {  // token = #
            break_cnt = 0;
            skip = true;
        } else {
            if (previous_two_hash) std::cout << "##" << std::endl;
            previous_two_hash = false;
        }

        if (is_codellama && new_prompt) {
            new_prompt = false;
            // continue;
        }

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
        embd.push_back(id);
        generate_ids.push_back(id);
        input_ids = std::vector<int>{id};

        if (interactive && !skip) {
            output += llama_id_to_token(vocab, id);
            std::cout << llama_id_to_token(vocab, id) << std::flush;
            if (voicechat) {
                // Remove quotes
                output.erase(std::remove(output.begin(), output.end(), '\"'), output.end());
                // Remove hashtags
                output.erase(std::remove(output.begin(), output.end(), '#'), output.end());
                // Remove dashes
                std::replace(output.begin(), output.end(), '-', ' ');

                size_t lastPos;
                // starts ealier but slows down dictation
                bool ended = false;
                if (output.find(", ") != std::string::npos){
                    lastPos = output.rfind(',');
                    ended = true;
                }
                if (output.find("\n") != std::string::npos){
                    lastPos = output.rfind('\n');
                    ended = true;
                }
                else if (output.find(". ") != std::string::npos){
                    lastPos = output.rfind('.');
                    ended = true;
                }
                else if (output.find("! ") != std::string::npos){
                    lastPos = output.rfind('!');
                    ended = true;
                }
                else if (output.find("? ") != std::string::npos){
                    lastPos = output.rfind('?');
                    ended = true;
    
                }
                else if (output.find(": ") != std::string::npos){
                    lastPos = output.rfind(':');
                    ended = true;
                }
                if (ended){
                    // Extract sentence 1 (up to and including the last period)
                    std::string output_copy = output.substr(0, lastPos + 1);
                    // Extract beginning of sentence 2 (excluding the space after the last period)
                    output = output.substr(lastPos + 1); // Skip the last period and space
                    std::thread sayThread(sayInBackground, output_copy);
                    sayThread.detach(); 
                } 
            } 
        }

        new_prompt = false;
        --n_remain;
    }
    if (voicechat && interactive){
        sayInBackground(output);
    }

    if (interactive && !first_prompt) {
        std::cout << std::endl;
    }
    first_prompt = false;

    Profiler::getInstance().report_internal();
    Profiler::getInstance().reset();
    return output;
}


/*
The codes below for image preprocessing are adapted from llama.cpp:
https://github.com/ggerganov/llama.cpp
*/
static struct llava_image_embed* load_image(std::string image, void *clip_model_ptr, bool is_vila) {
    // load and preprocess the image
    llava_image_embed *embed = NULL;
    clip_model_config *clip_config = new clip_model_config();
    embed = llava_image_embed_make_with_filename(clip_config, clip_model_ptr, image.c_str(), is_vila);
    if (!embed) {
        fprintf(stderr, "%s: is %s really an image file?\n", __func__, image.c_str());
        return NULL;
    }

    return embed;
}

struct llava_image_embed * llava_image_embed_make_with_filename(clip_model_config *clip_config, void *clip_model_ptr, const char *image_path, bool is_vila) {
    unsigned char *image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(image_path, &image_bytes, &image_bytes_length);
    if (!loaded) {
        fprintf(stderr, "%s: failed to load %s\n", __func__, image_path);
        return NULL;
    }

    auto embed = llava_image_embed_make_with_bytes(clip_config, clip_model_ptr, image_bytes, image_bytes_length, is_vila);
    free(image_bytes);

    return embed;
}

static bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long *sizeOut) {
    auto file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char *)malloc(fileSize); // Allocate memory to hold the file data
    if (buffer == NULL) {
        fprintf(stderr, "%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        fclose(file);
        return false;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file); // Read the file into the buffer
    if (ferror(file)) {
        fprintf(stderr, "%s: read error: %s\n", __func__, strerror(errno));
        fclose(file);
        return false;
    }
    if (ret != (size_t) fileSize) {
        fprintf(stderr, "%s: unexpectedly reached end of file\n", __func__);
        fclose(file);
        return false;
    }
    fclose(file); // Close the file

    *bytesOut = buffer;
    *sizeOut = fileSize;
    return true;
}

struct llava_image_embed* llava_image_embed_make_with_bytes(clip_model_config *clip_config, void *clip_model_ptr, const unsigned char *image_bytes, int image_bytes_length, bool is_vila) {
    clip_image_u8 *img = make_clip_image_u8();
    if (!clip_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        clip_image_u8_free(img);
        fprintf(stderr, "%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }

    float* image_embed = NULL;
    int n_image_pos = 0;
    bool image_embed_result = llava_image_embed_make_with_clip_img(clip_config, clip_model_ptr, img, &image_embed, &n_image_pos, is_vila);
    if (!image_embed_result) {
        clip_image_u8_free(img);
        fprintf(stderr, "%s: coulnd't embed the image\n", __func__);
        return NULL;
    }

    clip_image_u8_free(img);
    auto result = (llava_image_embed*)malloc(sizeof(llava_image_embed));
    result->embed = image_embed;
    result->n_image_pos = n_image_pos;
    return result;
}

static void build_clip_img_from_data(const stbi_uc *data, int nx, int ny, clip_image_u8 *img) {
    img->nx = nx;
    img->ny = ny;
    img->size = nx * ny * 3;
    img->data = new uint8_t[img->size]();
    memcpy(img->data, data, img->size);
}

bool clip_image_load_from_bytes(const unsigned char *bytes, size_t bytes_length, struct clip_image_u8 *img) {
    int nx, ny, nc;
    auto data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to decode image bytes\n", __func__);
        return false;
    }
    build_clip_img_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

size_t clip_embd_nbytes(clip_model_config *clip_config) {
    return clip_config->num_patches * clip_config->mmproj_dim * sizeof(float);
}

static bool llava_image_embed_make_with_clip_img(clip_model_config *clip_config, void *clip_model_ptr, const clip_image_u8 *img, float **image_embd_out, int *n_img_pos_out, bool is_vila) {
    float *image_embd = (float *)malloc(clip_embd_nbytes(clip_config));
    if (!image_embd) {
        fprintf(stderr, "Unable to allocate memory for image embeddings\n");
        free(image_embd);
        return false;
    }

    int n_img_pos;
    if (!encode_image_with_clip(clip_config, clip_model_ptr, img, image_embd, &n_img_pos, is_vila)) {
        fprintf(stderr, "%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    *image_embd_out = image_embd;
    *n_img_pos_out = n_img_pos;

    return true;
}

static bool encode_image_with_clip(clip_model_config *clip_config, void *clip_model_ptr, const clip_image_u8 *img, float *image_embd, int *n_img_pos, bool is_vila) {
    clip_image_f32 *img_res = make_clip_image_f32();
    if (!clip_image_preprocess(clip_config, clip_model_ptr, img, img_res, /*pad2square =*/ true)) {
        fprintf(stderr, "%s: unable to preprocess image\n", __func__);
        clip_image_f32_free(img_res);
        return false;
    }

    Fp32CLIPVisionTransformer *clip_model = static_cast<Fp32CLIPVisionTransformer *>(clip_model_ptr);
    struct Fp32CLIPVisionTransformer_input model_input;
    struct Fp32CLIPVisionTransformer_output model_output;
    Matrix3D<float> input_image(img_res->data, 3, img_res->nx, img_res->ny);
    model_input = {input_image};
    model_output = clip_model->forward(model_input, is_vila);
    memcpy(image_embd, model_output.last_hidden_state.m_data, clip_embd_nbytes(clip_config));

    clip_image_f32_free(img_res);

    return true;
}

// normalize: x = (x - mean) / std
// TODO: implement bicubic interpolation instead of linear.
bool clip_image_preprocess(clip_model_config *clip_config, void *clip_model_ptr, const clip_image_u8 *img, clip_image_f32 *res, const bool pad2square) {
    // the logic below is to pad the shorter side to the longer side with a background color: rgb(122, 116, 104)
    // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156

    clip_image_u8 *temp = make_clip_image_u8(); // we will keep the input image data here temporarily
    if (pad2square && img->nx != img->ny) {
        int longer_side = std::max(img->nx, img->ny);
        temp->nx = longer_side;
        temp->ny = longer_side;
        temp->size = 3 * longer_side * longer_side;
        temp->data = new uint8_t[temp->size]();
        uint8_t bc[3] = {122, 116, 104}; // bakground color in RGB from LLaVA

        // fill with background color
        for (size_t i = 0; i < temp->size; i++) {
            temp->data[i] = bc[i % 3];
        }

        // copy from the input image
        for (int y = 0; y < img->ny; y++) {
            for (int x = 0; x < img->nx; x++) {
                const int i = 3 * (y * img->nx + x);
                const int j = 3 * (y * temp->nx + x);
                temp->data[j] = img->data[i];
                temp->data[j+1] = img->data[i+1];
                temp->data[j+2] = img->data[i+2];
            }
        }
    } else {
        temp->nx   = img->nx;
        temp->ny   = img->ny;
        temp->size = img->size;
        temp->data = new uint8_t[temp->size]();
        memcpy(&temp->data[0], &img->data[0], temp->size); // copy
    }

    const int nx = temp->nx;
    const int ny = temp->ny;

    const int nx2 = clip_config->image_size;
    const int ny2 = clip_config->image_size;

    res->nx = nx2;
    res->ny = ny2;
    res->size = 3 * nx2 * ny2;
    res->data = new float[res->size]();

    const float scale = std::max(nx, ny) / (float)clip_config->image_size;

    const int nx3 = int(nx / scale + 0.5f);
    const int ny3 = int(ny / scale + 0.5f);

    const auto &m3 = clip_config->image_mean; // {0.48145466f, 0.4578275f, 0.40821073f};
    const auto &s3 = clip_config->image_std;  // {0.26862954f, 0.26130258f, 0.27577711f};

    for (int y = 0; y < ny3; y++) {
        for (int x = 0; x < nx3; x++) {
            for (int c = 0; c < 3; c++) {
                // linear interpolation
                const float sx = (x + 0.5f) * scale - 0.5f;
                const float sy = (y + 0.5f) * scale - 0.5f;

                const int x0 = std::max(0, (int)std::floor(sx));
                const int y0 = std::max(0, (int)std::floor(sy));

                const int x1 = std::min(x0 + 1, nx - 1);
                const int y1 = std::min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3 * (y0 * nx + x0) + c;
                const int j01 = 3 * (y0 * nx + x1) + c;
                const int j10 = 3 * (y1 * nx + x0) + c;
                const int j11 = 3 * (y1 * nx + x1) + c;

                const float v00 = temp->data[j00];
                const float v01 = temp->data[j01];
                const float v10 = temp->data[j10];
                const float v11 = temp->data[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                const int i = 3 * (y * nx3 + x) + c;

                res->data[i] = ((float(v2) / 255.0f) - m3[c]) / s3[c];
            }
        }
    }
    clip_image_u8_free(temp);

    return true;
}
