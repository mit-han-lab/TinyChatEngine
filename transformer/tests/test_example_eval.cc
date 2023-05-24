#include "OPTForCausalLM.h"
#include "../input_input_ids.h"
#include "utils.h"
#include <vector>
#include <cstdlib>

#define LM_HEAD_SIZE 50272 * 768
float lm_head_weight[LM_HEAD_SIZE];
float last_token_logits_array[50272];
float gt_last_token_logits_array[50272];
#define pad_len 404

void test_OPTForCausalLM_512sqlen () {
    // const int num_heads = 12, embed_dim = 768, sqlen = 108, b = 1, hidden_dim = 3072, voc_size = 50272, padding_idx = 1, num_layers = 12;
    // Matrix3D<int> matrix_input_ids(input_ids, 1, 1, 512);

    // // Read lm_head
    // struct OPTForCausalLM_input input = {matrix_input_ids};
    // Matrix3D<float> lm_head(lm_head_weight, 1, 50272, 768);
    // OPTForCausalLM model("assets", voc_size, embed_dim, hidden_dim, num_heads,
    //                padding_idx, num_layers);

    // // smoothquant: outputs = model(input_ids)
    // auto output = model.forward(input);

    // // smoothquant: last_token_logits = outputs.logits[:, -2-pad_len, :]
    // Matrix3D<float> last_token_logits(last_token_logits_array, 1, 1, 50272);
    // for (int i = 0; i < 50272; i++){
    //     last_token_logits(0, 0, i) = output.logits(0, (512 - (2 + pad_len)), i);
    // }

    // // End result tests here
    // read_to_array((char*)"assets/tests/causallm/last_token_logits.bin", gt_last_token_logits_array, 50272);
    // bool sucess = check_two_equal(gt_last_token_logits_array, last_token_logits_array, 50272);
    // if (!sucess)
    //     std::cout << "-------- Test of " << __func__ << ": Fail! -------- "<< std::endl;
    // else
    //     std::cout << "-------- Test of " << __func__ << ": Passed! -------- "<< std::endl;
    return;
}


int main() {test_OPTForCausalLM_512sqlen();};