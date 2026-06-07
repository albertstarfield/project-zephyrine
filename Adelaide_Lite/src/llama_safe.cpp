#include "llama.h"
#include <stdexcept>
#include <iostream>

extern "C" {
    struct llama_model * llama_model_load_from_file_safe(const char * path_model, struct llama_model_params params) {
        try {
            return llama_model_load_from_file(path_model, params);
        } catch (const std::exception& e) {
            std::cerr << "Caught C++ exception in llama_model_load_from_file: " << e.what() << std::endl;
            return nullptr;
        } catch (...) {
            std::cerr << "Caught unknown C++ exception in llama_model_load_from_file" << std::endl;
            return nullptr;
        }
    }

    void llama_batch_add_safe(struct llama_batch * batch, int32_t token, int32_t pos, int32_t seq_id, bool logits) {
        batch->token   [batch->n_tokens] = token;
        batch->pos     [batch->n_tokens] = pos;
        batch->n_seq_id[batch->n_tokens] = 1;
        batch->seq_id  [batch->n_tokens][0] = seq_id;
        batch->logits  [batch->n_tokens] = logits ? 1 : 0;
        batch->n_tokens++;
    }

    void llama_batch_clear_safe(struct llama_batch * batch) {
        batch->n_tokens = 0;
    }
}
