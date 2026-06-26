#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"
#include <stdexcept>
#include <iostream>
#include <cstring>

extern "C" {
    // ===== llama.cpp safe wrappers =====
    
    // ===== GPU MEMORY QUERY =====
    // Queries GPU device memory (free/total) through ggml backend.
    // Returns free and total memory in bytes.
    // Works for ALL backends: Metal (Apple), CUDA (NVIDIA), OneAPI/SYCL (Intel),
    // Vulkan (cross-platform), ROCm (AMD), NNA (Qualcomm), etc.
    // For CPU-only: returns 0,0 (inapplicable - caller should report "stable").
    void gpu_memory_query(size_t * free_bytes, size_t * total_bytes) {
        if (!free_bytes || !total_bytes) return;
        *free_bytes = 0;
        *total_bytes = 0;
        try {
            // Get the first backend registry (index 0 = system default)
            ggml_backend_reg_t reg = ggml_backend_reg_get(0);
            if (!reg) return;
            size_t n_devices = ggml_backend_reg_dev_count(reg);
            if (n_devices == 0) return;
            // Find the first GPU-type device (not CPU)
            // This covers Metal (Apple), CUDA (NVIDIA), OneAPI/SYCL (Intel),
            // Vulkan (cross-platform), ROCm (AMD), NNA (Qualcomm), etc.
            for (size_t i = 0; i < n_devices; i++) {
                ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
                if (!dev) continue;
                enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                    dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    ggml_backend_dev_memory(dev, free_bytes, total_bytes);
                    return;
                }
            }
        } catch (...) {
            *free_bytes = 0;
            *total_bytes = 0;
        }
    }

    // ===== CPU MEMORY QUERY =====
    // Queries CPU memory (free/total) through macOS host_statistics64 and sysctl.
    // Returns free and total memory in bytes.
    #ifdef __APPLE__
    // [DO NOT REMOVE] Specific XNU Fix for GCC compiling macOS SDK mach headers.
    // Apple's xnu_static_assert macros in <mach/message.h> rely on C11's _Static_assert
    // which causes syntax errors in C++ mode under GCC. We map it to C++'s static_assert.
    #ifndef _Static_assert
    #define _Static_assert(x, y) static_assert(x, y)
    #endif
    #include <sys/types.h>
    #include <sys/sysctl.h>
    #include <mach/mach.h>
    #include <mach/host_info.h>
    #include <mach/mach_host.h>
    void cpu_memory_query(size_t * free_bytes, size_t * total_bytes) {
        if (!free_bytes || !total_bytes) return;
        *free_bytes = 0;
        *total_bytes = 0;
        try {
            // Get total physical memory via sysctl
            int mib[2];
            mib[0] = CTL_HW;
            mib[1] = HW_MEMSIZE;
            uint64_t total_mem = 0;
            size_t len = sizeof(total_mem);
            if (sysctl(mib, 2, &total_mem, &len, NULL, 0) == 0) {
                *total_bytes = (size_t)total_mem;
            }
            // Get free memory via host_statistics64
            mach_port_t host = mach_host_self();
            vm_statistics64_data_t vm_stat;
            mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
            if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
                // free_pages + speculative_pages are available without paging
                uint64_t free_mem = ((uint64_t)vm_stat.free_count + vm_stat.speculative_count) * vm_page_size;
                *free_bytes = (size_t)free_mem;
            }
        } catch (...) {
            *free_bytes = 0;
            *total_bytes = 0;
        }
    }
    #else
    // Non-Apple platforms: return 0,0 (inapplicable)
    void cpu_memory_query(size_t * free_bytes, size_t * total_bytes) {
        if (free_bytes) *free_bytes = 0;
        if (total_bytes) *total_bytes = 0;
    }
    #endif
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

    // ===== mtmd (multimodal) safe wrappers =====
    // These wrap the mtmd C API with exception safety for Ada FFI.
    // Why: The mtmd API is the public interface for multimodal support in llama.cpp.
    //      We need these wrappers because Ada cannot directly call C++ code,
    //      and we want exception safety at the boundary.
    
    // Opaque handle types for Ada - these are just void pointers
    typedef void* mtmd_context_handle;
    typedef void* mtmd_bitmap_handle;
    typedef void* mtmd_input_chunks_handle;
    typedef void* mtmd_input_chunk_handle;

    // Initialize mtmd context from mmproj file
    // Returns nullptr on failure
    mtmd_context_handle mtmd_init_from_file_safe(const char * mmproj_fname,
                                                  const struct llama_model * text_model,
                                                  bool use_gpu,
                                                  int n_threads) {
        try {
            struct mtmd_context_params params = mtmd_context_params_default();
            params.use_gpu = use_gpu;
            params.n_threads = n_threads;
            params.warmup = true;
            return (mtmd_context_handle)mtmd_init_from_file(mmproj_fname, text_model, params);
        } catch (const std::exception& e) {
            std::cerr << "Caught C++ exception in mtmd_init_from_file: " << e.what() << std::endl;
            return nullptr;
        } catch (...) {
            std::cerr << "Caught unknown C++ exception in mtmd_init_from_file" << std::endl;
            return nullptr;
        }
    }

    // Free mtmd context
    void mtmd_free_safe(mtmd_context_handle ctx) {
        if (ctx) {
            try {
                mtmd_free((mtmd_context*)ctx);
            } catch (const std::exception& e) {
                std::cerr << "Caught C++ exception in mtmd_free: " << e.what() << std::endl;
            }
        }
    }

    // Create bitmap from raw RGB pixels
    // data must be nx * ny * 3 bytes in RGBRGBRGB... format
    mtmd_bitmap_handle mtmd_bitmap_init_safe(uint32_t nx, uint32_t ny, const unsigned char * data) {
        try {
            return (mtmd_bitmap_handle)mtmd_bitmap_init(nx, ny, data);
        } catch (const std::exception& e) {
            std::cerr << "Caught C++ exception in mtmd_bitmap_init: " << e.what() << std::endl;
            return nullptr;
        } catch (...) {
            std::cerr << "Caught unknown C++ exception in mtmd_bitmap_init" << std::endl;
            return nullptr;
        }
    }

    // Free bitmap
    void mtmd_bitmap_free_safe(mtmd_bitmap_handle bitmap) {
        if (bitmap) {
            try {
                mtmd_bitmap_free((mtmd_bitmap*)bitmap);
            } catch (const std::exception& e) {
                std::cerr << "Caught C++ exception in mtmd_bitmap_free: " << e.what() << std::endl;
            }
        }
    }

    // Get bitmap dimensions
    uint32_t mtmd_bitmap_get_nx_safe(mtmd_bitmap_handle bitmap) {
        if (!bitmap) return 0;
        try {
            return mtmd_bitmap_get_nx((const mtmd_bitmap*)bitmap);
        } catch (...) { return 0; }
    }

    uint32_t mtmd_bitmap_get_ny_safe(mtmd_bitmap_handle bitmap) {
        if (!bitmap) return 0;
        try {
            return mtmd_bitmap_get_ny((const mtmd_bitmap*)bitmap);
        } catch (...) { return 0; }
    }

    // Initialize empty input chunks list
    mtmd_input_chunks_handle mtmd_input_chunks_init_safe(void) {
        try {
            return (mtmd_input_chunks_handle)mtmd_input_chunks_init();
        } catch (const std::exception& e) {
            std::cerr << "Caught C++ exception in mtmd_input_chunks_init: " << e.what() << std::endl;
            return nullptr;
        } catch (...) {
            std::cerr << "Caught unknown C++ exception in mtmd_input_chunks_init" << std::endl;
            return nullptr;
        }
    }

    // Free input chunks
    void mtmd_input_chunks_free_safe(mtmd_input_chunks_handle chunks) {
        if (chunks) {
            try {
                mtmd_input_chunks_free((mtmd_input_chunks*)chunks);
            } catch (const std::exception& e) {
                std::cerr << "Caught C++ exception in mtmd_input_chunks_free: " << e.what() << std::endl;
            }
        }
    }

    // Get number of chunks
    size_t mtmd_input_chunks_size_safe(mtmd_input_chunks_handle chunks) {
        if (!chunks) return 0;
        try {
            return mtmd_input_chunks_size((const mtmd_input_chunks*)chunks);
        } catch (...) { return 0; }
    }

    // Get chunk type: 0=text, 1=image, 2=audio
    int32_t mtmd_input_chunk_get_type_safe(mtmd_input_chunk_handle chunk) {
        if (!chunk) return -1;
        try {
            return (int32_t)mtmd_input_chunk_get_type((const mtmd_input_chunk*)chunk);
        } catch (...) { return -1; }
    }

    // Get number of tokens in a chunk
    size_t mtmd_input_chunk_get_n_tokens_safe(mtmd_input_chunk_handle chunk) {
        if (!chunk) return 0;
        try {
            return mtmd_input_chunk_get_n_tokens((const mtmd_input_chunk*)chunk);
        } catch (...) { return 0; }
    }

    // Get text tokens from a text chunk
    // Returns pointer to internal token array, n_tokens_output receives count
    // WARNING: Do not free the returned pointer - it's owned by the chunk
    const int32_t * mtmd_input_chunk_get_tokens_text_safe(mtmd_input_chunk_handle chunk, size_t * n_tokens_output) {
        if (!chunk || !n_tokens_output) { *n_tokens_output = 0; return nullptr; }
        try {
            return mtmd_input_chunk_get_tokens_text((const mtmd_input_chunk*)chunk, n_tokens_output);
        } catch (...) { *n_tokens_output = 0; return nullptr; }
    }

    // Encode a chunk (image or audio) - must be called before using embeddings
    int32_t mtmd_encode_chunk_safe(mtmd_context_handle ctx, mtmd_input_chunk_handle chunk) {
        if (!ctx || !chunk) return -1;
        try {
            return mtmd_encode_chunk((mtmd_context*)ctx, (const mtmd_input_chunk*)chunk);
        } catch (const std::exception& e) {
            std::cerr << "Caught C++ exception in mtmd_encode_chunk: " << e.what() << std::endl;
            return -1;
        } catch (...) { return -1; }
    }

    // Get output embeddings after encoding
    // Returns pointer to float array, size = n_embd * n_tokens * sizeof(float)
    float * mtmd_get_output_embd_safe(mtmd_context_handle ctx) {
        if (!ctx) return nullptr;
        try {
            return mtmd_get_output_embd((mtmd_context*)ctx);
        } catch (...) { return nullptr; }
    }

    // Check if model supports vision
    int32_t mtmd_support_vision_safe(mtmd_context_handle ctx) {
        if (!ctx) return 0;
        try {
            return mtmd_support_vision((const mtmd_context*)ctx) ? 1 : 0;
        } catch (...) { return 0; }
    }

    // Check if chunk needs non-causal mask (for image chunks)
    int32_t mtmd_decode_use_non_causal_safe(mtmd_context_handle ctx, mtmd_input_chunk_handle chunk) {
        if (!ctx) return 0;
        try {
            return mtmd_decode_use_non_causal((const mtmd_context*)ctx, (const mtmd_input_chunk*)chunk) ? 1 : 0;
        } catch (...) { return 0; }
    }

    // Get default media marker string
    const char * mtmd_default_marker_safe(void) {
        try {
            return mtmd_default_marker();
        } catch (...) { return "<__media__>"; }
    }

    // ===== NEW: Tokenization and helper wrappers =====

    // Tokenize text prompt + bitmaps into input chunks.
    // The text must contain the media marker (default: "<__media__>").
    // Number of bitmaps must equal number of markers in text.
    // Returns 0 on success, 1 on marker/bitmap count mismatch, 2 on image error.
    int32_t mtmd_tokenize_safe(mtmd_context_handle ctx,
                               mtmd_input_chunks_handle output,
                               const char * text,
                               bool add_special,
                               bool parse_special,
                               const mtmd_bitmap_handle * bitmaps,
                               size_t n_bitmaps) {
        if (!ctx || !output || !text) return -1;
        try {
            struct mtmd_input_text input_text;
            input_text.text = text;
            input_text.add_special = add_special;
            input_text.parse_special = parse_special;
            return mtmd_tokenize((mtmd_context*)ctx,
                                 (mtmd_input_chunks*)output,
                                 &input_text,
                                 (const mtmd_bitmap**)bitmaps,
                                 n_bitmaps);
        } catch (const std::exception& e) {
            std::cerr << "Caught C++ exception in mtmd_tokenize: " << e.what() << std::endl;
            return -1;
        } catch (...) { return -1; }
    }

    // Create bitmap from image buffer (JPEG, PNG, etc.)
    // Uses stb_image internally to decode the image bytes.
    // Returns nullptr on failure.
    mtmd_bitmap_handle mtmd_helper_bitmap_init_from_buf_safe(mtmd_context_handle ctx,
                                                              const unsigned char * buf,
                                                              size_t len) {
        if (!ctx || !buf || len == 0) return nullptr;
        try {
            auto wrapper = mtmd_helper_bitmap_init_from_buf(
                (mtmd_context*)ctx, buf, len, false);
            return (mtmd_bitmap_handle)wrapper.bitmap;
        } catch (const std::exception& e) {
            std::cerr << "Caught C++ exception in mtmd_helper_bitmap_init_from_buf: " << e.what() << std::endl;
            return nullptr;
        } catch (...) { return nullptr; }
    }

    // Get a chunk from the chunks list by index.
    // Returns nullptr if index is out of range.
    // WARNING: The returned pointer is owned by the chunks list - do NOT free it.
    mtmd_input_chunk_handle mtmd_input_chunks_get_safe(mtmd_input_chunks_handle chunks, size_t idx) {
        if (!chunks) return nullptr;
        try {
            return (mtmd_input_chunk_handle)mtmd_input_chunks_get(
                (const mtmd_input_chunks*)chunks, idx);
        } catch (...) { return nullptr; }
    }

    // ===== FLOAT AT ADDRESS =====
    // Read a single float from a raw memory address.
    // Used by reranker to read llama_get_embeddings_seq output.
    float read_float_at_address(const void * addr) {
        if (!addr) return -1.0e9f;
        return *(const float *)addr;
    }
}
