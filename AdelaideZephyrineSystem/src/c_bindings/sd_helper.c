// sd_helper.c - Helper functions for Ada FFI
// Provides PNG encoding and Base64 conversion for stable-diffusion.cpp
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

// From miniz.h (already in stable-diffusion.cpp build)
extern void *tdefl_write_image_to_png_file_in_memory(const void *pImage,
                                                      int w, int h,
                                                      int num_chans,
                                                      size_t *pLen_out);
extern void mz_free(void *ptr);

// Base64 encoding table
static const char base64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Convert raw image data to Base64-encoded PNG
// Returns malloc'd string, caller must free with sd_free_string()
char* sd_image_to_base64_png(const void* image_data,
                              int width, int height, int channels) {
    // Convert raw pixels to PNG in memory
    size_t png_len = 0;
    void* png_data = tdefl_write_image_to_png_file_in_memory(
        image_data, width, height, channels, &png_len);

    if (!png_data || png_len == 0) {
        return NULL;
    }

    // Calculate base64 output length
    size_t b64_len = 4 * ((png_len + 2) / 3) + 1;
    // NULL check follows
    char* b64_str = (char*)malloc(b64_len);
    if (!b64_str) {
        mz_free(png_data);
        return NULL;
    }

    // Encode to base64
    const uint8_t* src = (const uint8_t*)png_data;
    char* dst = b64_str;
    size_t i;

    for (i = 0; i + 2 < png_len; i += 3) {
        *dst++ = base64_table[(src[i] >> 2) & 0x3F];
        *dst++ = base64_table[((src[i] & 0x3) << 4) | ((src[i+1] >> 4) & 0xF)];
        *dst++ = base64_table[((src[i+1] & 0xF) << 2) | ((src[i+2] >> 6) & 0x3)];
        *dst++ = base64_table[src[i+2] & 0x3F];
    }

    if (i < png_len) {
        *dst++ = base64_table[(src[i] >> 2) & 0x3F];
        if (i + 1 < png_len) {
            *dst++ = base64_table[((src[i] & 0x3) << 4) | ((src[i+1] >> 4) & 0xF)];
            *dst++ = base64_table[(src[i+1] & 0xF) << 2];
        } else {
            *dst++ = base64_table[(src[i] & 0x3) << 4];
            *dst++ = '=';
        }
        *dst++ = '=';
    }

    *dst = '\0';

    mz_free(png_data);
    return b64_str;
}

// Free string allocated by sd_image_to_base64_png
void sd_free_string(char* str) {
    if (str) free(str);
}
