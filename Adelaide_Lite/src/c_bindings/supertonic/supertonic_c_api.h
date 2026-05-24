#ifndef SUPERTONIC_C_API_H
#define SUPERTONIC_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SupertonicTTS_ SupertonicTTS;
typedef struct SupertonicStyle_ SupertonicStyle;

SupertonicTTS* supertonic_init(const char* onnx_dir, int use_gpu);
void supertonic_free(SupertonicTTS* tts);

SupertonicStyle* supertonic_load_style(const char** voice_style_paths, int num_paths);
void supertonic_free_style(SupertonicStyle* style);

float* supertonic_synthesize(
    SupertonicTTS* tts,
    const char* text,
    const char* lang,
    SupertonicStyle* style,
    int total_step,
    float speed,
    float silence_duration,
    size_t* out_samples
);

void supertonic_free_audio(float* audio);

int supertonic_get_sample_rate(SupertonicTTS* tts);

#ifdef __cplusplus
}
#endif

#endif // SUPERTONIC_C_API_H
