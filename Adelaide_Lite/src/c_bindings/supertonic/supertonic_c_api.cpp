#include "supertonic_c_api.h"
#include "helper.h"
#include <vector>
#include <string>
#include <iostream>

struct SupertonicTTS_ {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "Supertonic"};
    std::unique_ptr<TextToSpeech> tts;
};

struct SupertonicStyle_ {
    std::unique_ptr<Style> style;
};

extern "C" {

SupertonicTTS* supertonic_init(const char* onnx_dir, int use_gpu) {
    auto tts_wrapper = new SupertonicTTS_();
    try {
        tts_wrapper->tts = loadTextToSpeech(tts_wrapper->env, onnx_dir, use_gpu != 0);
        return tts_wrapper;
    } catch (const std::exception& e) {
        std::cerr << "Supertonic init error: " << e.what() << std::endl;
        delete tts_wrapper;
        return nullptr;
    }
}

void supertonic_free(SupertonicTTS* tts) {
    if (tts) {
        delete tts;
    }
}

SupertonicStyle* supertonic_load_style(const char** voice_style_paths, int num_paths) {
    std::vector<std::string> paths;
    for (int i = 0; i < num_paths; ++i) {
        paths.push_back(voice_style_paths[i]);
    }
    auto style_wrapper = new SupertonicStyle_();
    try {
        style_wrapper->style = std::make_unique<Style>(loadVoiceStyle(paths, false));
        return style_wrapper;
    } catch (const std::exception& e) {
        std::cerr << "Supertonic style load error: " << e.what() << std::endl;
        delete style_wrapper;
        return nullptr;
    }
}

void supertonic_free_style(SupertonicStyle* style) {
    if (style) {
        delete style;
    }
}

float* supertonic_synthesize(
    SupertonicTTS* tts,
    const char* text,
    const char* lang,
    SupertonicStyle* style,
    int total_step,
    float speed,
    float silence_duration,
    size_t* out_samples
) {
    if (!tts || !tts->tts || !style || !style->style || !text || !lang || !out_samples) {
        return nullptr;
    }
    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto result = tts->tts->call(
            memory_info,
            text,
            lang,
            *(style->style),
            total_step,
            speed,
            silence_duration
        );
        
        *out_samples = result.wav.size();
        float* audio_data = (float*)malloc(*out_samples * sizeof(float));
        if (audio_data) {
            std::copy(result.wav.begin(), result.wav.end(), audio_data);
        }
        return audio_data;
    } catch (const std::exception& e) {
        std::cerr << "Supertonic synthesis error: " << e.what() << std::endl;
        return nullptr;
    }
}

void supertonic_free_audio(float* audio) {
    if (audio) {
        free(audio);
    }
}

int supertonic_get_sample_rate(SupertonicTTS* tts) {
    if (tts && tts->tts) {
        return tts->tts->getSampleRate();
    }
    return 24000;
}

}
