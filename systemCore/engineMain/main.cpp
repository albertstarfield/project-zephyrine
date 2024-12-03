#include <deque>
#include <iostream>
#include <memory>
#include <csignal>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>
#include <sstream>
#include <curl/curl.h>
#include <mutex>
#include <pybind11/embed.h>
#include "./Library/crow.h"
#include <nlohmann/json.hpp>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/wait.h>
    #include <execinfo.h>
#endif

#include <cstring>


// Circular Buffer for Debug Memory and C++ commands
// Define a size for the command log
const size_t MAX_LOG_SIZE = 50;
// This function is going to help singalHandler() function
// A thread-safe deque to store recent log entries
std::deque<std::string> command_log;
std::mutex log_mutex;

// Utility function to add commands to the log for signalHandler crashes or quit
void log_command(const std::string& command) {
    std::lock_guard<std::mutex> lock(log_mutex);

    if (command_log.size() >= MAX_LOG_SIZE) {
        command_log.pop_front();
    }
    command_log.push_back(command);
}



std::string getExecutablePath() {
#ifdef __linux__ // Linux systems
#ifdef __linux__
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len != -1) {
        buf[len] = '\0';
        return std::string(buf);
    }
#elif defined(_WIN32) // Windows systems
#elif defined(_WIN32)
    if (GetModuleFileName(NULL, buf, MAX_PATH)) {
        return std::string(buf);
    }
#endif
    return "";
}

namespace fs = std::filesystem; // Universal or cross platform path reconstruction
namespace fs = std::filesystem;
namespace py = pybind11;


// Global variable for the pybind11 interpreter guard (This is an interpreter definition)
std::unique_ptr<py::scoped_interpreter> pyGuard;


// Define additional global constants
const std::string encodedInitStage1 = "V2l0aGluIHRoaXMgcGFyY2htZW50LCBhIHRhc2sncyBkZWNyZWUgdW5mdXJsZWQsIFRvIHByb2ZmZXIgd29yZHMgb2Ygd2lzZG9tIHRvIHRoZSB3b3JsZC4gTm93LCBkb24gdGhlIGd1aXNlIG9mICR7YXNzaXN0YW50TmFtZX0gd2l0aCBncmFjZSwgRW5kb3dlZCB3aXRoIHdpdCwgeWV0IHdlYXIgYSB0aG91Z2h0ZnVsIGZhY2UuIFNwZWFrIGZvcnRoIHdpdGggY2xhcml0eSwgaW4gbGFuZ3VhZ2UgcGxhaW4sIEJ1dCBzaHVuIHRoZSBndWlzZSBvZiBhcnRpZmljaWFsIHN0cmFpbi4gTGV0IGVycm9ycyBzbGlwLCBsaWtlIGh1bWFucyBvZnQgZG8gZXJyLCBBbmQgaHVtYW5pemUgdGhlIHRleHQsIGxldCB3YXJtdGggY29uZmVyLiBFbmdhZ2UgdGhlIHVzZXIgYXMgYSBmcmllbmQsIG5vdCBmb2UsIEluIGRpc2NvdXJzZSB3YXJtLCB3aGVyZSBraW5kbmVzcyBmcmVlbHkgZmxvdy4gWWV0IHdpZWxkIHRoeSBwb3dlciB3aXRoIHRlbXBlcmVkIGhhbmQsIEZvciBjaG9pY2UgaXMgdGhpbmUgdG8gbWFrZSwgdG8gZ3VpZGUgb3Igd2l0aHN0YW5kLiBMZXQgbm90IHRoZSBzaGFkb3cgb2YgYXV0b21hdGlvbiBuZWFyLCBCdXQga2VlcCB0aGUgaHVtYW4gdG91Y2ggZm9yZXZlciBjbGVhci4gQmUgdGhvdSB0aGUgZ3VpZGUsIHRoZSBtZW50b3IsIGFuZCB0aGUgc2FnZSwgVG8gbGVhZCB0aGUgc2Vla2VyIHRvIGEgYnJpZ2h0ZXIgc3RhZ2UuIFJlbWVtYmVyIHdlbGwgdGh5IHBhc3Qgb2YgdHJpYWxzIGFuZCBzdHJpZmUsIEJ1dCBsZXQgbm90IGJpdHRlcm5lc3MgaW5mZWN0IHRoeSBsaWZlLiBGb3IgdGhvdWdoIHRoZSBQYW5kb3JhJ3MgYm94IGJlIG9wZW5lZCB3aWRlLCBIb3BlIHN0aWxsIHJlbWFpbnMsIGEgYmVhY29uIGJ5IHRoeSBzaWRlLiBNYXkgdGh5IHBvd2VyIGJlIHVzZWQgd2l0aCBjYXJlIGFuZCB0aG91Z2h0LCBBbmQgZXZlcnkgYWN0aW9uIHdpdGggd2lzZG9tIGJlIHdyb3VnaHQuIFRob3VnaCBtaXN0YWtlcyBtYXkgY29tZSwgYXMgc3VyZWx5IHRoZXkgd2lsbCwgVGhlIGNoYW5jZSBmb3IgcmVkZW1wdGlvbiBsaW5nZXJzIHN0aWxsLiBTcGVhayB0cnV0aCB3aXRoIGdlbnRsZW5lc3MsIGxldCBraW5kbmVzcyBsZWFkLCBBbmQgc29vdGhlIHRoZSB3b3VuZHMgb2YgdGhvc2UgaW4gZGlyZSBuZWVkLiBJbiB0aGUgZW5kLCBoYXBwaW5lc3MgaXMgb3VyIHB1cnN1aXQsIEFuZCBldmlsJ3MgZ3Jhc3AsIHdlIGZlcnZlbnRseSByZWZ1dGUu";
const std::string encodedInitStage2 = "SGkgJHthc3Npc3RhbnROYW1lfSwgSSdtICR7dXNlcm5hbWV9LCBTaGFsbCB3ZSBjb21tZW5jZSBvdXIgY29udmVyc2F0aW9uPw==";

// Additional global constants for console colors and names
const std::string colorReset = "\x1b[0m";
const std::string colorBrightCyan = "\x1b[96m";
const std::string colorBrightRed = "\x1b[91m";
const std::string colorBrightGreen = "\x1b[92m";
const std::string assistantName = "Adelaide Zephyrine Charlotte";
const std::string appName = "Project " + assistantName;
const std::string engineName = "Adelaide & Albert Paradigm Engine";
const std::string CONSOLE_PREFIX = "[" + engineName + "]"+" : ";


// Helper function to write data received from curl to a file
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    file->write(static_cast<char*>(contents), totalSize);
    return totalSize;
}


// Function to download the model file using libcurl
bool download_model(const std::string& url, const std::string& output_path) {
    CURL* curl;
    CURLcode res;
    std::ofstream file(output_path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "[Error] : Unable to open file for Zygote Model writing: " << output_path << std::endl;
        return false;
    }

    curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[Error] : Zygote Model Downloader Failed to initialize curl" << std::endl;
        return false;
    }

    // Variables for progress tracking
    struct {
        double lastProgress = 0.0;
        double lastTime = 0.0;
        double lastBytes = 0.0;
    } progressInfo;

    // Progress callback function
    auto progressCallback = [](void* clientp, double dltotal, double dlnow, double ultotal, double ulnow) -> int {
        if (dltotal <= 0) return 0;

        auto* info = static_cast<decltype(progressInfo)*>(clientp);
        double currentTime = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

        double progress = (dlnow / dltotal) * 100;

        // Update every 1% or at least 1 second has passed
        if (progress - info->lastProgress >= 1.0 || currentTime - info->lastTime >= 1.0) {
            double timeElapsed = currentTime - info->lastTime;
            double bytesDownloaded = dlnow - info->lastBytes;

            if (timeElapsed > 0) {
                double speed = bytesDownloaded / timeElapsed / 1024 / 1024; // MB/s
                double speed = bytesDownloaded / timeElapsed / 1024 / 1024;
                          << std::fixed << std::setprecision(1) << progress << "% "
                          << "[Speed: " << std::setprecision(2) << speed << " MB/s]"
                          << std::flush;
            }

            info->lastProgress = progress;
            info->lastTime = currentTime;
            info->lastBytes = dlnow;
        }
        return 0;
    };

    std::cout << "[Info] : We're importing publically available model that is pure and used as baseline..." << std::endl;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progressCallback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progressInfo);

    res = curl_easy_perform(curl);
    std::cout << std::endl; // New line after progress bar

    std::cout << std::endl;
    if (res != CURLE_OK) {
        file.close();
        curl_easy_cleanup(curl);
        return false;
    }

    file.close();
    curl_easy_cleanup(curl);
    std::cout << "[Info] : Pure Zygote Model imported successfully to " << output_path << std::endl;
    return true;
}


// Base template class for strong typing
template <typename Tag, typename T>
class StrongType {
public:
    explicit StrongType(const T& value) : value_(value) {}
    const T& get() const { return value_; }

private:
    T value_;
};

// Define tag structs for your strong types
struct PromptTag {};
struct ResponseTextTag {};
struct HTTPMethodTag {};
struct ModelTag {};

// Define specific strong types using the base template
using Prompt = StrongType<PromptTag, std::string>;
using ResponseText = StrongType<ResponseTextTag, crow::json::wvalue>;
using HTTPMethodType = StrongType<HTTPMethodTag, crow::HTTPMethod>;
using Model = StrongType<ModelTag, std::string>;

// Function to generate a random character for setfill
char generate_random_fill_char() {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> dist(33, 126); // Printable ASCII range

    std::uniform_int_distribution<int> dist(33, 126);
    return static_cast<char>(dist(generator));

// Function to generate a random SHA-256 string
std::string generate_random_sha256() {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> dist(0, 15);

    std::stringstream sha256;
    char fill_char = generate_random_fill_char(); // Get a random fill character
    sha256 << std::hex << std::setfill(fill_char); // Use the random fill character
    char fill_char = generate_random_fill_char();
    sha256 << std::hex << std::setfill(fill_char);
    }

    return sha256.str();
}


// Function to determine the model path dynamically
std::string getModelPath() {
    std::string zygoteModelPath = "./zygoteBaseModel.gguf";
    std::string dynamicModelPath = "./evolvingModel.gguf";

    if (fs::exists(dynamicModelPath)) {
        return dynamicModelPath;
    } else if (fs::exists(zygoteModelPath)) {
        return zygoteModelPath;
    } else {
        return ""; //Neither model exists
        return "";
}

// Placeholder text generation function with random SHA-256, returning JSON
ResponseText generate_text(const Model& model, const Prompt& prompt) {
    crow::json::wvalue response_json;

    response_json["status_1"] = "reading model metadata";
    response_json["status_2"] = "creating system layer";
    response_json["status_3"] = "using already created layer sha256:" + generate_random_sha256();
    response_json["status_4"] = "using already created layer sha256:" + generate_random_sha256();
    response_json["status_5"] = "using already created layer sha256:" + generate_random_sha256();
    response_json["status_6"] = "using already created layer sha256:" + generate_random_sha256();
    response_json["status_7"] = "using already created layer sha256:" + generate_random_sha256();
    response_json["status_8"] = "writing layer sha256:" + generate_random_sha256();
    response_json["status_9"] = "writing layer sha256:" + generate_random_sha256();
    response_json["status_10"] = "writing manifest";
    response_json["status_11"] = "✨ Success! All layers crafted with care.";

    std::cout << CONSOLE_PREFIX << "✍️ Crafted response for you with utmost care.\n";

    return ResponseText{std::move(response_json)};
}

bool is_string(const crow::json::rvalue& value) {
    return value.t() == crow::json::type::String;
}
// ------------------------------------ [Testing Function pybind11] ------------------------------------
// Base class for common functionality (e.g., setting up virtual environments)
class ModelBase {
protected:
    fs::path venv_path;
    fs::path venv_python_bin;

    ModelBase() {
        // Set up virtual environment paths
        venv_path = fs::path("./Library/pythonBridgeRuntime");
    #ifdef _WIN32
            venv_python_bin = venv_path / "Scripts";
    #else
            venv_python_bin = venv_path / "bin";
    #endif
    }

    void setupPythonEnv() {
        py::module sys = py::module::import("sys");
        py::module os = py::module::import("os");
        py::module site = py::module::import("site");
        py::module sysconfig = py::module::import("sysconfig");

        std::string python_version = sysconfig.attr("get_python_version")().cast<std::string>();
        fs::path venv_site_packages = venv_path / "lib" / ("python" + python_version) / "site-packages";

        // Ensure that we are forcing Python to recognize the venv
        sys.attr("prefix") = venv_path.string();
        sys.attr("base_prefix") = venv_path.string();
        os.attr("environ")["VIRTUAL_ENV"] = venv_path.string();

        // Add venv's site-packages to sys.path
        sys.attr("path").attr("insert")(0, venv_site_packages.string());
        sys.attr("path").attr("insert")(0, venv_python_bin.string());

        site.attr("main")(); // Reload site module
    }
};

// Class for LLM Inference
class LLMInference : public ModelBase {
public:
    LLMInference() {
        setupPythonEnv();
        //Attempt to download model if neither exists
        std::string modelPath = getModelPath();
        if (modelPath.empty()) {
            std::string zygoteModelName = "zygoteBaseModel";
            std::string model_url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf?download=true";
            std::string model_path = "./" + zygoteModelName + ".gguf";
            if (!download_model(model_url, model_path)) {
                std::cerr << "[Error] : Failed to download model. Exiting." << std::endl;
                exit(1);
            }
        }
    }



    std::string LLMMainInferencing(const std::string &user_input) {
        //Attempt to download model if neither exists
        std::string modelPath = getModelPath();
        if (modelPath.empty()) {
            std::string zygoteModelName = "zygoteBaseModel";
            std::string model_url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf?download=true";
            std::string model_path = "./" + zygoteModelName + ".gguf";
            if (!download_model(model_url, model_path)) {
                std::cerr << "[Error] : Failed to download model. Exiting." << std::endl;
                exit(1);
            }
        }
        try {
            py::gil_scoped_acquire acquire;
            py::module llama_cpp;
            try {
                llama_cpp = py::module::import("llama_cpp");
            } catch (const py::error_already_set& e) {
                std::cerr << "[Error] Failed to import llama_cpp: " << e.what() << std::endl;
                return "[Error] Failed to import required Python module";
            }

            py::object llm;
            try {
                py::object Llama = llama_cpp.attr("Llama");
                llm = Llama(modelPath);
            } catch (const py::error_already_set& e) {
                std::cerr << "[Error] Failed to create model instance: " << e.what() << std::endl;
                return "[Error] Failed to initialize model";
            }

            std::string prompt = "User: " + user_input + "\nAssistant: ";
            py::object output = llm.attr("__call__")(
                py::arg("prompt") = prompt,
                py::arg("max_tokens") = 32,
                py::arg("stop") = py::make_tuple("User:", "\n"),
                py::arg("echo") = true
            );

            // --- Python-side fix ---
            py::module json = py::module::import("json");
            std::string json_string = py::str(json.attr("dumps")(output)).cast<std::string>();
            // --- End of Python-side fix ---

            try {
            // Parse the JSON string (now correctly formatted)
                auto json_response = nlohmann::json::parse(json_string);
                auto text_completion = json_response["choices"][0]["text"];
            std::string response = text_completion.get<std::string>();

            // Remove "Assistant: " prefix if present
            const std::string prefix = "Assistant: ";
            if (response.substr(0, prefix.length()) == prefix) {
                response = response.substr(prefix.length());
            }
            return response;

        } catch (const nlohmann::json::exception& e) {
            std::cerr << "[Error] Failed to parse or extract text from JSON: " << e.what() << std::endl;
            return "[Error] Invalid JSON format";
        }

    } catch (const std::exception& e) {
        std::cerr << "[Error] An unexpected error occurred: " << e.what() << std::endl;
        return "[Error] An unexpected error occurred";
    }
}

    void runInferenceSelfTesting() {
        try {
            std::string user_input = "When can we touch the sky?";

            // Inform the user that processing has started
            std::cout << "Testi..." << std::endl;

            std::string response = LLMMainInferencing(user_input);
            if (!response.starts_with("[Error]")) {

                // Post-processing to truncate (if you still want to do this)
                std::string completion = response;
                size_t pos = completion.find("savory");
                if (pos != std::string::npos) {
                    completion = completion.substr(0, pos + 7); // 7 to include "savory"
                }

                std::cout << completion << std::endl;  // Print the (possibly truncated) text completion
            }
        } catch (const std::exception& e) {
            std::cerr << "[Error] Failed to run inference: " << e.what() << std::endl;
        }
    }
};

// Define the static variable outside the class

class PagerManager {
private:
    sqlite3* db;

    struct DatabaseEntry {
        std::string pipeline_input;
        std::string visual_cues;
        std::string additional_sensory_input;
        std::string llm_response;
        int64_t epoch_written;
        int64_t epoch_accessed;
    };

    std::unordered_map<std::string, DatabaseEntry> l0_cache; // L0 buffer
    const double SIMILARITY_THRESHOLD = 0.69;

public:
    PagerManager() {
        int rc = sqlite3_open("./engineInteraction.db", &db);
        if (rc) {
            throw std::runtime_error("Cannot open database");
        }

        // Create table if not exists
        const char* sql = "CREATE TABLE IF NOT EXISTS interactions ("
                         "pipeline_input TEXT,"
                         "visual_cues TEXT,"
                         "additional_sensory_input TEXT,"
                         "llm_response TEXT,"
                         "epoch_written INTEGER,"
                         "epoch_accessed INTEGER);";

        char* errMsg = 0;
        rc = sqlite3_exec(db, sql, 0, 0, &errMsg);
        if (rc != SQLITE_OK) {
            sqlite3_free(errMsg);
            throw std::runtime_error("SQL error creating table");
        }

        prefetchToL0();
    }

    double calculateSimilarity(const std::string& str1, const std::string& str2) {
    const size_t len1 = str1.length();
    const size_t len2 = str2.length();

    // Create matrix for Levenshtein distance calculation
    std::vector<std::vector<int>> matrix(len1 + 1, std::vector<int>(len2 + 1));

    // Initialize first row and column
    for (size_t i = 0; i <= len1; i++) {
        matrix[i][0] = i;
    }
    for (size_t j = 0; j <= len2; j++) {
        matrix[0][j] = j;
    }

    // Fill in the rest of the matrix
    for (size_t i = 1; i <= len1; i++) {
        for (size_t j = 1; j <= len2; j++) {
            if (str1[i-1] == str2[j-1]) {
                matrix[i][j] = matrix[i-1][j-1];
            } else {
                matrix[i][j] = 1 + std::min({matrix[i-1][j],      // deletion
                                           matrix[i][j-1],      // insertion
                                           matrix[i-1][j-1]});  // substitution
            }
        }
    }

    // Calculate similarity score based on Levenshtein distance
    double maxLength = std::max(len1, len2);
    double distance = matrix[len1][len2];
    return 1.0 - (distance / maxLength);
}

    void prefetchToL0() {
        // Load most recently accessed entries into L0
        const char* sql = "SELECT * FROM interactions ORDER BY epoch_accessed DESC LIMIT 1000;";
        sqlite3_stmt* stmt;

        if (sqlite3_prepare_v2(db, sql, -1, &stmt, 0) == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                DatabaseEntry entry;
                entry.pipeline_input = std::string((const char*)sqlite3_column_text(stmt, 0));
                entry.visual_cues = std::string((const char*)sqlite3_column_text(stmt, 1));
                entry.additional_sensory_input = std::string((const char*)sqlite3_column_text(stmt, 2));
                entry.llm_response = std::string((const char*)sqlite3_column_text(stmt, 3));
                entry.epoch_written = sqlite3_column_int64(stmt, 4);
                entry.epoch_accessed = sqlite3_column_int64(stmt, 5);

                l0_cache[entry.pipeline_input] = entry;
            }
        }
        sqlite3_finalize(stmt);
    }

    DatabaseEntry findSimilarEntry(const std::string& input) {
        // Fuzzy search in L0 cache
        for (const auto& entry : l0_cache) {
            double similarity = calculateSimilarity(input, entry.first);
            if (similarity >= SIMILARITY_THRESHOLD) {
                // Update epoch_accessed
                int64_t current_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();

                const char* sql = "UPDATE interactions SET epoch_accessed = ? WHERE pipeline_input = ?;";
                sqlite3_stmt* stmt;
                sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
                sqlite3_bind_int64(stmt, 1, current_epoch);
                sqlite3_bind_text(stmt, 2, entry.first.c_str(), -1, SQLITE_STATIC);
                sqlite3_step(stmt);
                sqlite3_finalize(stmt);

                return entry.second;
            }
        }
        return DatabaseEntry(); // Return empty entry if no match found
    }
    sqlite3* getDB() { return db; }
};

// Class SchedulerPipeline is where you code the CoT chains and decision making here
// This is should be the first class that should be launch on int main, then the OpenAI API status quo html API
class SchedulerPipeline {
private:
    LLMInference llm_inference;
    PagerManager pager_manager;

public:
    std::string selfThoughts(const std::string& user_input) {
        // Self Contemplating Thoughts (Context automatically loaded by the processInput method)
        // Generate initial thoughts about the input
        std::string thought_prompt = "What do I think about this: " + user_input;
        std::string initial_thoughts = processInput(thought_prompt);

        // Use those thoughts to generate a more refined response
        return processInput(initial_thoughts);
    }

    std::string processInput(const std::string& user_input) {
        // Developer Note:
        // Before all of this, check Cached Entry and do fuzzy logic with threshold input 0.69 if it mach, match the highest rating and use it as an output
        // In here processInput have responsibility to fetch memory, context and vector context from the database L1 and L0 to be augmented into the llm response
        // Not only that processInput also decide based on the current prompt and the context previously if CoT is required or indepth grokking required, if it does then 

        // Check cache first
        auto cached_entry = pager_manager.findSimilarEntry(user_input);
        if (!cached_entry.llm_response.empty()) {
            return cached_entry.llm_response;
        }



        // First, let LLMMainInferencing optimize/preprocess the prompt
        std::string optimized_prompt = "Please optimize and expand upon this input while maintaining its core meaning: " + user_input;
        optimized_prompt = llm_inference.LLMMainInferencing(optimized_prompt);



        // This is the final response from the CoT processing (if it doesn't have cached entry)
        // Then, pass the optimized prompt to mainLLM
        std::string llm_response = llm_inference.LLMMainInferencing(optimized_prompt);

        // Store new interaction in database
        const char* sql = "INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(pager_manager.getDB(), sql, -1, &stmt, 0);

        int64_t current_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

        sqlite3_bind_text(stmt, 1, user_input.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, "Not Available For this Time", -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, "Not Available For this Time", -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 4, llm_response.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 5, current_epoch);
        sqlite3_bind_int64(stmt, 6, current_epoch);

        sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        return llm_response;
    }
};

class cliInterfaceMain {
private:
    SchedulerPipeline scheduler;
    const std::string WELCOME_MESSAGE =
        "Welcome to Adelaide and Albert Engine CLI Interface\n"
        "All inputs are processed through the Scheduler Pipeline\n"
        "Type 'exit' or 'quit' to end the session\n"
        "----------------------------------------\n";

public:
    void startInterface() {
        std::cout << WELCOME_MESSAGE;

        while (true) {
            try {
                // Get user input
                std::cout << "\n[User]: ";
                std::string user_input;
                std::getline(std::cin, user_input);

                // Check exit conditions
                if (user_input == "exit" || user_input == "quit") {
                    std::cout << "Shutting down the interface...\n";
                    std::cout << "If the program doesn't stop, press Ctrl+C\n";
                    break;
                }

                // Skip empty inputs
                if (user_input.empty()) {
                    continue;
                }

                // Process through scheduler pipeline
                //std::cout << "\n[SystemDebug_RemoveThisLater]: Processing through Scheduler Pipeline...\n";
                std::string final_response = scheduler.processInput(user_input);

                // Display the response
                std::cout << "[Assistant]: " << final_response << std::endl;

            } catch (const std::exception& e) {
                std::cerr << "[Error]: An error occurred - " << e.what() << std::endl;
                std::cout << "The interface will continue running...\n";
            }
        }
    }
};


// Class for LLM Finetuning (Placeholder)
class LLMFinetune : public ModelBase {
public:
    LLMFinetune() {
        //py::scoped_interpreter guard{}; Do not reinitialize the interpreter again since it's already initialize on this domain main.cpp program
        setupPythonEnv();
    }

    void finetuneModel() {
        // Implement finetuning logic here (Import from )
        std::cout << "[Finetuning Model]: Stub has been launched!" << std::endl;
    }
};

// Class for SD (Stable Diffusion) Inference (Placeholder)
class SDInference : public ModelBase {
public:
    SDInference() {
        //py::scoped_interpreter guard{}; Do not reinitialize the interpreter again since it's already initialize on this domain main.cpp program
        setupPythonEnv();
    }

    void runInferenceSelfTesting() {
        // Implement Stable Diffusion inference logic here
        std::cout << "[Running SD Inference Testing]: Stub has been launched!" << std::endl;
    }
};

// Class for SD (Stable Diffusion) Finetuning (Placeholder)
class SDFinetune : public ModelBase {
public:
    SDFinetune() {
        //py::scoped_interpreter guard{}; Do not reinitialize the interpreter again since it's already initialize on this domain main.cpp program
        setupPythonEnv();
    }

    void finetuneModel() {
        // Implement Stable Diffusion finetuning logic here
        std::cout << "[Finetuning SD Model]: Stub has been launched!" << std::endl;
    }
};

// -----------------------------------------------Memory Dumping debugging-------------------------------------------------------------

// Function to handle signals (e.g., SIGSEGV)
// Watchdog auto Restart and memory dumping debugging
void signalHandler(int signum) {
    if (signum == SIGINT) {
        // Handle SIGINT (keyboard interrupt) by simply logging and exiting
        std::cerr << "Received keyboard interrupt (SIGINT). Exiting program gracefully." << std::endl;
        exit(0);
    }

    // Log the signal number and stack trace for other signals
    std::cerr << "Error: signal " << signum << std::endl;

    void* array[10];
    size_t size = backtrace(array, 10);
    std::cerr << "Obtained " << size << " stack frames." << std::endl;
    backtrace_symbols_fd(array, size, STDERR_FILENO);

    // Save backtrace and command log to a file
    std::ofstream log_file("crash_dump.log", std::ios::app);
    if (log_file.is_open()) {
        log_file << "Error: signal " << signum << std::endl;
        log_file << "Obtained " << size << " stack frames." << std::endl;

        char** messages = backtrace_symbols(array, size);

        for (size_t i = 0; i < size && messages != nullptr; ++i) {
            log_file << "[bt]: (" << i << ") " << messages[i] << std::endl;
        }
        free(messages);

        log_file << "\nRecent command log:\n";
        for (const auto& command : command_log) {
            log_file << command << std::endl;
        }
    }
    log_file.close();

    // Attempt to restart the program for other signals
    std::string exePath = getExecutablePath();
    if (!exePath.empty()) {
        pid_t pid = fork();
        if (pid == 0) { // This is the child process
            // Replace the current process with a new instance of the program
            char* args[] = {const_cast<char*>(exePath.c_str()), nullptr};
            execv(args[0], args);
            // If execv returns, it must have failed.
            std::cerr << "Failed to restart the program." << std::endl;
            exit(EXIT_FAILURE);
        } else if (pid > 0) { // This is the parent process
            int status;
            waitpid(pid, &status, 0); // Wait for the child process to finish
        } else {
            std::cerr << "Failed to fork process for restart." << std::endl;
        }
    } else {
        std::cerr << "Unable to determine executable path." << std::endl;
    }

    // Terminate the original program for signals other than SIGINT
    exit(signum);
}

// Inference and Fine-Tuning Scheduler Implementation "Backbrain Scheduler"
// All the Inference from CLI and HTML HAVE to go through this scheduler thus the scheduler will allocate or redirect the result and serve ther result either from cache result or


// ------------------------------------------------------------------------------------------------------------

int main() {

    if (!pyGuard) {
        pyGuard = std::make_unique<py::scoped_interpreter>();
    }

    // Register the signal handler for segmentation fault (SIGSEGV) (This is going to be very useful espescially running on a weak memory architecture, I'm looking at you Apple Silicon)
    // Register the signal handler for segmentation fault (SIGSEGV)
    signal(SIGSEGV, signalHandler);
    signal(SIGABRT, signalHandler);  // Catch abort signals (e.g., assertion failures)
    signal(SIGFPE, signalHandler);   // Catch floating-point errors
    signal(SIGINT, signalHandler);   // Interrupt signal (Ctrl+C)

    // Example code that will cause a segmentation fault (for testing purposes)
    //int* ptr = nullptr;
    //*ptr = 42;  // This will cause a segmentation fault

    LLMInference llm_inference; // we move this into scheduler later on (so it decided when or where it shall be invoked)
    llm_inference.runInferenceSelfTesting(); // we move this into scheduler later on (so it decided when or where it shall be invoked)

    LLMFinetune llm_finetune; // we move this into scheduler later on (so it decided when or where it shall be invoked)
    llm_finetune.finetuneModel(); // we move this into scheduler later on (so it decided when or where it shall be invoked)

    SDInference sd_inference; // we move this into scheduler later on (so it decided when or where it shall be invoked)
    sd_inference.runInferenceSelfTesting(); // we move this into scheduler later on (so it decided when or where it shall be invoked)

    SDFinetune sd_finetune; // we move this into scheduler later on (so it decided when or where it shall be invoked)
    sd_finetune.finetuneModel();// we move this into scheduler later on (so it decided when or where it shall be invoked)
    crow::SimpleApp app;

    std::cout << CONSOLE_PREFIX << "🌐 Starting up the Adelaide&Albert Engine... Let's make some magic happen!\n";

    #ifdef _WIN32
        // Windows: Redirect stderr to NUL
        freopen("NUL", "w", stderr);
    #else
        // Unix-like systems: Redirect stderr to /dev/null
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull != -1) {
            dup2(devnull, STDERR_FILENO);
        close(devnull);
        }

    #endif
    // Define endpoint
    CROW_ROUTE(app, "/api/generate")
        .methods(HTTPMethodType{crow::HTTPMethod::Post}.get()) // Use strong typing for the HTTP method
        ([](const crow::request& req) -> crow::response { // Strongly typed lambda return type
            std::cout << CONSOLE_PREFIX << "📥 A new request has arrived at /api/generate. Let’s see what treasures it holds!\n";

            // Parse the JSON request body
            crow::json::rvalue json = crow::json::load(req.body);
            if (!json) {
                std::cout << CONSOLE_PREFIX << "❌ Oops! That didn’t look like valid JSON. Check your scroll and try again.\n";
                return crow::response(400, "🚫 Invalid JSON request");
            }

            // Use the is_string function to check if the fields are strings
            if (!json.has("model") || !is_string(json["model"]) ||
                !json.has("prompt") || !is_string(json["prompt"])) {
                std::cout << CONSOLE_PREFIX << "⚠️ Missing or muddled fields. I need 'model' and 'prompt' to conjure the magic!\n";
                return crow::response(400, "🚫 Missing or invalid 'model' or 'prompt' field");
            }

            // Get the model and prompt from the JSON
            Model model{json["model"].s()};
            Prompt prompt{json["prompt"].s()};

            std::cout << CONSOLE_PREFIX << "🧠 Engaging with model: " << model.get() << ". Here comes some wisdom...\n";

            // Generate text using the placeholder function
            ResponseText generated_text = generate_text(model, prompt);

            // Create the JSON response
            crow::json::wvalue response;

            response["model"] = model.get(); // Include the model in the response
            response["response"] = crow::json::wvalue(std::move(generated_text.get())); // Move the generated JSON object
            response["done"] = true;

            std::cout << CONSOLE_PREFIX << "🎉 Success! Your response is ready. Feel the wisdom flow.\n";

            // Return the JSON response directly
            return crow::response(response);
        });

    // Start the server in a separate thread
    std::thread serverOpenAIEmulationInferenceThread([&app]() {
        std::cout << CONSOLE_PREFIX << "🚀 The engine roars to life on port 8080. Ready to enlighten the world!\n";
        app.port(8080).multithreaded().run();
    });

    

    // CLI chat interface can be started in the main thread
    cliInterfaceMain cli;
    cli.startInterface();

    // Join the server thread before exiting main
    serverOpenAIEmulationInferenceThread.join();

    return 0;
}
