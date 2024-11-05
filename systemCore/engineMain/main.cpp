#include <iostream>
#include <memory>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
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
#include <pybind11/embed.h>
#include "./Library/crow.h"
#include <sys/wait.h>  // For waitpid

#ifdef _WIN32
#include <windows.h>
#include <cstring>
#endif

std::string getExecutablePath() {
#ifdef __linux__ // Linux systems
    char buf[1024];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len != -1) {
        buf[len] = '\0';
        return std::string(buf);
    }
#elif defined(_WIN32) // Windows systems
    char buf[MAX_PATH];
    if (GetModuleFileName(NULL, buf, MAX_PATH)) {
        return std::string(buf);
    }
#endif
    return "";
}

namespace fs = std::filesystem; // Universal or cross platform path reconstruction
namespace py = pybind11;


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
        std::cerr << "[Error] :  Zygote Model Downloader Failed to initialize curl" << std::endl;
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // Follow redirects if necessary

    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "[Error] : curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        file.close();
        curl_easy_cleanup(curl);
        return false;
    }

    file.close();
    curl_easy_cleanup(curl);
    std::cout << "[Info] : Zygote Model downloaded successfully to " << output_path << std::endl;
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

    return static_cast<char>(dist(generator));
}

// Function to generate a random SHA-256 string
std::string generate_random_sha256() {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> dist(0, 15);

    std::stringstream sha256;
    char fill_char = generate_random_fill_char(); // Get a random fill character
    sha256 << std::hex << std::setfill(fill_char); // Use the random fill character
    for (int i = 0; i < 64; ++i) {
        sha256 << std::setw(1) << dist(generator);
    }

    return sha256.str();
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
private:
    std::unique_ptr<py::scoped_interpreter> guard;
    
public:
    LLMInference() {
        // Initialize Python interpreter only once
        guard = std::make_unique<py::scoped_interpreter>();
        setupPythonEnv();
    }

    std::string LLMchild(const std::string &user_input) {
        std::string model_path = "./tinyLLaMatestModel.gguf";
        if (!fs::exists(model_path)) {
            std::cout << "[Info] : Model file not found, downloading..." << std::endl;
            std::string url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf?download=true";
            if (!download_model(url, model_path)) {
                return "[Error] Model download failed";
            }
        }

        try {
            // Create a local scope for Python objects
            py::gil_scoped_acquire acquire;
            
            // Import with error checking
            py::module llama_cpp;
            try {
                llama_cpp = py::module::import("llama_cpp");
            } catch (const py::error_already_set& e) {
                std::cerr << "[Error] Failed to import llama_cpp: " << e.what() << std::endl;
                return "[Error] Failed to import required Python module";
            }

            // Create model instance with error checking
            py::object llm;
            try {
                py::object Llama = llama_cpp.attr("Llama");
                llm = Llama(model_path);
            } catch (const py::error_already_set& e) {
                std::cerr << "[Error] Failed to create model instance: " << e.what() << std::endl;
                return "[Error] Failed to initialize model";
            }

            // Construct prompt and generate response
            std::string prompt = "User: " + user_input + "\nAssistant: ";
            py::object output = llm.attr("__call__")(
                py::arg("prompt") = prompt,
                py::arg("max_tokens") = 32,
                py::arg("stop") = py::make_tuple("User:", "\n"),
                py::arg("echo") = true
            );

            // Convert output to string with error checking
            try {
                return py::str(output).cast<std::string>();
            } catch (const py::cast_error& e) {
                std::cerr << "[Error] Failed to convert model output to string: " << e.what() << std::endl;
                return "[Error] Invalid model output format";
            }

        } catch (const std::exception& e) {
            std::cerr << "[Error] Unexpected error: " << e.what() << std::endl;
            return "[Error] An unexpected error occurred";
        }
    }

    void runInference() {
        try {
            std::string user_input = "When can we touch the sky? You were born fated to not have a place on the earth. You were born to be with the stars!";
            std::string response = LLMchild(user_input);
            std::cout << "[Inference Result] : " << response << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Error] Failed to run inference: " << e.what() << std::endl;
        }
    }
};

// Class for LLM Finetuning (Placeholder)
class LLMFinetune : public ModelBase {
public:
    LLMFinetune() {
        py::scoped_interpreter guard{};
        setupPythonEnv();
    }

    void finetuneModel() {
        // Implement finetuning logic here
        std::cout << "[Finetuning Model]" << std::endl;
    }
};

// Class for SD (Stable Diffusion) Inference (Placeholder)
class SDInference : public ModelBase {
public:
    SDInference() {
        py::scoped_interpreter guard{};
        setupPythonEnv();
    }

    void runInference() {
        // Implement Stable Diffusion inference logic here
        std::cout << "[Running SD Inference]" << std::endl;
    }
};

// Class for SD (Stable Diffusion) Finetuning (Placeholder)
class SDFinetune : public ModelBase {
public:
    SDFinetune() {
        py::scoped_interpreter guard{};
        setupPythonEnv();
    }

    void finetuneModel() {
        // Implement Stable Diffusion finetuning logic here
        std::cout << "[Finetuning SD Model]" << std::endl;
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

    // Generate the backtrace (stack trace)
    void* array[10];
    size_t size = backtrace(array, 10);
    std::cerr << "Obtained " << size << " stack frames." << std::endl;
    backtrace_symbols_fd(array, size, STDERR_FILENO);

    // Optionally write the backtrace to a file
    std::ofstream log_file("crash_dump.log", std::ios::app);
    if (log_file.is_open()) {
        log_file << "Error: signal " << signum << std::endl;
        log_file << "Obtained " << size << " stack frames." << std::endl;
        char** messages = backtrace_symbols(array, size);
        for (size_t i = 0; i < size && messages != nullptr; ++i) {
            log_file << "[bt]: (" << i << ") " << messages[i] << std::endl;
        }
        free(messages);
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


// ------------------------------------------------------------------------------------------------------------

int main() {
    // Register the signal handler for segmentation fault (SIGSEGV)
    signal(SIGSEGV, signalHandler);
    signal(SIGABRT, signalHandler);  // Catch abort signals (e.g., assertion failures)
    signal(SIGFPE, signalHandler);   // Catch floating-point errors
    signal(SIGINT, signalHandler);   // Interrupt signal (Ctrl+C)

    // Example code that will cause a segmentation fault (for testing purposes)
    //int* ptr = nullptr;
    //*ptr = 42;  // This will cause a segmentation fault

    LLMInference llm_inference;
    llm_inference.runInference();

    LLMFinetune llm_finetune;
    llm_finetune.finetuneModel();

    SDInference sd_inference;
    sd_inference.runInference();

    SDFinetune sd_finetune;
    sd_finetune.finetuneModel();
    crow::SimpleApp app;

    std::cout << CONSOLE_PREFIX << "🌐 Starting up the Adelaide&Albert Engine... Let's make some magic happen!\n";

    // /api/generate endpoint
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

    // Start the server
    std::cout << CONSOLE_PREFIX << "🚀 The engine roars to life on port 8080. Ready to enlighten the world!\n";
    app.port(8080).multithreaded().run();
    return 0;
}