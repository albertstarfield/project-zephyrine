#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <iomanip>
#include "./Library/crow.h"
#include <pybind11/embed.h>
// #include "./Library/curl.h"

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
const std::string engineName = "Adelaide Paradigm Engine";


const std::string CONSOLE_PREFIX = "[Adelaide&Albert Engine] : ";


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




int inference() { // in here define the virtual environment?
    py::scoped_interpreter guard{};  // Start the Python interpreter
    try {
        // Import the Python module (ensure the path is set correctly)
        py::module llama_infer = py::module::import("llama_infer");
        // Get the 'infer' function from the module
        py::object infer = llama_infer.attr("infer");

        // Sample input text
        std::string input_text = "Your input text for the model.";

        // Call the Python function and get the result
        py::object result = infer(input_text);

        // Output the result to the console
        std::cout << "[Adelaide&Albert Paradigm Engine] : " << result.cast<std::string>() << std::endl;
    } catch (const py::error_already_set& e) {
        std::cerr << "[Error] : " << e.what() << std::endl;
    }

    return 0;
}

int main() {
    inference(); // Call the inference function for testing
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
    app.port(8080).multithreaded().run();

    std::cout << CONSOLE_PREFIX << "🚀 The engine roars to life on port 8080. Ready to enlighten the world!\n";

    return 0;
}