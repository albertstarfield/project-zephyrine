#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <iomanip>
#include "./Library/crow.h"

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
    response_json["status_11"] = "success";

    return ResponseText{std::move(response_json)};
}

bool is_string(const crow::json::rvalue& value) {
    return value.t() == crow::json::type::String;
}

int main() {
    crow::SimpleApp app;

    // /api/generate endpoint
    CROW_ROUTE(app, "/api/generate")
    .methods(HTTPMethodType{crow::HTTPMethod::Post}.get()) // Use strong typing for the HTTP method
    ([](const crow::request& req) -> crow::response { // Strongly typed lambda return type
        // Parse the JSON request body
        crow::json::rvalue json = crow::json::load(req.body);
        if (!json) {
            return crow::response(400, "Invalid JSON request");
        }

        // Use the is_string function to check if the fields are strings
        if (!json.has("model") || !is_string(json["model"]) ||
            !json.has("prompt") || !is_string(json["prompt"])) {
            return crow::response(400, "Missing or invalid 'model' or 'prompt' field");
        }
        
        // Get the model and prompt from the JSON
        Model model{json["model"].s()};
        Prompt prompt{json["prompt"].s()};

        // Generate text using the placeholder function 
        ResponseText generated_text = generate_text(model, prompt);

        // Create the JSON response
        crow::json::wvalue response;
        
        response["model"] = model.get(); // Include the model in the response
        response["response"] = crow::json::wvalue(std::move(generated_text.get())); // Move the generated JSON object
        response["done"] = true;

        // Return the JSON response directly
        return crow::response(response);
    });

    // Start the server
    app.port(8080).multithreaded().run();

    return 0;
}
