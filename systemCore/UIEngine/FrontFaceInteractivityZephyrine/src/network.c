#include <stdio.h>
#include <string.h>
#include <stdlib.h> // For malloc, free
#include <curl/curl.h>
#include "cJSON.h"
#include "network.h" // Include network header

static char *_target_ip = NULL;
static int _target_port = 0;

// Struct to hold the response from curl
struct MemoryStruct {
    char *memory;
    size_t size;
};

// Callback function for curl to write received data
static size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;

    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if (!ptr) {
        // out of memory!
        fprintf(stderr, "not enough memory (realloc returned NULL)\n");
        return 0;
    }

    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;

    return realsize;
}

// Global initialization for curl
void network_init(void) {
    curl_global_init(CURL_GLOBAL_ALL);
}

// Global cleanup for curl
void network_cleanup(void) {
    curl_global_cleanup();
    g_free(_target_ip);
    _target_ip = NULL;
}

void network_set_target(const char *ip, int port) {
    if (_target_ip) {
        g_free(_target_ip);
    }
    _target_ip = g_strdup(ip);
    _target_port = port;
}

// Function to make a synchronous HTTP GET request
char* http_get_sync(const char* url) {
    CURL *curl;
    CURLcode res;
    struct MemoryStruct chunk;

    chunk.memory = malloc(1);
    chunk.size = 0;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
        
        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
            free(chunk.memory);
            return NULL;
        }
        curl_easy_cleanup(curl);
    }
    return chunk.memory; // Caller must free this
}

// Function to make a synchronous HTTP POST request
char* http_post_sync(const char* url, const char* post_fields) {
    CURL *curl;
    CURLcode res;
    struct MemoryStruct chunk;

    chunk.memory = malloc(1);
    chunk.size = 0;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_fields);
        
        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
            free(chunk.memory);
            return NULL;
        }
        curl_easy_cleanup(curl);
    }
    return chunk.memory; // Caller must free this
}

// Thread function for asynchronous network requests
static gpointer network_thread_func(gpointer data) {
    NetworkRequest *request = (NetworkRequest *)data;
    char *response_data = NULL;
    char full_url[512];

    // Construct full URL if a relative path is provided, otherwise use as is.
    // For simplicity, assuming `url` in NetworkRequest is already a full URL for now.
    // In a real scenario, you'd check if request->url starts with "http://" or "https://".
    if (_target_ip && _target_port != 0 && strstr(request->url, "://") == NULL) {
        snprintf(full_url, sizeof(full_url), "http://%s:%d%s", _target_ip, _target_port, request->url);
    } else {
        snprintf(full_url, sizeof(full_url), "%s", request->url);
    }

    if (strcmp(request->method, "GET") == 0) {
        response_data = http_get_sync(full_url);
    } else if (strcmp(request->method, "POST") == 0) {
        response_data = http_post_sync(full_url, request->body);
    }

    if (response_data) {
        request->response_json = cJSON_Parse(response_data);
        free(response_data);
    } else {
        request->response_json = NULL;
    }
    
    g_async_queue_push(request->queue, request); // Push the completed request back to the main thread
    return NULL;
}

// Function to initiate an asynchronous network request in a separate thread
void network_perform_request_async(const char *url, const char *method, const char *body, GAsyncQueue *response_queue, NetworkRequestType type) {
    NetworkRequest *request = g_new(NetworkRequest, 1);
    request->url = g_strdup(url);
    request->method = g_strdup(method);
    request->body = body ? g_strdup(body) : NULL;
    request->response_json = NULL;
    request->queue = response_queue;
    request->request_type = type; // Set the request type

    g_thread_new("network_thread", network_thread_func, request);
}
