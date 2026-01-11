#ifndef NETWORK_H
#define NETWORK_H

#include <glib.h> // For GThread, GMutex, etc.
#include "cJSON.h"

typedef enum {
    NETWORK_REQUEST_TYPE_FCS_DATA,
    NETWORK_REQUEST_TYPE_CHAT_RESPONSE,
    NETWORK_REQUEST_TYPE_CHAT_NOTIFICATION,
    NETWORK_REQUEST_TYPE_CORTEX_CONFIG
} NetworkRequestType;

// Define a structure to hold network request parameters
typedef struct {
    char *url;
    char *method; // "GET", "POST", etc.
    char *body;   // For POST requests
    cJSON *response_json; // To store parsed JSON response
    GAsyncQueue *queue; // Queue for thread-safe communication
    NetworkRequestType request_type; // New field to distinguish request types
} NetworkRequest;

// Function to set the target IP and Port for network requests
void network_set_target(const char *ip, int port);

// Function to initialize network (curl global init)
void network_init(void);

// Function to clean up network (curl global cleanup)
void network_cleanup(void);

// Function to perform an HTTP GET request synchronously
char* http_get_sync(const char* url);

// Function to perform an HTTP POST request synchronously
char* http_post_sync(const char* url, const char* post_fields);

// Function to initiate an asynchronous network request in a separate thread
void network_perform_request_async(const char *url, const char *method, const char *body, GAsyncQueue *response_queue, NetworkRequestType type);

#endif // NETWORK_H