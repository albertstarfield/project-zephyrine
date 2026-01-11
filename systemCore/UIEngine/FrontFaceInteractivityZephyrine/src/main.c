#include <gtk/gtk.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mode_fcs.h"
#include "mode_chat.h"
#include "network.h" // Include network header

static gchar *target_ip = NULL;
static gint target_port = 0;

static GAsyncQueue *network_response_queue; // Queue for network responses
static GtkWidget *fcs_drawing_area_global = NULL; // Global reference to FCS drawing area

// Callback for processing network responses
static gboolean handle_network_response(gpointer user_data) {
    if (g_async_queue_length(network_response_queue) > 0) {
        NetworkRequest *request = (NetworkRequest *)g_async_queue_pop(network_response_queue);

        if (request) {
            if (request->response_json) {
                char *json_str = cJSON_Print(request->response_json);
                if (json_str) {
                    switch (request->request_type) {
                        case NETWORK_REQUEST_TYPE_FCS_DATA:
                            if (fcs_drawing_area_global) {
                                mode_fcs_update_data(fcs_drawing_area_global, json_str);
                            }
                            break;
                        case NETWORK_REQUEST_TYPE_CHAT_RESPONSE: {
                            // Assuming chat response is a JSON with a "text" field
                            cJSON *text_obj = cJSON_GetObjectItemCaseSensitive(request->response_json, "text");
                            if (cJSON_IsString(text_obj)) {
                                mode_chat_add_message("Bot", cJSON_GetStringValue(text_obj));
                            } else {
                                mode_chat_add_message("Bot", json_str); // Fallback if no "text" field
                            }
                            break;
                        }
                        case NETWORK_REQUEST_TYPE_CHAT_NOTIFICATION:
                            // Handle chat notifications
                            break;
                        case NETWORK_REQUEST_TYPE_CORTEX_CONFIG:
                            // Handle cortex config response
                            break;
                    }
                    g_free(json_str);
                }
                cJSON_Delete(request->response_json);
            } else {
                // Handle cases where response_json is NULL (e.g., network error)
                g_printerr("Network request of type %d failed or returned empty response.\n", request->request_type);
            }
            // Free request data
            g_free(request->url);
            g_free(request->method);
            if (request->body) g_free(request->body);
            g_free(request);
        }
    }
    return G_SOURCE_CONTINUE; // Keep checking
}

static void on_mode_toggle(GtkToggleButton *button, gpointer user_data) {
    GtkStack *stack = GTK_STACK(user_data);
    if (gtk_toggle_button_get_active(button)) {
        gtk_stack_set_visible_child_name(stack, "chat_mode");
    } else {
        gtk_stack_set_visible_child_name(stack, "fcs_mode");
    }
}

static void activate(GtkApplication *app, gpointer user_data) {
    GtkWidget *window;
    GtkWidget *box;
    GtkWidget *toggle_button;
    GtkWidget *stack;
    GtkWidget *chat_widget;

    window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(window), "Project Zephyrine");
    gtk_window_set_default_size(GTK_WINDOW(window), 800, 600);
    
    box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_window_set_child(GTK_WINDOW(window), box);

    toggle_button = gtk_toggle_button_new_with_label("Switch to Chat Mode");
    gtk_widget_set_hexpand(toggle_button, TRUE);
    // Removed the problematic g_signal_connect
    gtk_box_append(GTK_BOX(box), toggle_button);

    stack = gtk_stack_new();
    gtk_widget_set_hexpand(stack, TRUE);
    gtk_widget_set_vexpand(stack, TRUE);
    gtk_box_append(GTK_BOX(box), stack);

    fcs_drawing_area_global = mode_fcs_create_widget(); // Assign to global
    gtk_stack_add_titled(GTK_STACK(stack), fcs_drawing_area_global, "fcs_mode", "Avionics View");

    chat_widget = mode_chat_create_widget();
    gtk_stack_add_titled(GTK_STACK(stack), chat_widget, "chat_mode", "Cognitive Interface");

    // Initially show FCS mode
    gtk_stack_set_visible_child_name(stack, "fcs_mode");
    g_signal_connect(toggle_button, "toggled", G_CALLBACK(on_mode_toggle), stack); // Connect with correct user_data

    gtk_window_present(GTK_WINDOW(window));

    // Start network polling for FCS data
    g_timeout_add_seconds(1, handle_network_response, NULL);
    char *fcs_url = g_strdup_printf("/instrumentviewportdatastreamlowpriopreview"); // network module handles full URL
    network_perform_request_async(fcs_url, "GET", NULL, network_response_queue, NETWORK_REQUEST_TYPE_FCS_DATA);
    g_free(fcs_url);
}

int main(int argc, char *argv[]) {
    GtkApplication *app;
    int status;
    int i;

    // Initialize network response queue
    network_response_queue = g_async_queue_new();
    network_init(); // Initialize libcurl globally

    // Parse command-line arguments
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--target") == 0 && (i + 1) < argc) {
            char *token;
            char *arg_copy = g_strdup(argv[++i]); // Duplicate string for strtok_r
            char *saveptr;

            token = strtok_r(arg_copy, ":", &saveptr);
            if (token != NULL) {
                target_ip = g_strdup(token);
                token = strtok_r(NULL, ":", &saveptr);
                if (token != NULL) {
                    target_port = atoi(token);
                }
            }
            g_free(arg_copy);
        }
    }

    if (!target_ip || target_port == 0) {
        g_printerr("Usage: %s --target <IP>:<PORT>\n", argv[0]);
        return 1;
    }
    
    // Set target IP and Port in network module
    network_set_target(target_ip, target_port);

    app = gtk_application_new("org.zephyrine.zephy_gui", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);

    network_cleanup(); // Clean up libcurl globally
    g_async_queue_unref(network_response_queue);
    g_free(target_ip); // Free allocated memory for target_ip

    return status;
}
