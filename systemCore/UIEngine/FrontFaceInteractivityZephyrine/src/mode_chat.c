#include <gtk/gtk.h>
#include "mode_chat.h"
#include <string.h>
#include <curl/curl.h> // For curl_easy_escape
#include "network.h" // For network_perform_request_async and NetworkRequestType

// Explicit prototypes for GtkEntry functions that might not be correctly picked up
const char *gtk_entry_get_text(GtkEntry *entry);
void gtk_entry_set_text(GtkEntry *entry, const char *text);

extern GAsyncQueue *network_response_queue; // External reference to the main network response queue

static GtkWidget *chat_history_box_global = NULL;
static GtkWidget *chat_scrolled_window_global = NULL;

void mode_chat_add_message(const char *sender, const char *message) {
    if (!chat_history_box_global) return;

    GtkWidget *label = gtk_label_new(NULL);
    char *markup = g_strdup_printf("<b>%s:</b> %s", sender, message);
    gtk_label_set_markup(GTK_LABEL(label), markup);
    g_free(markup);

    gtk_label_set_wrap(GTK_LABEL(label), TRUE);
    gtk_label_set_xalign(GTK_LABEL(label), 0.0); // Align text to the start

    gtk_box_append(GTK_BOX(chat_history_box_global), label);
    gtk_widget_set_visible(label, TRUE);

    // Scroll to the bottom
    GtkAdjustment *adjustment = gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(chat_scrolled_window_global));
    if (adjustment) {
        gtk_adjustment_set_value(adjustment, gtk_adjustment_get_upper(adjustment));
    }
}

static void on_send_button_clicked(GtkButton *button, gpointer user_data) {
    GtkEntry *entry = GTK_ENTRY(user_data);
    const char *text = gtk_entry_get_text(entry);
    if (strlen(text) > 0) {
        mode_chat_add_message("You", text);
        
        CURL *curl = curl_easy_init();
        if (curl) {
            char *encoded_message = curl_easy_escape(curl, text, strlen(text));
            if (encoded_message) {
                char *url = g_strdup_printf("/v1/chat/responsezeph?message=%s", encoded_message);
                network_perform_request_async(url, "GET", NULL, network_response_queue, NETWORK_REQUEST_TYPE_CHAT_RESPONSE);
                g_free(url);
                curl_free(encoded_message);
            }
            curl_easy_cleanup(curl);
        }

        gtk_entry_set_text(entry, ""); // Clear the input field
    }
}

static void on_message_entry_activate(GtkEntry *entry, gpointer user_data) {
    on_send_button_clicked(NULL, entry); // Simulate button click
}


static GtkWidget *create_chat_page(void) {
    GtkWidget *vbox;
    GtkWidget *scrolled_window;
    GtkWidget *chat_history_box;
    GtkWidget *hbox;
    GtkWidget *message_entry;
    GtkWidget *send_button;

    vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_vexpand(vbox, TRUE);

    scrolled_window = gtk_scrolled_window_new();
    gtk_widget_set_hexpand(scrolled_window, TRUE);
    gtk_widget_set_vexpand(scrolled_window, TRUE);
    gtk_box_append(GTK_BOX(vbox), scrolled_window);
    chat_scrolled_window_global = scrolled_window; // Set global reference

    chat_history_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scrolled_window), chat_history_box);
    chat_history_box_global = chat_history_box; // Set global reference

    hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_widget_set_hexpand(hbox, TRUE);
    gtk_box_append(GTK_BOX(vbox), hbox);

    message_entry = gtk_entry_new();
    gtk_widget_set_hexpand(message_entry, TRUE);
    g_signal_connect(message_entry, "activate", G_CALLBACK(on_message_entry_activate), NULL); // Send on Enter
    gtk_box_append(GTK_BOX(hbox), message_entry);

    send_button = gtk_button_new_with_label("Send");
    g_signal_connect(send_button, "clicked", G_CALLBACK(on_send_button_clicked), message_entry);
    gtk_box_append(GTK_BOX(hbox), send_button);

    return vbox;
}

// Placeholder for Mode 2 (Cognitive Interface) functionality
GtkWidget *mode_chat_create_widget(void) {
    GtkWidget *notebook = gtk_notebook_new();
    gtk_widget_set_vexpand(notebook, TRUE);
    gtk_widget_set_hexpand(notebook, TRUE);

    // Tab A: Interaction History (Chat View)
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), create_chat_page(), gtk_label_new("Chat"));

    // Tab B: Image Generation
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), gtk_label_new("Image Generation Content"), gtk_label_new("Image"));

    // Tab C: Knowledge Tuning
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), gtk_label_new("Knowledge Tuning Content"), gtk_label_new("Knowledge"));

    // Tab D: Settings
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), gtk_label_new("Settings Content"), gtk_label_new("Settings"));
    
    return notebook;
}