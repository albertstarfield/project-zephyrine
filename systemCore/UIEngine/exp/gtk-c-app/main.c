#include <gtk/gtk.h>
#include <libwebsockets.h>
#include <pthread.h>
#include <glib.h>
#include <json-glib/json-glib.h>

static struct lws_context *context;
static struct lws *wsi;
static int interrupted;

typedef struct {
    GtkWidget *chat_feed;
    GtkWidget *history_list;
    GtkWidget *info_bar;
    GtkWidget *info_label;
    GtkWidget *welcome_screen;
    GtkWidget *main_stack;
} UIWidgets;

typedef struct {
    GtkWidget *chat_feed;
    char *message;
} MessageData;

static GtkWidget *last_assistant_message = NULL;

static void show_error_info_bar(gpointer user_data, const char* message) {
    UIWidgets* widgets = (UIWidgets*)user_data;
    gtk_label_set_text(GTK_LABEL(widgets->info_label), message);
    gtk_info_bar_set_revealed(GTK_INFO_BAR(widgets->info_bar), TRUE);
}

static gboolean add_message_to_chat_feed(gpointer user_data) {
    MessageData *data = (MessageData *)user_data;
    GtkWidget *label = gtk_label_new(data->message);
    gtk_widget_set_halign(label, GTK_ALIGN_START);
    gtk_widget_add_css_class(label, "assistant-message");
    gtk_box_append(GTK_BOX(data->chat_feed), label);
    g_free(data->message);
    g_slice_free(MessageData, data);
    return G_SOURCE_REMOVE;
}

static gboolean append_to_assistant_message(gpointer user_data) {
    MessageData *data = (MessageData *)user_data;
    if (!last_assistant_message) {
        last_assistant_message = gtk_label_new(data->message);
        gtk_widget_set_halign(last_assistant_message, GTK_ALIGN_START);
        gtk_widget_add_css_class(last_assistant_message, "assistant-message");
        gtk_box_append(GTK_BOX(data->chat_feed), last_assistant_message);
    } else {
        const char *old_text = gtk_label_get_text(GTK_LABEL(last_assistant_message));
        char *new_text = g_strdup_printf("%s%s", old_text, data->message);
        gtk_label_set_text(GTK_LABEL(last_assistant_message), new_text);
        g_free(new_text);
    }
    g_free(data->message);
    g_slice_free(MessageData, data);
    return G_SOURCE_REMOVE;
}


static void on_chat_selected(GtkListBox *box, GtkListBoxRow *row, gpointer user_data);
static gboolean update_history_list(gpointer user_data);

static int callback_echo(struct lws *wsi, enum lws_callback_reasons reason,
                         void *user, void *in, size_t len)
{
    UIWidgets *widgets = (UIWidgets *)user;
    switch (reason) {
    case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
        lwsl_err("CLIENT_CONNECTION_ERROR: %s\n", in ? (char *)in : "(null)");
        show_error_info_bar(widgets, "Connection error");
        wsi = NULL;
        break;

    case LWS_CALLBACK_CLIENT_ESTABLISHED:
        {
            lwsl_user("CLIENT_ESTABLISHED\n");
            const char *get_history_message = "{\"type\":\"get_chat_history_list\",\"payload\":{\"userId\":\"user\"}}";
            unsigned char* buf = g_malloc(LWS_PRE + strlen(get_history_message));
            memcpy(&buf[LWS_PRE], get_history_message, strlen(get_history_message));
            lws_write(wsi, &buf[LWS_PRE], strlen(get_history_message), LWS_WRITE_TEXT);
            g_free(buf);
        }
        break;

    case LWS_CALLBACK_CLIENT_RECEIVE:
        {
            lwsl_user("Received: %.*s\n", (int)len, (char *)in);
            
            JsonParser *parser = json_parser_new();
            gboolean success = json_parser_load_from_data(parser, (const char*)in, len, NULL);
            if(success) {
                JsonNode *root = json_parser_get_root(parser);
                JsonObject *root_obj = json_node_get_object(root);
                if (json_object_has_member(root_obj, "type")) {
                    const char *type = json_object_get_string_member(root_obj, "type");
                    if (strcmp(type, "chat_history_list") == 0) {
                         g_idle_add(update_history_list, g_strndup((const char*)in, len));
                    } else if (strcmp(type, "chat_history") == 0) {
                        gtk_stack_set_visible_child_name(GTK_STACK(widgets->main_stack), "chat");
                        JsonArray *messages = json_object_get_array_member(root_obj, "payload");
                        for (guint i = 0; i < json_array_get_length(messages); i++) {
                            JsonObject *msg = json_array_get_object_element(messages, i);
                            const char *content = json_object_get_string_member(msg, "content");
                             MessageData *data = g_slice_new(MessageData);
                            data->chat_feed = widgets->chat_feed;
                            data->message = g_strdup(content);
                            g_idle_add(add_message_to_chat_feed, data);
                        }
                    } else if (strcmp(type, "chunk") == 0) {
                        gtk_stack_set_visible_child_name(GTK_STACK(widgets->main_stack), "chat");
                        const char *content = json_object_get_string_member(root_obj, "payload");
                        MessageData *data = g_slice_new(MessageData);
                        data->chat_feed = widgets->chat_feed;
                        data->message = g_strdup(content);
                        g_idle_add(append_to_assistant_message, data);
                    } else if (strcmp(type, "end") == 0) {
                        last_assistant_message = NULL;
                    } else if (strcmp(type, "error") == 0) {
                        const char *error_message = json_object_get_string_member(root_obj, "payload");
                        show_error_info_bar(widgets, error_message);
                    } else {
                        MessageData *data = g_slice_new(MessageData);
                        data->chat_feed = widgets->chat_feed;
                        data->message = g_strndup((const char*)in, len);
                        g_idle_add(add_message_to_chat_feed, data);
                    }
                }
            }
            g_object_unref(parser);
        }
        break;

    case LWS_CALLBACK_CLIENT_CLOSED:
        lwsl_user("CLIENT_CLOSED\n");
        show_error_info_bar(widgets, "Connection closed");
        wsi = NULL;
        break;

    default:
        break;
    }

    return 0;
}
//...
GtkWidget* create_welcome_screen() {
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 20);
    gtk_widget_set_halign(box, GTK_ALIGN_CENTER);
    gtk_widget_set_valign(box, GTK_ALIGN_CENTER);

    GtkWidget *title = gtk_label_new("Project Zephyrine");
    gtk_widget_add_css_class(title, "welcome-title");
    gtk_box_append(GTK_BOX(box), title);

    GtkWidget *subtitle = gtk_label_new("Your friendly AI assistant");
    gtk_widget_add_css_class(subtitle, "welcome-subtitle");
    gtk_box_append(GTK_BOX(box), subtitle);

    GtkWidget *grid = gtk_grid_new();
    gtk_grid_set_column_spacing(GTK_GRID(grid), 10);
    gtk_grid_set_row_spacing(GTK_GRID(grid), 10);
    gtk_box_append(GTK_BOX(box), grid);

    GtkWidget *button1 = gtk_button_new_with_label("Explain quantum computing in simple terms");
    gtk_grid_attach(GTK_GRID(grid), button1, 0, 0, 1, 1);

    GtkWidget *button2 = gtk_button_new_with_label("Got any creative ideas for a 10 year old’s birthday?");
    gtk_grid_attach(GTK_GRID(grid), button2, 1, 0, 1, 1);

    GtkWidget *button3 = gtk_button_new_with_label("How do I make an HTTP request in Javascript?");
    gtk_grid_attach(GTK_GRID(grid), button3, 0, 1, 1, 1);
    
    GtkWidget *button4 = gtk_button_new_with_label("What is the meaning of life?");
    gtk_grid_attach(GTK_GRID(grid), button4, 1, 1, 1, 1);

    return box;
}

static void
activate (GtkApplication* app,
          gpointer        user_data)
{
  GtkWidget *window;
  GtkWidget *header_bar;
  GtkWidget *main_box;
  GtkWidget *sidebar;
  GtkWidget *user_label;
  GtkWidget *new_chat_button;
  GtkWidget *main_page_button;
  GtkWidget *voice_assistant_button;
  GtkWidget *image_generation_button;
  GtkWidget *knowledge_tuning_button;
  GtkWidget *settings_button;
  GtkWidget *chat_view;
  GtkWidget *sidebar_revealer;
  GtkWidget *menu_button;
  GtkWidget *history_list;
  GtkWidget *main_stack;
  UIWidgets *ui_widgets = g_slice_new(UIWidgets);

  GtkCssProvider *provider = gtk_css_provider_new();
  gtk_css_provider_load_from_path(provider, "style.css");
  gtk_style_context_add_provider_for_display(
      gdk_display_get_default(),
      GTK_STYLE_PROVIDER(provider),
      GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
  );
  g_object_unref(provider);

  window = gtk_application_window_new (app);
  gtk_window_set_title (GTK_WINDOW (window), "Project Zephyrine");
  gtk_window_set_default_size (GTK_WINDOW (window), 1200, 800);

  header_bar = gtk_header_bar_new ();
  gtk_header_bar_set_show_title_buttons (GTK_HEADER_BAR (header_bar), TRUE);
  gtk_window_set_titlebar (GTK_WINDOW (window), header_bar);

  menu_button = gtk_button_new_from_icon_name("open-menu-symbolic");
  gtk_header_bar_pack_start(GTK_HEADER_BAR(header_bar), menu_button);

  GtkWidget *logo = gtk_image_new_from_file("assets/img/ProjectZephy023LogoRenewal.png");
  gtk_header_bar_set_title_widget(GTK_HEADER_BAR(header_bar), logo);

  user_label = gtk_label_new ("Welcome, user");
  gtk_header_bar_pack_end (GTK_HEADER_BAR (header_bar), user_label);

  settings_button = gtk_button_new_from_icon_name("emblem-system-symbolic");
  g_signal_connect(settings_button, "clicked", G_CALLBACK(on_settings_clicked), window);
  gtk_header_bar_pack_end(GTK_HEADER_BAR(header_bar), settings_button);

  main_box = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_window_set_child (GTK_WINDOW (window), main_box);
  
  GtkWidget *info_bar = gtk_info_bar_new();
  GtkWidget *info_label = gtk_label_new("");
  gtk_info_bar_add_child(GTK_INFO_BAR(info_bar), info_label);
  gtk_info_bar_add_button(GTK_INFO_BAR(info_bar), "_Close", GTK_RESPONSE_CLOSE);
  g_signal_connect(info_bar, "response", G_CALLBACK(gtk_info_bar_set_revealed), FALSE);
  ui_widgets->info_bar = info_bar;
  ui_widgets->info_label = info_label;
  gtk_box_prepend(GTK_BOX(main_box), info_bar);

  sidebar_revealer = gtk_revealer_new();
  gtk_revealer_set_transition_type(GTK_REVEALER(sidebar_revealer), GTK_REVEALER_TRANSITION_TYPE_SLIDE_RIGHT);
  gtk_revealer_set_reveal_child(GTK_REVEALER(sidebar_revealer), TRUE);
  gtk_box_append(GTK_BOX(main_box), sidebar_revealer);

  sidebar = gtk_box_new (GTK_ORIENTATION_VERTICAL, 6);
  gtk_widget_set_size_request(sidebar, 250, -1);
  gtk_widget_add_css_class(sidebar, "sidebar");
  gtk_revealer_set_child(GTK_REVEALER(sidebar_revealer), sidebar);

  g_signal_connect(menu_button, "clicked", G_CALLBACK(toggle_sidebar_cb), sidebar_revealer);

  new_chat_button = gtk_button_new_from_icon_name ("document-new-symbolic");
  gtk_box_append (GTK_BOX (sidebar), new_chat_button);

  main_page_button = gtk_button_new_from_icon_name ("go-home-symbolic");
  gtk_box_append (GTK_BOX (sidebar), main_page_button);

  voice_assistant_button = gtk_button_new_from_icon_name ("audio-input-microphone-symbolic");
  gtk_box_append (GTK_BOX (sidebar), voice_assistant_button);
  
  image_generation_button = gtk_button_new_from_icon_name("image-x-generic-symbolic");
  gtk_box_append(GTK_BOX(sidebar), image_generation_button);

  knowledge_tuning_button = gtk_button_new_from_icon_name("document-properties-symbolic");
  gtk_box_append(GTK_BOX(sidebar), knowledge_tuning_button);

  history_list = gtk_list_box_new();
  ui_widgets->history_list = history_list;
  g_signal_connect(history_list, "row-selected", G_CALLBACK(on_chat_selected), ui_widgets);
  gtk_box_append(GTK_BOX(sidebar), history_list);

  GtkWidget *user_button = gtk_button_new_with_label("user@example.com");
  gtk_widget_add_css_class(user_button, "user-button");
  GtkWidget *logout_button = gtk_button_new_from_icon_name("log-out-symbolic");
  GtkWidget *user_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_box_append(GTK_BOX(user_box), user_button);
  gtk_box_append(GTK_BOX(user_box), logout_button);
  gtk_box_append(GTK_BOX(sidebar), user_box);

  main_stack = gtk_stack_new();
  ui_widgets->main_stack = main_stack;
  gtk_widget_set_hexpand(main_stack, TRUE);
  gtk_box_append (GTK_BOX (main_box), main_stack);
  
  // Create Chat View
  chat_view = gtk_stack_new();
  GtkWidget *welcome_screen = create_welcome_screen();
  gtk_stack_add_named(GTK_STACK(chat_view), welcome_screen, "welcome");
  ui_widgets->welcome_screen = welcome_screen;

  GtkWidget *chat_scrolled_window = gtk_scrolled_window_new ();
  gtk_widget_set_vexpand (chat_scrolled_window, TRUE);
  GtkWidget *chat_feed = gtk_box_new (GTK_ORIENTATION_VERTICAL, 6);
  gtk_widget_add_css_class(chat_feed, "chat-feed");
  gtk_scrolled_window_set_child (GTK_SCROLLED_WINDOW (chat_scrolled_window), chat_feed);
  gtk_stack_add_named(GTK_STACK(chat_view), chat_scrolled_window, "chat_feed");
  ui_widgets->chat_feed = chat_feed;
  
  if (gtk_widget_get_first_child(chat_feed) == NULL) {
      gtk_stack_set_visible_child_name(GTK_STACK(chat_view), "welcome");
  }

  GtkWidget *input_box = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 6);
  gtk_widget_add_css_class(input_box, "input-box");
  
  GtkWidget *attach_button = gtk_button_new_from_icon_name("attachment-symbolic");
  g_signal_connect(attach_button, "clicked", G_CALLBACK(on_attach_file_clicked), window);
  gtk_box_append(GTK_BOX(input_box), attach_button);

  GtkWidget *input_entry = gtk_entry_new ();
  gtk_widget_set_hexpand (input_entry, TRUE);
  gtk_box_append (GTK_BOX (input_box), input_entry);
  GtkWidget *send_button = gtk_button_new_with_label ("Send");
  gtk_widget_add_css_class(send_button, "send-button");
  ChatWidgets *widgets = g_slice_new(ChatWidgets);
  widgets->input_entry = input_entry;
  widgets->chat_feed = chat_feed;
  g_signal_connect(send_button, "clicked", G_CALLBACK(send_message_cb), widgets);
  gtk_box_append (GTK_BOX (input_box), send_button);
  
  GtkWidget *chat_view_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
  gtk_box_append(GTK_BOX(chat_view_box), chat_view);
  gtk_box_append(GTK_BOX(chat_view_box), input_box);

  gtk_stack_add_named(GTK_STACK(main_stack), chat_view_box, "chat");

  // Create Image Generation View
  GtkWidget *image_page = create_image_generation_page();
  gtk_stack_add_named(GTK_STACK(main_stack), image_page, "image");

  // Create Knowledge Tuning View
  GtkWidget *knowledge_page = create_knowledge_tuning_page();
  gtk_stack_add_named(GTK_STACK(main_stack), knowledge_page, "knowledge");
  
  // Create Voice Assistant View
  GtkWidget *voice_page = create_voice_assistant_page();
  gtk_stack_add_named(GTK_STACK(main_stack), voice_page, "voice");

  g_signal_connect_swapped(main_page_button, "clicked", G_CALLBACK(gtk_stack_set_visible_child_name), main_stack, "chat");
  g_signal_connect_swapped(image_generation_button, "clicked", G_CALLBACK(gtk_stack_set_visible_child_name), main_stack, "image");
  g_signal_connect_swapped(knowledge_tuning_button, "clicked", G_CALLBACK(gtk_stack_set_visible_child_name), main_stack, "knowledge");
  g_signal_connect_swapped(voice_assistant_button, "clicked", G_CALLBACK(gtk_stack_set_visible_child_name), main_stack, "voice");

  connect_websocket(ui_widgets);

  gtk_window_present (GTK_WINDOW (window));
}


int
main (int    argc,
      char **argv)
{
  GtkApplication *app;
  int status;

  app = gtk_application_new ("com.projectzephyrine.ui", G_APPLICATION_DEFAULT_FLAGS);
  g_signal_connect (app, "activate", G_CALLBACK (activate), NULL);
  status = g_application_run (G_APPLICATION (app), argc, argv);
  g_object_unref (app);

  return status;
}