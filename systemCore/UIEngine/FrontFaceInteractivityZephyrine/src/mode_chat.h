#ifndef MODE_CHAT_H
#define MODE_CHAT_H

#include <gtk/gtk.h>

GtkWidget *mode_chat_create_widget(void);
void mode_chat_add_message(const char *sender, const char *message);

#endif // MODE_CHAT_H