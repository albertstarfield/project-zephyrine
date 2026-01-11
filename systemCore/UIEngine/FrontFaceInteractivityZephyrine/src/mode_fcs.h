#ifndef MODE_FCS_H
#define MODE_FCS_H

#include <gtk/gtk.h>

GtkWidget *mode_fcs_create_widget(void);
void mode_fcs_update_data(GtkWidget *drawing_area, const char *data);

#endif // MODE_FCS_H