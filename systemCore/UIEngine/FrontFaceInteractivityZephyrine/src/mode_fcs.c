#include "mode_fcs.h"
#include <gtk/gtk.h>
#include <cairo.h>
#include <stdio.h> // For snprintf

// Structure to hold data for FCS display
typedef struct {
    char telemetry_data[256];
} FcsData;

static FcsData current_fcs_data = {"Loading telemetry data..."};

// Function to update FCS data (to be called by network thread)
void mode_fcs_update_data(GtkWidget *drawing_area, const char *data) {
    snprintf(current_fcs_data.telemetry_data, sizeof(current_fcs_data.telemetry_data), "Telemetry Data: %s", data);
    gtk_widget_queue_draw(drawing_area);
}


static void draw_cb(GtkDrawingArea *drawing_area, cairo_t *cr, int width, int height, gpointer user_data) {
    // Brutalist, High-Contrast, "Glass Cockpit" aesthetic
    // Background
    cairo_set_source_rgb(cr, 0.1, 0.1, 0.1); // Dark grey background
    cairo_paint(cr);

    // Grid lines
    cairo_set_source_rgb(cr, 0.2, 0.8, 0.2); // Green grid
    cairo_set_line_width(cr, 1.0);
    for (int i = 0; i < width; i += 50) {
        cairo_move_to(cr, i, 0);
        cairo_line_to(cr, i, height);
    }
    for (int i = 0; i < height; i += 50) {
        cairo_move_to(cr, 0, i);
        cairo_line_to(cr, width, i);
    }
    cairo_stroke(cr);

    // Border
    cairo_set_source_rgb(cr, 0.8, 0.8, 0.2); // Yellow border
    cairo_set_line_width(cr, 3.0);
    cairo_rectangle(cr, 0, 0, width, height);
    cairo_stroke(cr);

    // Text display
    cairo_set_source_rgb(cr, 0.0, 1.0, 0.0); // Bright green text
    cairo_select_font_face(cr, "monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, 24);

    cairo_move_to(cr, 20, 40);
    cairo_show_text(cr, "OMS/FCS Display");

    cairo_move_to(cr, 20, 80);
    cairo_show_text(cr, current_fcs_data.telemetry_data);

    // Example telemetry fields
    cairo_set_font_size(cr, 18);
    cairo_move_to(cr, 20, 120);
    cairo_show_text(cr, "ALTITUDE: 10000 FT");
    cairo_move_to(cr, 20, 150);
    cairo_show_text(cr, "SPEED: 500 KTS");
    cairo_move_to(cr, 20, 180);
    cairo_show_text(cr, "HEADING: 270 DEG");
}

// Placeholder for Mode 1 (OMS/FCS) functionality
GtkWidget *mode_fcs_create_widget(void) {
    GtkWidget *drawing_area = gtk_drawing_area_new();
    gtk_widget_set_size_request(drawing_area, 800, 600); // Example size
    g_signal_connect(G_OBJECT(drawing_area), "draw", G_CALLBACK(draw_cb), NULL);
    return drawing_area;
}