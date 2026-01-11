cmake_minimum_required(VERSION 3.10)
project(zephy_gui C)

# Find GTK4
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK4 REQUIRED gtk4)

# Capture full linker flags for GTK4
execute_process(
    COMMAND pkg-config --libs gtk4
    OUTPUT_VARIABLE GTK4_RAW_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find GLib for threading
pkg_check_modules(GLIB REQUIRED glib-2.0)

# Capture full linker flags for GLib
execute_process(
    COMMAND pkg-config --libs glib-2.0
    OUTPUT_VARIABLE GLIB_RAW_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find libcurl
find_package(CURL REQUIRED)

# Add cJSON as a source for now. This will be replaced with proper find_package if available.
# Or I can try to use FetchContent if the user allows for it.
# For now, we will add an empty target to signify its intention.

add_subdirectory(src)