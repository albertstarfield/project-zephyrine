add_executable(zephy_gui main.c mode_fcs.c mode_chat.c network.c cJSON.c)

target_include_directories(zephy_gui PRIVATE ${GTK4_INCLUDE_DIRS} ${CURL_INCLUDE_DIRS}
    /opt/homebrew/Cellar/glib/2.86.3/include/glib-2.0
    /opt/homebrew/Cellar/glib/2.86.3/lib/glib-2.0/include
    /opt/homebrew/opt/gettext/include
    /opt/homebrew/Cellar/pcre2/10.47/include
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(zephy_gui PRIVATE
    /opt/homebrew/Cellar/gtk4/4.20.3/lib/libgtk-4.dylib
    /opt/homebrew/Cellar/pango/1.57.0_1/lib/libpangocairo-1.0.dylib
    /opt/homebrew/Cellar/pango/1.57.0_1/lib/libpango-1.0.dylib
    /opt/homebrew/Cellar/harfbuzz/12.3.0/lib/libharfbuzz.dylib
    /opt/homebrew/lib/libgdk_pixbuf-2.0.dylib
    /opt/homebrew/Cellar/cairo/1.18.4/lib/libcairo-gobject.dylib
    /opt/homebrew/Cellar/cairo/1.18.4/lib/libcairo.dylib
    /opt/homebrew/Cellar/graphene/1.10.8/lib/libgraphene-1.0.dylib
    /opt/homebrew/Cellar/glib/2.86.3/lib/libgio-2.0.dylib
    /opt/homebrew/Cellar/glib/2.86.3/lib/libgobject-2.0.dylib
    /opt/homebrew/Cellar/glib/2.86.3/lib/libglib-2.0.dylib
    /opt/homebrew/opt/gettext/lib/libintl.dylib
    ${CURL_LIBRARIES}
)

target_compile_options(zephy_gui PRIVATE ${GTK4_CFLAGS} ${GLIB_CFLAGS})
target_compile_definitions(zephy_gui PRIVATE GTK_DISABLE_DEPRECATED GTK_COMPILATION)