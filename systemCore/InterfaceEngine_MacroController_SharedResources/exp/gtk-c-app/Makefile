# Simple Makefile for a GTK-4 application

# Compiler
CC = clang

# Executable name
TARGET = zephyrine-gtk

# GTK-4 flags
GTK_CFLAGS = $(shell pkg-config --cflags gtk4)
GTK_LIBS = $(shell pkg-config --libs gtk4)

# libwebsockets flags
LWS_CFLAGS = $(shell pkg-config --cflags libwebsockets)
LWS_LIBS = $(shell pkg-config --libs libwebsockets)

# json-glib flags
JSON_CFLAGS = $(shell pkg-config --cflags json-glib-1.0)
JSON_LIBS = $(shell pkg-config --libs json-glib-1.0)

# Compiler flags
CFLAGS = -Wall -g $(GTK_CFLAGS) $(LWS_CFLAGS) $(JSON_CFLAGS)

# Linker flags
LDFLAGS = $(GTK_LIBS) $(LWS_LIBS) $(JSON_LIBS)

# Source files
SOURCES = main.c

# Object files
OBJECTS = $(SOURCES:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)

run: all
	./$(TARGET)
