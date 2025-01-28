#include "InterpreterManager.h"

// Define the static variable
std::unique_ptr<py::scoped_interpreter> InterpreterManager::guard = nullptr;

void InterpreterManager::init() {
    if (!guard) {
        guard = std::make_unique<py::scoped_interpreter>();
    }
}
